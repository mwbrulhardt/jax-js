import { Backend, BackendOp, Slot, SlotError } from "../backend";
import { DEBUG } from "../utils";

/** Implementation of `Backend` that uses WebGPU in browsers. */
export class WebGPUBackend implements Backend {
  readonly pipelines: ShaderPipelineCache;
  readonly buffers: Map<Slot, { ref: number; buffer: GPUBuffer }>;
  nextSlot: number;

  constructor(readonly device: GPUDevice) {
    if (DEBUG) {
      console.info(
        "webgpu adapter:",
        device.adapterInfo.vendor,
        device.adapterInfo.architecture,
      );
    }
    this.pipelines = new ShaderPipelineCache(device);
    this.buffers = new Map();
    this.nextSlot = 1;
  }

  malloc(size: number, initialData?: ArrayBuffer): Slot {
    let buffer: GPUBuffer;
    if (initialData) {
      if (initialData.byteLength !== size) {
        throw new Error("initialData size does not match buffer size");
      }
      buffer = this.#createBuffer(size, { mapped: true });
      new Uint8Array(buffer.getMappedRange()).set(new Uint8Array(initialData));
      buffer.unmap();
    } else {
      buffer = this.#createBuffer(size);
    }

    const slot = this.nextSlot++;
    this.buffers.set(slot, { buffer, ref: 1 });
    return slot;
  }

  incRef(slot: Slot): void {
    const buffer = this.buffers.get(slot);
    if (!buffer) throw new SlotError(slot);
    buffer.ref++;
  }

  decRef(slot: Slot): void {
    const buffer = this.buffers.get(slot);
    if (!buffer) throw new SlotError(slot);
    buffer.ref--;
    if (buffer.ref === 0) {
      this.buffers.delete(slot);
      buffer.buffer.destroy();
    }
  }

  async read(slot: Slot, start?: number, count?: number): Promise<ArrayBuffer> {
    const buffer = this.#getBuffer(slot);
    if (start === undefined) start = 0;
    if (count === undefined) count = buffer.size - start;

    // Need a GPUBuffer with MAP_READ usage when transfering data to host.
    const staging = this.#createBuffer(count, { read: true });
    try {
      const commandEncoder = this.device.createCommandEncoder();
      commandEncoder.copyBufferToBuffer(buffer, start, staging, 0, count);
      this.device.queue.submit([commandEncoder.finish()]);

      await staging.mapAsync(GPUMapMode.READ);
      const arrayBuffer = staging.getMappedRange();
      const data = new Float32Array(arrayBuffer);
      return data.slice();
    } finally {
      staging.destroy();
    }
  }

  readSync(slot: Slot, start?: number, count?: number): ArrayBuffer {
    // TODO: WebGL hack
    // https://github.com/tensorflow/tfjs/blob/2644bd0d6cea677f80e44ed4a44bea5e04aabeb3/tfjs-backend-webgl/src/backend_webgl.ts#L271
    throw new Error("readSync() not implemented for WebGPU");
  }

  async executeOp(
    op: BackendOp,
    inputs: Slot[],
    outputs: Slot[],
  ): Promise<void> {
    const inputBuffers = inputs.map((slot) => this.#getBuffer(slot));
    const outputBuffers = outputs.map((slot) => this.#getBuffer(slot));
    const pipeline = await this.pipelines.get(pipelineSource(op));
    pipelineSubmit(op, this.device, pipeline, inputBuffers, outputBuffers);
  }

  executeOpSync(op: BackendOp, inputs: Slot[], outputs: Slot[]): void {
    const inputBuffers = inputs.map((slot) => this.#getBuffer(slot));
    const outputBuffers = outputs.map((slot) => this.#getBuffer(slot));
    const pipeline = this.pipelines.getSync(pipelineSource(op));
    pipelineSubmit(op, this.device, pipeline, inputBuffers, outputBuffers);
  }

  #getBuffer(slot: Slot): GPUBuffer {
    const buffer = this.buffers.get(slot);
    if (!buffer) throw new SlotError(slot);
    return buffer.buffer;
  }

  /**
   * Create a GPU buffer.
   *
   * By default, this creates a general-purpose buffer with the given size.
   *
   * - If `mapped` is true, initialize the buffer in mapped mode so that it can
   *   be populated with data from the CPU. (Call `.unmap()` later.)
   * - If `read` is true, create a staging buffer for returning data to CPU.
   *   (Call `.mapAsync()` later.)
   */
  #createBuffer(
    size: number,
    { mapped = false, read = false } = {},
  ): GPUBuffer {
    if (read && mapped) {
      throw new Error("mapped and read cannot both be true");
    }
    const buffer = this.device.createBuffer({
      size,
      usage: read
        ? GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST
        : GPUBufferUsage.STORAGE |
          GPUBufferUsage.COPY_SRC |
          GPUBufferUsage.COPY_DST,
      mappedAtCreation: mapped,
    });
    return buffer;
  }
}

function pipelineSource(op: BackendOp) {
  switch (op) {
    case BackendOp.Add:
      return `
@group(0) @binding(0) var<storage, read> arrayA : array<f32>;
@group(0) @binding(1) var<storage, read> arrayB : array<f32>;
@group(0) @binding(2) var<storage, read_write> result : array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id : vec3<u32>) {
  if (id.x < arrayLength(&arrayA)) {
    result[id.x] = arrayA[id.x] + arrayB[id.x];
  }
}`;
    case BackendOp.Mul:
      return `
@group(0) @binding(0) var<storage, read> arrayA : array<f32>;
@group(0) @binding(1) var<storage, read> arrayB : array<f32>;
@group(0) @binding(2) var<storage, read_write> result : array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id : vec3<u32>) {
  if (id.x < arrayLength(&arrayA)) {
    result[id.x] = arrayA[id.x] * arrayB[id.x];
  }
}`;
    default:
      throw new Error(`Unknown operation: ${op}`);
  }
}

function pipelineSubmit(
  op: BackendOp,
  device: GPUDevice,
  pipeline: GPUComputePipeline,
  inputs: GPUBuffer[],
  outputs: GPUBuffer[],
) {
  // TODO: Needs to support other ops later.
  const bindGroup = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: inputs[0] } },
      { binding: 1, resource: { buffer: inputs[1] } },
      { binding: 2, resource: { buffer: outputs[0] } },
    ],
  });

  const commandEncoder = device.createCommandEncoder();
  const passEncoder = commandEncoder.beginComputePass();
  passEncoder.setPipeline(pipeline);
  passEncoder.setBindGroup(0, bindGroup);
  passEncoder.dispatchWorkgroups(Math.ceil(inputs[0].size / 64));
  passEncoder.end();
  device.queue.submit([commandEncoder.finish()]);
}

/**
 * A cache for compiled GPU compute pipelines, keyed by the shader source.
 *
 * This supports both async compilation (recommended) and a synchronous variant.
 * If the pipeline is not in the cache, it will be compiled and added. For async
 * compilation, only one compilation will be in progress at a time for a given
 * shader source.
 */
class ShaderPipelineCache {
  cache: Map<string, GPUComputePipeline>;
  inProgress: Map<string, Promise<GPUComputePipeline>>;

  constructor(readonly device: GPUDevice) {
    this.cache = new Map();
    this.inProgress = new Map();
  }

  async get(code: string): Promise<GPUComputePipeline> {
    const existingPipeline = this.cache.get(code);
    if (existingPipeline) {
      return existingPipeline;
    }

    const existingPromise = this.inProgress.get(code);
    if (existingPromise) {
      return await existingPromise;
    }

    const shaderModule = this.device.createShaderModule({ code });
    const promise = (async () => {
      this.device.pushErrorScope("validation");
      try {
        const pipeline = await this.device.createComputePipelineAsync({
          layout: "auto",
          compute: {
            module: shaderModule,
            entryPoint: "main",
          },
        });
        await this.device.popErrorScope();
        return pipeline;
      } catch (e) {
        // This can race with other compilations, but it shouldn't happen in
        // correct code. Any validation error here is a bug in `jax-js`.
        const scope = await this.device.popErrorScope();
        throw new Error(`Failed to compile shader: ${scope?.message}\n${code}`);
      }
    })();
    this.inProgress.set(code, promise);

    // This could race against getSync(), but it's okay since shader pipeline
    // creation is deterministic + idempotent.
    const pipeline = await promise;
    this.cache.set(code, pipeline);
    return pipeline;
  }

  getSync(code: string): GPUComputePipeline {
    const existingPipeline = this.cache.get(code);
    if (existingPipeline) {
      return existingPipeline;
    }

    const shaderModule = this.device.createShaderModule({ code });
    this.device.pushErrorScope("validation");
    const pipeline = this.device.createComputePipeline({
      layout: "auto",
      compute: {
        module: shaderModule,
        entryPoint: "main",
      },
    });
    this.device.popErrorScope().then((scope) => {
      // This happens asynchronously, so we can't throw here. But shader syntax
      // validation errors should never occur in correct code. Any issues here
      // reflect bugs in jax-js.
      if (scope !== null) {
        console.error(`Failed to compile shader: ${scope.message}\n${code}`);
      }
    });
    this.cache.set(code, pipeline);
    return pipeline;
  }
}
