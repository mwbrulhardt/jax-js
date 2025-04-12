import { AluExp, AluGroup, AluOp, DType, Kernel } from "../alu";
import { Backend, BackendType, Executable, Slot, SlotError } from "../backend";
import { DEBUG } from "../utils";

type ShaderDispatch = {
  pipeline: GPUComputePipeline;
  grid: [number, number];
};

/** Implementation of `Backend` that uses WebGPU in browsers. */
export class WebGPUBackend implements Backend {
  type: BackendType = "webgpu";

  readonly pipelines: ShaderPipelineCache;
  readonly syncReader: SyncReader;
  readonly buffers: Map<Slot, { ref: number; buffer: GPUBuffer }>;
  nextSlot: number;

  constructor(readonly device: GPUDevice) {
    if (DEBUG >= 3) {
      console.info(
        "webgpu adapter:",
        device.adapterInfo.vendor,
        device.adapterInfo.architecture,
      );
    }
    this.pipelines = new ShaderPipelineCache(device);
    this.syncReader = new SyncReader(device);
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
      // The GPUBuffer.destroy() method does not actually free the memory until
      // pending work is done.
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
    const buffer = this.#getBuffer(slot);
    if (start === undefined) start = 0;
    if (count === undefined) count = buffer.size - start;
    return this.syncReader.read(buffer, start, count);
  }

  async prepare(kernel: Kernel): Promise<Executable<ShaderDispatch>> {
    const { shader, grid } = pipelineSource(this.device, kernel);
    const pipeline = await this.pipelines.prepare(shader);
    return new Executable(kernel, { pipeline, grid });
  }

  prepareSync(kernel: Kernel): Executable<ShaderDispatch> {
    const { shader, grid } = pipelineSource(this.device, kernel);
    const pipeline = this.pipelines.prepareSync(shader);
    return new Executable(kernel, { pipeline, grid });
  }

  dispatch(
    exe: Executable<ShaderDispatch>,
    inputs: Slot[],
    outputs: Slot[],
  ): void {
    const inputBuffers = inputs.map((slot) => this.#getBuffer(slot));
    const outputBuffers = outputs.map((slot) => this.#getBuffer(slot));
    pipelineSubmit(this.device, exe.data, inputBuffers, outputBuffers);
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

function dtypeToWgsl(dtype: DType): string {
  switch (dtype) {
    case DType.Bool:
      return "bool";
    case DType.Int32:
      return "i32";
    case DType.Float32:
      return "f32";
    default:
      throw new Error(`Unsupported dtype: ${dtype}`);
  }
}

function constToWgsl(dtype: DType, value: any): string {
  if (dtype === DType.Bool) return value ? "true" : "false";
  if (dtype === DType.Int32) return value.toString();
  if (dtype === DType.Float32) {
    let s = value.toString();
    if (!s.includes(".")) s += ".0";
    return s;
  }
  throw new Error(`Unsupported const dtype: ${dtype}`);
}

/**
 * Compiles an expression into WebGPU shader source code.
 *
 * Returns the shader source and the number of workgroups to dispatch along x
 * and y axes, to run the kernel.
 */
function pipelineSource(
  device: GPUDevice,
  kernel: Kernel,
): { shader: string; grid: [number, number] } {
  let { exp, nargs } = kernel;

  exp = exp.simplify();
  const args = Array.from({ length: nargs }, (_, i) => `in${i}`);

  // binding(0..n-1): input buffers
  // binding(n): output buffer

  const shader: string[] = []; // line-separated

  for (let i = 0; i < nargs; i++) {
    shader.push(
      `@group(0) @binding(${i}) var<storage, read> ${args[i]} : array<f32>;`,
    );
  }
  shader.push(
    `@group(0) @binding(${nargs}) var<storage, read_write> result : array<f32>;`,
  );

  const workgroupSize = device.limits.maxComputeWorkgroupSizeX;

  shader.push(
    `\n@compute @workgroup_size(${workgroupSize})`,
    "fn main(@builtin(global_invocation_id) id : vec3<u32>) {",
    `  if (id.x >= ${kernel.size}) { return; }`,
    "  let gidx: i32 = i32(id.x);",
  );

  // Generate code for each AluExp operation.
  // Some expressions may be used twice, so we keep track of them.
  let gensymCount = 0;
  const gensym = () => `alu${gensymCount++}`;

  const usedArgs = Array.from({ length: nargs }, () => false);
  const references = new Map<AluExp, number>();
  const seen = new Set<AluExp>();
  const countReferences = (exp: AluExp) => {
    references.set(exp, (references.get(exp) ?? 0) + 1);
    if (exp.op === AluOp.GlobalIndex) usedArgs[exp.arg] = true;
    if (!seen.has(exp)) {
      seen.add(exp);
      for (const src of exp.src) countReferences(src);
    }
  };
  countReferences(exp);

  // Insert phony assignments for inputs that are not in use.
  // https://github.com/gpuweb/gpuweb/discussions/4582#discussioncomment-9146686
  for (let i = 0; i < args.length; i++) {
    if (!usedArgs[i]) shader.push(`  _ = &${args[i]};`);
  }

  const expContext = new Map<AluExp, string>();
  const gen = (exp: AluExp): string => {
    if (expContext.has(exp)) return expContext.get(exp)!;
    const { op, src, dtype, arg } = exp;

    // Some of these cases early `return` to force-inline them.
    let source = "";
    if (AluGroup.Binary.has(op) || AluGroup.Compare.has(op)) {
      const a = gen(src[0]);
      const b = gen(src[1]);
      if (op === AluOp.Add) source = `(${a} + ${b})`;
      else if (op === AluOp.Sub) source = `(${a} - ${b})`;
      else if (op === AluOp.Mul) source = `(${a} * ${b})`;
      else if (op === AluOp.Idiv)
        source = dtype === DType.Int32 ? `(${a} / ${b})` : `floor(${a} / ${b})`;
      else if (op === AluOp.Mod) source = `(${a} % ${b})`;
      else if (op === AluOp.Min) source = `min(${a}, ${b})`;
      else if (op === AluOp.Max) source = `max(${a}, ${b})`;
      else if (op === AluOp.Cmplt) source = `(${a} < ${b})`;
      else if (op === AluOp.Cmpne) source = `(${a} != ${b})`;
    } else if (AluGroup.Unary.has(op)) {
      const a = gen(src[0]);
      if (op === AluOp.Sin) source = `sin(${a})`;
      else if (op === AluOp.Cos) source = `cos(${a})`;
      else if (op === AluOp.Cast) source = `${dtypeToWgsl(dtype)}(${a})`;
    } else if (op === AluOp.Where) {
      // select(f, t, cond) -> cond ? t : f
      source = `select(${gen(src[2])}, ${gen(src[1])}, ${gen(src[0])})`;
    } else if (op === AluOp.Const) {
      return constToWgsl(dtype, arg);
    } else if (op === AluOp.Special) {
      return arg[0] as string;
    } else if (op === AluOp.Variable) {
      return arg as string;
    } else if (op === AluOp.GlobalIndex) {
      source = `${args[arg]}[${gen(src[0])}]`;
    }

    if (!source) throw new Error(`Missing impl for op: ${op}`);
    const typeName = dtypeToWgsl(dtype);
    if ((references.get(exp) ?? 0) > 1) {
      const name = gensym();
      expContext.set(exp, name);
      shader.push(`  let ${name}: ${typeName} = ${source};`);
      return name;
    } else {
      expContext.set(exp, source);
      return source;
    }
  };

  shader.push(`  result[gidx] = ${gen(exp)};`, "}");
  return {
    shader: shader.join("\n"),
    grid: [Math.ceil(kernel.size / workgroupSize), 1],
  };
}

function pipelineSubmit(
  device: GPUDevice,
  { pipeline, grid }: ShaderDispatch,
  inputs: GPUBuffer[],
  outputs: GPUBuffer[],
) {
  if (
    inputs.length + outputs.length >
    device.limits.maxStorageBuffersPerShaderStage
  ) {
    // This is a hard limit in WebGPU. All platforms have at least 8 storage
    // buffers per shader stage, and >99% support 10. If you pass more than this
    // many inputs then you risk running into this limit.
    const actual = inputs.length + outputs.length;
    const max = device.limits.maxStorageBuffersPerShaderStage;
    throw new Error(
      `Too many buffers (${actual}) for WebGPU pipeline (max: ${max})`,
    );
  }

  const bindGroup = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      ...inputs.map((buffer, i) => {
        return { binding: i, resource: { buffer } };
      }),
      { binding: inputs.length, resource: { buffer: outputs[0] } },
    ],
  });

  const workgroupSize = device.limits.maxComputeWorkgroupSizeX;

  const commandEncoder = device.createCommandEncoder();
  const passEncoder = commandEncoder.beginComputePass();
  passEncoder.setPipeline(pipeline);
  passEncoder.setBindGroup(0, bindGroup);
  passEncoder.dispatchWorkgroups(grid[0], grid[1]);
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

  async prepare(code: string): Promise<GPUComputePipeline> {
    const existingPipeline = this.cache.get(code);
    if (existingPipeline) return existingPipeline;

    const existingPromise = this.inProgress.get(code);
    if (existingPromise) return await existingPromise;

    if (DEBUG >= 2) {
      console.info("=========== WebGPU shader ===========\n" + code);
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
        const emsg = await compileError(shaderModule, scope, code);
        throw new Error(emsg);
      }
    })();
    this.inProgress.set(code, promise);

    // This could race against getSync(), but it's okay since shader pipeline
    // creation is deterministic + idempotent.
    const pipeline = await promise;
    this.cache.set(code, pipeline);
    return pipeline;
  }

  prepareSync(code: string): GPUComputePipeline {
    const existingPipeline = this.cache.get(code);
    if (existingPipeline) return existingPipeline;

    if (DEBUG >= 2) {
      console.info("=========== WebGPU shader ===========\n" + code);
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
    this.device.popErrorScope().then(async (scope) => {
      // This happens asynchronously, so we can't throw here. But shader syntax
      // validation errors should never occur in correct code. Any issues here
      // reflect bugs in jax-js.
      if (scope !== null) {
        const emsg = await compileError(shaderModule, scope, code);
        console.error(emsg);
      }
    });
    this.cache.set(code, pipeline);
    return pipeline;
  }
}

/** Gather information about a compilation error and format it. */
async function compileError(
  shaderModule: GPUShaderModule,
  scope: GPUError | null,
  code: string,
): Promise<string> {
  let message = `Failed to compile shader: ${scope ? scope.message : "(no error scope)"}`;
  const info = await shaderModule.getCompilationInfo();
  for (const msg of info.messages) {
    message += `\n  [${msg.type} at ${msg.lineNum}:${msg.linePos}] ${msg.message}`;
  }
  if (code) {
    message += `\n\n${code}`;
  }
  return message;
}

/**
 * Graphics state used to synchronously read data from WebGPU buffers.
 *
 * This trick is borrowed from TensorFlow.js. Basically, the idea is to create
 * an offscreen canvas with one pixel for every 4 bytes ("device storage"), then
 * configure it with a WebGPU context. Copy the buffer to a texture, then draw
 * the canvas onto another offscreen canvas with '2d' context ("host storage").
 *
 * Once it's on host storage, we can use `getImageData()` to read the pixels
 * from the image directly.
 *
 * We use 256x256 canvases here (256 KiB). The performance of this is bad
 * because it involves multiple data copies, but it still works. We also
 * actually need to copy the image twice: once in "opaque" mode for the RGB
 * values, and once in "premultiplied" mode for the alpha channel.
 *
 * https://github.com/tensorflow/tfjs/blob/tfjs-v4.22.0/tfjs-backend-webgpu/src/backend_webgpu.ts#L379
 */
class SyncReader {
  static readonly alphaModes: GPUCanvasAlphaMode[] = [
    "opaque",
    "premultiplied",
  ];
  static readonly width = 256;
  static readonly height = 256;

  initialized = false;
  deviceStorage?: OffscreenCanvas[];
  deviceContexts?: GPUCanvasContext[];
  hostStorage?: OffscreenCanvas;
  hostContext?: OffscreenCanvasRenderingContext2D;

  constructor(readonly device: GPUDevice) {}

  #init() {
    const makeCanvas = () =>
      new OffscreenCanvas(SyncReader.width, SyncReader.height);
    this.deviceStorage = SyncReader.alphaModes.map(makeCanvas);
    this.deviceContexts = this.deviceStorage.map((canvas, i) => {
      const context = canvas.getContext("webgpu")!;
      context.configure({
        device: this.device,
        // rgba8unorm is not supported on Chrome for macOS.
        // https://bugs.chromium.org/p/chromium/issues/detail?id=1298618
        format: "bgra8unorm",
        usage: GPUTextureUsage.COPY_DST,
        alphaMode: SyncReader.alphaModes[i],
      });
      return context;
    });
    this.hostStorage = makeCanvas();
    this.hostContext = this.hostStorage.getContext("2d", {
      willReadFrequently: true,
    })!;
    this.initialized = true;
  }

  read(buffer: GPUBuffer, start: number, count: number): ArrayBuffer {
    if (!this.initialized) this.#init();

    if (count % 4 !== 0) {
      throw new Error("Read size must be a multiple of 4 bytes");
    }

    const deviceStorage = this.deviceStorage!;
    const deviceContexts = this.deviceContexts!;
    const hostContext = this.hostContext!;

    const pixelsSize = count / 4;
    const bytesPerRow = SyncReader.width * 4;
    const valsGPU = new ArrayBuffer(count);

    for (let i = 0; i < deviceContexts.length; i++) {
      const texture = deviceContexts[i].getCurrentTexture();
      // Read data using a (width, height) image at `offset` in valsGPU.
      const readData = (width: number, height: number, offset: number) => {
        const encoder = this.device.createCommandEncoder();
        encoder.copyBufferToTexture(
          { buffer, bytesPerRow, offset: offset + start },
          { texture },
          { width, height, depthOrArrayLayers: 1 },
        );
        const commandBuffer = encoder.finish();
        this.device.queue.submit([commandBuffer]);

        hostContext.clearRect(0, 0, width, height);
        hostContext.drawImage(deviceStorage[i], 0, 0);
        const values = hostContext.getImageData(0, 0, width, height).data;
        const span = new Uint8ClampedArray(valsGPU, offset, 4 * width * height);
        const alphaMode = SyncReader.alphaModes[i];
        for (let k = 0; k < span.length; k += 4) {
          if (alphaMode === "premultiplied") {
            span[k + 3] = values[k + 3];
          } else {
            span[k] = values[k + 2]; // opaque (BGRA)
            span[k + 1] = values[k + 1];
            span[k + 2] = values[k];
          }
        }
      };

      const pixelsPerCanvas = SyncReader.width * SyncReader.height;
      const wholeChunks = Math.floor(pixelsSize / pixelsPerCanvas);
      let remainder = pixelsSize % pixelsPerCanvas;
      const remainderRows = Math.floor(remainder / SyncReader.width);
      remainder = remainder % SyncReader.width;

      let offset = 0;
      // Read entire canvases.
      for (let j = 0; j < wholeChunks; j++) {
        readData(SyncReader.width, SyncReader.height, offset);
        offset += pixelsPerCanvas * 4;
      }
      // Read a partial canvas with whole rows.
      if (remainderRows > 0) {
        readData(SyncReader.width, remainderRows, offset);
        offset += remainderRows * SyncReader.width * 4;
      }
      // Read a partial canvas with some columns in the first row.
      if (remainder > 0) readData(remainder, 1, offset);
    }

    return valsGPU;
  }
}
