export enum Operation { // TODO: This is temporary
  Add,
  Mul,
}

export class WebGPUBackend {
  constructor(readonly device: GPUDevice) {
    console.info(
      "webgpu adapter:",
      device.adapterInfo.vendor,
      device.adapterInfo.architecture,
    );
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
  createBuffer(size: number, { mapped = false, read = false } = {}): GPUBuffer {
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

  async executeOperation(
    op: Operation,
    inputs: GPUBuffer[],
    outputs: GPUBuffer[],
  ) {
    if (op === Operation.Add) {
      const pipeline = await this.compileShader(`
@group(0) @binding(0) var<storage, read> arrayA : array<f32>;
@group(0) @binding(1) var<storage, read> arrayB : array<f32>;
@group(0) @binding(2) var<storage, read_write> result : array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id : vec3<u32>) {
  if (id.x < arrayLength(&arrayA)) {
    result[id.x] = arrayA[id.x] + arrayB[id.x];
  }
}`);

      const bindGroup = this.device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: inputs[0] } },
          { binding: 1, resource: { buffer: inputs[1] } },
          { binding: 2, resource: { buffer: outputs[0] } },
        ],
      });

      const commandEncoder = this.device.createCommandEncoder();
      const passEncoder = commandEncoder.beginComputePass();
      passEncoder.setPipeline(pipeline);
      passEncoder.setBindGroup(0, bindGroup);
      passEncoder.dispatchWorkgroups(Math.ceil(inputs[0].size / 64));
      passEncoder.end();
      this.device.queue.submit([commandEncoder.finish()]);
    } else if (op === Operation.Mul) {
      const pipeline = await this.compileShader(`
@group(0) @binding(0) var<storage, read> arrayA : array<f32>;
@group(0) @binding(1) var<storage, read> arrayB : array<f32>;
@group(0) @binding(2) var<storage, read_write> result : array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id : vec3<u32>) {
  if (id.x < arrayLength(&arrayA)) {
    result[id.x] = arrayA[id.x] * arrayB[id.x];
  }
}`);

      const bindGroup = this.device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: inputs[0] } },
          { binding: 1, resource: { buffer: inputs[1] } },
          { binding: 2, resource: { buffer: outputs[0] } },
        ],
      });

      const commandEncoder = this.device.createCommandEncoder();
      const passEncoder = commandEncoder.beginComputePass();
      passEncoder.setPipeline(pipeline);
      passEncoder.setBindGroup(0, bindGroup);
      passEncoder.dispatchWorkgroups(Math.ceil(inputs[0].size / 64));
      passEncoder.end();
      this.device.queue.submit([commandEncoder.finish()]);
    } else {
      throw new Error(`Unknown operation: ${op}`);
    }
  }

  async compileShader(code: string): Promise<GPUComputePipeline> {
    this.device.pushErrorScope("validation");
    try {
      const shaderModule = this.device.createShaderModule({
        code,
      });
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
      const scope = await this.device.popErrorScope();
      throw new Error(`Failed to compile shader: ${scope?.message}\n${code}`);
    }
  }

  async readBuffer(buffer: GPUBuffer): Promise<Float32Array> {
    let createdStaging = false;
    let staging: GPUBuffer;
    if (buffer.usage & GPUBufferUsage.MAP_READ) {
      staging = buffer;
    } else {
      createdStaging = true;
      staging = this.createBuffer(buffer.size, { read: true });
      const commandEncoder = this.device.createCommandEncoder();
      commandEncoder.copyBufferToBuffer(buffer, 0, staging, 0, buffer.size);
      this.device.queue.submit([commandEncoder.finish()]);
    }

    try {
      await staging.mapAsync(GPUMapMode.READ);
      const arrayBuffer = staging.getMappedRange();
      const data = new Float32Array(arrayBuffer);
      return data.slice();
    } finally {
      if (createdStaging) {
        staging.destroy();
      }
    }
  }
}
