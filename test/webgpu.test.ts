import { test } from "vitest";
import { webgpu } from "@jax-js/core";

test("eric", async () => {
  const adapter = (await navigator.gpu.requestAdapter())!;
  const device = (await adapter.requestDevice())!;

  const backend = new webgpu.WebGPUBackend(device);

  const a = backend.createBuffer(3 * 4, { mapped: true });
  const b = backend.createBuffer(3 * 4, { mapped: true });
  const c = backend.createBuffer(3 * 4);

  new Float32Array(a.getMappedRange()).set([1, 2, 3]);
  new Float32Array(b.getMappedRange()).set([4, 5, 6]);

  a.unmap();
  b.unmap();

  await backend.executeOperation(webgpu.Operation.Mul, [a, b], [c]);

  console.log("result:", await backend.readBuffer(c));
});
