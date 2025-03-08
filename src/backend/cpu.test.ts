import { expect, test } from "vitest";
import { getBackend, BackendOp } from "../backend";

test("can run cpu operations", async ({ skip }) => {
  const backend = await getBackend("cpu");
  if (!backend) {
    // Not all environments support WebGPU, especially in CI.
    return skip();
  }

  const a = backend.malloc(3 * 4, new Float32Array([1, 2, 3]).buffer);
  const b = backend.malloc(3 * 4, new Float32Array([4, 5, 6]).buffer);
  const c = backend.malloc(3 * 4);

  try {
    await backend.executeOp(BackendOp.Mul, [a, b], [c]);
    const buf = await backend.read(c);
    expect(new Float32Array(buf)).toEqual(new Float32Array([4, 10, 18]));

    await backend.executeOp(BackendOp.Add, [a, b], [c]);
    const buf2 = await backend.read(c);
    expect(new Float32Array(buf2)).toEqual(new Float32Array([5, 7, 9]));
  } finally {
    backend.decRef(a);
    backend.decRef(b);
    backend.decRef(c);
  }
});
