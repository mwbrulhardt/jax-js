import { describe, expect, test as globalTest } from "vitest";

import { backendTypes, init } from "../backend";
import { Array, array } from "./array";

const backendsAvailable = await init(...backendTypes);

describe.each(backendTypes)("backend:%s", (backend) => {
  const skipped = !backendsAvailable.includes(backend);
  const test = globalTest.skipIf(skipped);

  test("can construct Array.zeros()", async () => {
    const ar = Array.zeros([3, 3], { backend });
    expect(ar.shape).toEqual([3, 3]);
    expect(ar.dtype).toEqual("float32");
    expect(await ar.data()).toEqual(
      new Float32Array([0, 0, 0, 0, 0, 0, 0, 0, 0]),
    );
    expect(await ar.transpose().data()).toEqual(
      new Float32Array([0, 0, 0, 0, 0, 0, 0, 0, 0]),
    );
    expect(ar.transpose().dataSync()).toEqual(
      new Float32Array([0, 0, 0, 0, 0, 0, 0, 0, 0]),
    );
  });

  test("can construct Array.ones()", async () => {
    const ar = Array.ones([2, 2], { backend });
    expect(ar.shape).toEqual([2, 2]);
    expect(ar.dtype).toEqual("float32");
    expect(await ar.data()).toEqual(new Float32Array([1, 1, 1, 1]));
  });

  test("can add two arrays", async () => {
    const ar1 = Array.ones([2, 2], { backend });
    const ar2 = Array.ones([2, 2], { backend });
    const ar3 = ar1.add(ar2);
    expect(ar3.shape).toEqual([2, 2]);
    expect(ar3.dtype).toEqual("float32");
    expect(await ar3.data()).toEqual(new Float32Array([2, 2, 2, 2]));
  });

  test("can construct arrays from data", () => {
    const a = array([1, 2, 3, 4], { backend });
    const b = array([10, 5, 2, -8.5], { backend });
    const c = a.mul(b);
    expect(c.shape).toEqual([4]);
    expect(c.dtype).toEqual("float32");
    expect(c.dataSync()).toEqual(new Float32Array([10, 10, 6, -34]));
    expect(c.reshape([2, 2]).transpose().dataSync()).toEqual(
      new Float32Array([10, 6, 10, -34]),
    );
  });

  test("can add array to itself", () => {
    const a = array([1, 2, 3], { backend });
    // Make sure duplicate references don't trip up the backend.
    const b = a.add(a).add(a);
    expect(b.dataSync()).toEqual(new Float32Array([3, 6, 9]));
  });

  test("can coerce array to primitive", () => {
    const a = array(42, { backend });
    expect(a).toBeCloseTo(42);

    // https://github.com/microsoft/TypeScript/issues/42218
    expect(+(a as any)).toEqual(42);
    expect((a as any) + 1).toEqual(43);
    expect((a as any) ** 2).toEqual(42 ** 2);
  });
});
