import { expect, test } from "vitest";

import { fromNested, toNested } from "./safetensors";

test("toNested() converts flat dictionary to nested object", () => {
  const b0 = 1;
  const w0 = 2;
  const b1 = 3;
  const w1 = 4;

  const flat = {
    "layers.0.bias": b0,
    "layers.0.weight": w0,
    "layers.1.bias": b1,
    "layers.1.weight": w1,
  };

  const nested = toNested(flat);

  expect(nested).toEqual({
    layers: [
      { bias: b0, weight: w0 },
      { bias: b1, weight: w1 },
    ],
  });
});

test("fromNested() converts nested object to flat dictionary", () => {
  const b0 = 100;
  const w0 = 200;
  const b1 = 312;
  const w1 = 434;

  const nested = {
    layers: [
      { bias: b0, weight: w0 },
      { bias: b1, weight: w1 },
    ],
  };

  const flat = fromNested(nested);

  expect(flat).toEqual({
    "layers.0.bias": b0,
    "layers.0.weight": w0,
    "layers.1.bias": b1,
    "layers.1.weight": w1,
  });
});
