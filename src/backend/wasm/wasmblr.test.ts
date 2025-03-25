import { suite, test, expect } from "vitest";
import { CodeGenerator } from "./wasmblr";

suite("CodeGenerator", () => {
  test("assembles the add() function", async () => {
    const cg = new CodeGenerator();

    const addFunc = cg.function([cg.f32, cg.f32], [cg.f32], () => {
      cg.local.get(0);
      cg.local.get(1);
      cg.f32.add();
    });
    cg.export(addFunc, "add");

    const wasmBytes = cg.finish();
    const { instance } = await WebAssembly.instantiate(wasmBytes);
    const { add } = instance.exports as { add(a: number, b: number): number };

    expect(add(1, 2)).toBe(3);
    expect(add(3.5, 4.6)).toBeCloseTo(8.1, 5);
  });

  test("assembles recursive factorial", async () => {
    const cg = new CodeGenerator();

    const factorialFunc = cg.function([cg.f32], [cg.f32], () => {
      cg.local.get(0);
      cg.f32.const(1.0);
      cg.f32.lt();
      cg.if(cg.f32); // base case
      {
        cg.f32.const(1.0);
      }
      cg.else();
      {
        cg.local.get(0);
        cg.local.get(0);
        cg.f32.const(1.0);
        cg.f32.sub();
        cg.call(factorialFunc);
        cg.f32.mul();
      }
      cg.end();
    });

    cg.export(factorialFunc, "factorial");

    const wasmBytes = cg.finish();
    const { instance } = await WebAssembly.instantiate(wasmBytes);
    const { factorial } = instance.exports as { factorial(x: number): number };

    expect(factorial(0)).toBe(1);
    expect(factorial(1)).toBe(1);
    expect(factorial(2)).toBe(2);
    expect(factorial(3)).toBe(6);
    expect(factorial(7)).toBe(5040);
  });
});
