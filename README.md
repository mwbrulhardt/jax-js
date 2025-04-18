# jax-js

Under construction.

```bash
npm install
npm run build:watch
npm test
```

## Next on Eric's mind

- How to do optimizations?? map out the plan
- Think about two-stage `cumsum()`
- Think about [`Symbol.toPrimitive`](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/Symbol/toPrimitive)
- Disposal by enforcing a `.ref` getter? This isn't included in console.log, but it's included in `{...spread}` syntax, hopefully not used often with arrays.
  ```js
  let f = (x) => x.ref.mul(x.ref).mul(x);
  let df = grad(f); // d/dx (x^3) = 3x^2
  expect(df(4)).toBeAllclose(48);
  expect(df(5)).toBeAllclose(75);
  ```

## Milestones

- [x] It works!
- [x] Demos: Browser REPL / editor
- [x] First custom kernel
- [x] Custom WebGPU backend, removing tfjs dependency
  - [x] Low-level operations
  - [x] Create `class Array {}` wrappers
  - [x] Reduction operations
- [ ] Kernel tuning
  - [ ] "Group" optimizations
  - [ ] "Unroll" optimizations
  - [ ] "Upcast" optimizations (i.e., Wasm SIMD)
- [ ] We figure out the `dispose()` / refcount / linear types stuff
- [ ] Demos: Navier-Stokes, neural networks, statistics
- [ ] Wasm backend (needs malloc)
- [ ] Device switching with `.to()` between webgpu/cpu/wasm
- [ ] `jit()` support via Jaxprs and kernel fusion
- [ ] Other dtypes like int32 and bool
- [ ] numpy/jax API compatibility table
- [ ] Import tfjs models
