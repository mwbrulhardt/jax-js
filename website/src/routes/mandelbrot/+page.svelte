<script lang="ts">
  import { init, jit, numpy as np, setBackend } from "@jax-js/jax";
  import { onMount } from "svelte";

  const width = 1000;
  const height = 800;

  onMount(async () => {
    await init("webgpu");
    setBackend("webgpu");
  });

  function mandelbrotIteration(
    A: np.Array,
    B: np.Array,
    V: np.Array,
    X: np.Array,
    Y: np.Array,
  ) {
    const Asq = A.ref.mul(A.ref);
    const Bsq = B.ref.mul(B.ref);
    V = V.add(np.where(Asq.ref.add(Bsq.ref).less(100), 1, 0));
    const A2 = np.clip(Asq.sub(Bsq).add(X), -50, 50);
    const B2 = np.clip(A.mul(B).mul(2).add(Y), -50, 50);
    return [A2, B2, V];
  }

  const mandelbrotMultiple = (iters: number) =>
    jit((A: np.Array, B: np.Array, V: np.Array, X: np.Array, Y: np.Array) => {
      for (let i = 0; i < iters; i++) {
        [A, B, V] = mandelbrotIteration(A, B, V, X.ref, Y.ref);
      }
      X.dispose();
      Y.dispose();
      return [A, B, V];
    });

  function calculateMandelbrot(iters: number) {
    const x = np.linspace(-2, 0.5, width);
    const y = np.linspace(-1, 1, height);

    const [X, Y] = np.meshgrid([x, y]);

    const f = jit(mandelbrotIteration);

    let A = np.zeros(X.shape);
    let B = np.zeros(Y.shape);
    let V = np.zeros(X.shape);
    for (let i = 0; i < iters; i++) {
      console.log(`Iteration ${i + 1}/${iters}`);
      [A, B, V] = f(A, B, V, X.ref, Y.ref);
    }
    X.dispose();
    Y.dispose();
    A.dispose();
    B.dispose();

    return V;
  }

  function calculateMandelbrotJit10(iters: number) {
    const x = np.linspace(-2, 0.5, width);
    const y = np.linspace(-1, 1, height);

    const [X, Y] = np.meshgrid([x, y]);

    const f = mandelbrotMultiple(10);

    let A = np.zeros(X.shape);
    let B = np.zeros(Y.shape);
    let V = np.zeros(X.shape);
    for (let i = 0; i < iters / 10; i++) {
      console.log(`Iteration ${i + 1}/${iters / 10}`);
      [A, B, V] = f(A, B, V, X.ref, Y.ref);
    }
    X.dispose();
    Y.dispose();
    A.dispose();
    B.dispose();

    return V;
  }

  let canvas: HTMLCanvasElement;

  function renderMandelbrot(result: Int32Array) {
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const imageData = ctx.createImageData(canvas.width, canvas.height);
    const data = imageData.data;

    for (let i = 0; i < result.length; i++) {
      const value = 255 * (result[i] / 100);
      data[i * 4] = value; // Red
      data[i * 4 + 1] = value; // Green
      data[i * 4 + 2] = value; // Blue
      data[i * 4 + 3] = 255; // Alpha
    }

    ctx.putImageData(imageData, 0, 0);
  }
</script>

<main class="p-4">
  <h1 class="text-2xl mb-2">mandelbrot</h1>

  <button
    onclick={async () => {
      const start = performance.now();
      const result = (await calculateMandelbrot(100).data()) as Int32Array;
      console.log(`Mandelbrot calculated in ${performance.now() - start} ms`);
      renderMandelbrot(result);
    }}
  >
    Calculate Mandelbrot
  </button>

  <button
    onclick={async () => {
      const start = performance.now();
      const result = (await calculateMandelbrotJit10(100).data()) as Int32Array;
      console.log(`Mandelbrot calculated in ${performance.now() - start} ms`);
      renderMandelbrot(result);
    }}
  >
    Calculate Mandelbrot Jit10
  </button>

  <canvas bind:this={canvas} {width} {height} class="my-8"></canvas>
</main>

<style lang="postcss">
  @reference "$app.css";

  button {
    @apply border px-2 hover:bg-gray-100 active:scale-95;
  }
</style>
