<script lang="ts">
  import { base } from "$app/paths";
</script>

<svelte:head>
  <title>jax-js</title>
</svelte:head>

<main class="mx-auto my-12 px-4 sm:px-6 max-w-screen-md">
  <div class="mb-10">
    <h1 class="text-6xl text-center font-bold mb-4">jax-js</h1>
    <p class="italic text-center">Numerical and GPU computing on the web</p>
  </div>

  <p class="mb-6">
    Machine learning belongs in the browser—instantly accessible, and running on
    local GPUs. But existing libraries are slow and weren't <em>designed for</em
    > the web.
  </p>

  <pre class="mb-6 text-center"><code>npm install @jax-js/core</code></pre>

  <p class="mb-6">
    <code>jax-js</code> brings high-performance arrays and
    <a
      class="link"
      target="_blank"
      href="https://en.wikipedia.org/wiki/Automatic_differentiation">autograd</a
    > to JavaScript. So you can explore neural networks (AI), statistics, image processing,
    and computational science.
  </p>

  <div class="mb-6 grid grid-cols-2 sm:grid-cols-4 gap-2 sm:gap-4">
    <!-- TODO: Images -->
    <div class="aspect-video bg-gray-500">
      <span class="text-white text-xs italic">julia set</span>
    </div>
    <div class="aspect-video bg-gray-500">
      <span class="text-white text-xs italic">mnist</span>
    </div>
    <div class="aspect-video bg-gray-500">
      <span class="text-white text-xs italic">llama chatbot</span>
    </div>
    <div class="aspect-video bg-gray-500">
      <span class="text-white text-xs italic">bayesian regression</span>
    </div>
  </div>

  <p class="mb-6">
    Like the Python libraries it's inspired by (NumPy, PyTorch, JAX), <code
      >jax-js</code
    >
    is carefully hand-optimized, including a JIT compiler and scheduler for GPU kernels.
    But it's also
    <strong>extremely simple and portable</strong>, running on Chrome, Firefox,
    Safari, iOS, and Android. On each platform, the compiler generates
    specialized kernels for a variety of hardware.
  </p>

  <div class="mb-6 border h-48 flex items-center justify-center square-grid">
    <p class="text-sm italic text-center">
      <!-- TODO: Insert graph -->
      Performance graph of flops: cpu, wasm, webgl, webgpu
    </p>
  </div>

  <p class="mb-6">
    How is this possible? In short: we compile operations to shaders running in
    WebAssembly, WebGL, and WebGPU on modern browsers. With some performance
    engineering, we can get close to native speed, despite being a featherweight
    cross-platform API.
  </p>

  <p class="mb-6">
    <span class="italic"
      >XXX (Performance benchmarks coming soon—though expect <code>jax-js</code>
      to be faster than TensorFlow.js even at running TensorFlow.js models due to
      JIT and kernel efficiency.)</span
    >
  </p>

  <p class="mb-6">
    Running GPU code should be as easy as making a website. With <code
      >jax-js</code
    >, you can bring neural networks to your users in the browser, with zero
    installs or setup.
  </p>

  <p>
    Try it in the
    <a href="{base}/repl" class="link">jax-js REPL</a>.
  </p>
</main>

<style lang="postcss">
  @reference "$app.css";

  .square-grid {
    background-size: 40px 40px;
    background-image:
      linear-gradient(to right, var(--color-gray-200) 1px, transparent 1px),
      linear-gradient(to bottom, var(--color-gray-200) 1px, transparent 1px);
    background-position: center center;
  }

  a.link {
    @apply underline;
  }
</style>
