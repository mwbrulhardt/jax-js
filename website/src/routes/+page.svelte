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

  <p class="mb-5">
    Machine learning belongs in the browser—instantly accessible, and running on
    local GPUs. But existing libraries weren't <em>designed for</em> the web.
  </p>

  <pre class="mb-5 text-center"><code>npm install @jax-js/core</code></pre>

  <p class="mb-5">
    <code>jax-js</code> brings familiar array APIs and
    <a
      class="link"
      target="_blank"
      href="https://en.wikipedia.org/wiki/Automatic_differentiation">autograd</a
    > to JavaScript. So you can explore neural networks (AI), statistics, image processing,
    and computational science, all on the web platform.
  </p>

  <div class="mb-5 grid grid-cols-2 sm:grid-cols-4 gap-2 sm:gap-4">
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

  <p class="mb-5">
    Like the Python libraries it's inspired by (NumPy, PyTorch, JAX), <code
      >jax-js</code
    >
    is carefully hand-optimized, including a JIT compiler and scheduler for GPU kernels.
    But it's also
    <strong>extremely simple and portable across the web</strong>, running on
    Chrome, Firefox, Safari, iOS, and Android. On each platform, the compiler
    generates specialized kernels on a variety of hardware.
  </p>

  <div class="mb-5 border h-48 flex items-center justify-center square-grid">
    <p class="text-sm italic text-center">
      <!-- TODO: Insert graph -->
      Performance graph of flops: cpu, wasm, webgl, webgpu
    </p>
  </div>

  <p class="mb-5">
    How is this possible? In short: because of modern technologies like
    WebAssembly, WebGL, and WebGPU that give us low-level access to hardware, as
    ML models also become smaller and faster.
  </p>

  <p class="mb-5">
    <span class="italic"
      >XXX (Performance benchmarks coming soon—though expect <code>jax-js</code>
      to be faster than TensorFlow.js due to JIT.)</span
    >
  </p>

  <p class="mb-5">
    Running GPU code should be as easy as making a website. With <code
      >jax-js</code
    >, you can bring numerical computing to your users in the browser, with zero
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
