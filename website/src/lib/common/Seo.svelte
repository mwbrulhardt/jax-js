<script lang="ts">
  const {
    title = "jax-js – open-source ML framework",
    description = "Machine learning library and compiler for the web, written in pure JavaScript. Run high-performance WebAssembly and WebGPU kernels.",
  }: { title?: string; description?: string } = $props();
</script>

<svelte:head>
  <meta name="description" content={description} />

  <meta property="og:type" content="website" />
  <meta property="og:site_name" content="jax-js" />
  <meta property="og:title" content={title} />
  <meta property="og:description" content={description} />
  <!-- Needs to be an absolute URL, so this is hardcoded. -->
  <meta
    property="og:image"
    content="https://jax-js.com/images/social-image.png"
  />
</svelte:head>
