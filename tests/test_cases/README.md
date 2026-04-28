## Test Cases

Small JSON request sets for `llm_inference` smoke and runtime-sanity runs.

### LLM

- `llm_basic.json`
  Basic text-only smoke coverage.
  Recommended engine: vanilla or EAGLE LLM, `maxBatchSize >= 1`, `maxInputLen >= 1024`, `maxKVCacheCapacity >= 4096`.
- `llm_lora.json`
  Text-only LoRA coverage.
  Recommended engine: LLM engine built with the matching LoRA support, `maxBatchSize >= 1`, `maxInputLen >= 1024`.
- `llm_runtime_sanity_check.json`
  Text-runtime sanity coverage for chat templating, system-prompt KV cache, `disable_spec_decode`, and same-process reuse.
  Recommended engine: vanilla or EAGLE LLM, `maxBatchSize >= 1`, `maxInputLen >= 1024`, `maxKVCacheCapacity >= 4096`.

### VLM

- `vlm_basic.json`
  Basic multimodal smoke coverage.
  Recommended engine: VLM LLM + visual engines, `maxBatchSize >= 1`, `maxInputLen >= 2048`, visual token capacity sized for the sample images.
- `vlm_lora.json`
  Multimodal LoRA coverage.
  Recommended engine: VLM LLM + visual engines built with the matching LoRA support, `maxBatchSize >= 1`.
- `vlm_runtime_sanity_check.json`
  VLM runtime sanity coverage for text-only-on-VLM, single-image, multi-image, system-prompt KV cache, `disable_spec_decode`, and request-mode switching.
  Recommended engine: VLM vanilla or EAGLE, `maxBatchSize >= 1`, `maxInputLen >= 8192`, `maxKVCacheCapacity >= 8192`, visual token capacity sized for all referenced images.
  **Requires a visual engine built with a larger image-token shape range than the defaults** — the fixture contains multi-image requests whose combined ViT `cuSeqlens` exceeds the default `--maxImageTokens`. Rebuild the visual engine with a larger `--maxImageTokens` / `--maxImageTokensPerImage` before running this fixture.

### Notes

- These are runtime sanity tests, not semantic accuracy benchmarks.
- The sanity suites check request handling, state isolation, and output-shape sanity.
- The current sanity files use `batch_size: 1`.
