# Quantization

Use `experimental.quantization` when you start from an FP16 checkpoint and need to create a unified quantized checkpoint for `llm_loader`.

The quantization CLI writes a unified HuggingFace-style checkpoint directory that `llm_loader` can export directly.

Skip this step when you already have a supported pre-quantized HuggingFace checkpoint.

## Setup

```bash
export PYTHONPATH=/path/to/TensorRT-Edge-LLM:/path/to/TensorRT-Edge-LLM/experimental:$PYTHONPATH
python -m experimental.quantization.cli --help
```

## Quantize An LLM

```bash
python -m experimental.quantization.cli llm \
  --model_dir /path/to/Qwen3.5-0.8B \
  --output_dir /tmp/qwen35_nvfp4 \
  --quantization nvfp4 \
  --lm_head_quantization nvfp4
```

## Enable FP8 KV Cache

```bash
python -m experimental.quantization.cli llm \
  --model_dir /path/to/Qwen3-8B \
  --output_dir /tmp/qwen3_nvfp4_fp8kv \
  --quantization nvfp4 \
  --kv_cache_quantization fp8
```

When this checkpoint is exported with `llm_loader`, FP8 KV cache is detected from the checkpoint metadata automatically.

## Quantize An EAGLE3 Draft

```bash
python -m experimental.quantization.cli draft \
  --base_model_dir /path/to/base_model \
  --draft_model_dir /path/to/eagle3_draft \
  --output_dir /tmp/eagle3_draft_fp8 \
  --quantization fp8
```

## Export The Quantized Checkpoint

```bash
python -m llm_loader.export_all_cli \
  /tmp/qwen35_nvfp4 \
  /tmp/qwen35_onnx
```

To also store the runtime embedding table in FP8:

```bash
python -m llm_loader.export_all_cli \
  /tmp/qwen35_nvfp4 \
  /tmp/qwen35_onnx \
  --fp8-embedding
```

Build engines and run inference with the normal C++ tools. See [Quick Start Guide](../getting_started/quick-start-guide.md).

## Supported Methods

| Component | Methods |
|---|---|
| Backbone | `fp8`, `int4_awq`, `nvfp4`, `mxfp8`, `int8_sq` |
| LM head | `fp8`, `int4_awq`, `nvfp4`, `mxfp8` |
| KV cache | `fp8` |

## Notes

- The package writes unified checkpoints only. It does not export ONNX or build TensorRT engines.
- Audio and visual calibration are not implemented.
- GPTQ checkpoints are loaded as pre-quantized checkpoints; this package does not create GPTQ models.
