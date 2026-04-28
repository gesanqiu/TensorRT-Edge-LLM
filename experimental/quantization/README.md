# Standalone Quantization for TensorRT Edge-LLM

Decoupled quantization pipeline that produces **unified HuggingFace
checkpoints** (safetensors + config) consumable by `llm_loader`.
Runs in a clean venv — no `tensorrt_edgellm` dependency.
See `requirements.txt` for the full dependency list.

## Design

### Problem

The existing quantization code lives inside `tensorrt_edgellm.quantization`
and imports model-specific loaders, custom decoder layers, ONNX plugins, and
TensorRT native ops.  This couples quantization (a GPU-only, PyTorch-only
step) with the export pipeline, making the venv heavy and brittle.

### Solution

```
┌──────────────────────────────────────────────┐
│           experimental/quantization          │
│                                              │
│  quantization_configs.py  Recipe presets     │
│  quantize.py              Load → quant → save│
│  models/                                     │
│    eagle3_draft.py        Eagle3 draft model │
│  cli.py                   CLI entry point    │
└──────────────────┬───────────────────────────┘
                   │ unified checkpoint
                   ▼
┌──────────────────────────────────────────────┐
│          experimental/llm_loader             │
│                                              │
│  Load checkpoint → build ONNX models        │
└──────────────────────────────────────────────┘
```

**Key decisions:**

1. **Model loading** uses only `AutoModelForCausalLM` /
   `AutoModelForImageTextToText`.  Model-specific workarounds
   (Phi-4MM LoRA merge, NemotronH Mamba stub, GPTQ gate fixes) are
   not yet implemented.

2. **`models/` subfolder** holds standalone model implementations for
   architectures not available in HuggingFace `transformers`.  For
   example, EAGLE3 (speculative-decoding draft model) is reimplemented
   using standalone PyTorch building blocks (`RMSNorm`, `SwiGLUMLP`,
   `RotaryEmbedding`) with no dependency on `transformers` model
   classes.  Only the calibration `forward` is implemented; the full
   inference forward with KV-cache and GatherND belongs in the export
   layer.

3. **Calibration** uses text data only (cnn_dailymail or a local
   dataset).  Audio/visual calibration for Omni models is not yet
   implemented.

4. **Output** is always a unified HuggingFace checkpoint produced by
   `modelopt.torch.export.export_hf_checkpoint`.  This is the format
   that `llm_loader` reads.

### Supported quantization methods

| Backbone        | LM-head         | KV-cache |
|-----------------|-----------------|----------|
| `fp8`           | `fp8`           | `fp8`    |
| `int4_awq`      | `int4_awq`      | —        |
| `nvfp4`         | `nvfp4`         | —        |
| `mxfp8`         | `mxfp8`         | —        |
| `int8_sq`       | —               | —        |

Any backbone method can be combined with any lm_head method.  Examples:

- `nvfp4` backbone + `nvfp4` lm_head → full NVFP4
- `nvfp4` backbone + `fp8` lm_head → mixed precision
- `nvfp4` backbone + `int4_awq` lm_head → INT4 lm_head
- `fp8` backbone + `fp8` lm_head + `fp8` kv_cache → full FP8

## Usage

### LLM quantization

```bash
python -m experimental.quantization.cli llm \
    --model_dir /path/to/Qwen3.5-0.8B \
    --output_dir /path/to/output \
    --quantization nvfp4 \
    --lm_head_quantization nvfp4
```

### Eagle3 draft quantization

```bash
python -m experimental.quantization.cli draft \
    --base_model_dir /path/to/base_model \
    --draft_model_dir /path/to/eagle3_draft \
    --output_dir /path/to/output \
    --quantization fp8
```

### Common options

| Flag | Default | Description |
|------|---------|-------------|
| `--quantization` | None | Backbone method |
| `--lm_head_quantization` | None | LM-head method |
| `--kv_cache_quantization` | None | KV-cache method (fp8) |
| `--dtype` | fp16 | Loading dtype |
| `--device` | cuda | CUDA device |
| `--dataset` | cnn_dailymail | Calibration dataset |
| `--num_samples` | 512 | Number of calibration samples |

### Python API

```python
from experimental.quantization import quantize_and_export

quantize_and_export(
    model_dir="/path/to/model",
    output_dir="/path/to/output",
    quantization="nvfp4",
    lm_head_quantization="nvfp4",
)
```

## End-to-end workflow

This package covers **step 1** (quantization) of the TensorRT Edge-LLM
export pipeline.  The full workflow is:

```bash
# 1. Quantize (this package — runs on any CUDA host)
python -m experimental.quantization.cli llm \
    --model_dir /data/Qwen3.5-0.8B \
    --output_dir /tmp/qwen35-nvfp4 \
    --quantization nvfp4 --lm_head_quantization nvfp4

# 2. Export to ONNX (llm_loader — runs on any CUDA host)
python -m llm_loader.export_all_cli \
    /tmp/qwen35-nvfp4 /tmp/qwen35-onnx
```

Steps 3 (TRT engine build) and 4 (inference) run on the **edge device**
using the C++ runtime.  See the
[Quick Start Guide](https://nvidia.github.io/TensorRT-Edge-LLM/latest/user_guide/getting_started/quick-start-guide.html)
for engine build and inference instructions.

