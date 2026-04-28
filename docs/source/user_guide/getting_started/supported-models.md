# Supported Models

> **Code Location:** `experimental/llm_loader/` (recommended export), `experimental/quantization/` (checkpoint quantization), `experimental/server/` (Python API/server), `tensorrt_edgellm/` (legacy export), `cpp/` (runtime)
>
> **Pre-Quantized Checkpoints:** When a supported pre-quantized checkpoint is available, the [checkpoint-based loader](../../developer_guide/software-design/llm-loader.md) can export it directly without a separate quantization step.

## Support Policy

TensorRT Edge-LLM supports the checkpoint IDs listed below. Dense LLM families include official dense checkpoints below 30B parameters. Larger dense checkpoints and non-dense variants require case-by-case validation. MoE, multimodal, audio, TTS, omni, and EAGLE support is limited to the listed rows.

The model coverage list is not comprehensive, and not every listed checkpoint has been fully verified on every supported platform and precision. If a listed model does not export, build, or run correctly, please report an issue with the checkpoint ID, precision, platform, and command line used.

The model class names were checked against the installed `transformers==5.3.0` package and the upstream [Transformers model source tree](https://github.com/huggingface/transformers/tree/main/src/transformers/models). Checkpoint IDs are linked to their Hugging Face pages and grouped into original checkpoints and quantized checkpoints.

## Precision Notes

- Dense precision set: FP16/BF16 checkpoints, ModelOpt FP8/MXFP8/FP4/NVFP4/INT4 AWQ/INT8 SmoothQuant checkpoints, and INT4 GPTQ checkpoints. INT8 GPTQ is not supported.
- For FP16/BF16 source checkpoints, use the [Quantization](../features/quantization.md) script to create a unified quantized checkpoint for `llm_loader`, then export the generated checkpoint.
- FP8 KV cache is detected automatically from checkpoint metadata by `llm_loader`.
- `llm_loader` exports visual encoders in FP16. FP8 visual encoder export is available through the legacy `tensorrt_edgellm` visual quantization/export tools.
- MXFP8 and FP4/NVFP4 require Blackwell-class hardware for runtime execution.

## Support Matrix

| Category | Model series | Transformers class / checkpoint architecture | `llm_loader` handling | Checkpoint type | Checkpoint ID | Supported precisions |
|----------|--------------|----------------------------------------------|-----------------------|-----------------|---------------|----------------------|
| Dense LLM | Llama 3.x Instruct / selected Llama-derived Instruct | [`LlamaForCausalLM`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py) | `llama` -> default `CausalLM` | Original | [meta-llama/Meta-Llama-3-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct) | Dense precision set |
|  |  |  |  | Original | [meta-llama/Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) |  |
|  |  |  |  | Original | [meta-llama/Llama-3.2-1B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct) |  |
|  |  |  |  | Original | [meta-llama/Llama-3.2-3B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct) |  |
|  |  |  |  | Quantized | [nvidia/Llama-3.1-8B-Instruct-FP8](https://huggingface.co/nvidia/Llama-3.1-8B-Instruct-FP8) |  |
|  |  |  |  | Quantized | [nvidia/Llama-3.1-8B-Instruct-NVFP4](https://huggingface.co/nvidia/Llama-3.1-8B-Instruct-NVFP4) |  |
| Dense LLM | Qwen2/Qwen2.5 dense and Qwen-derived dense | [`Qwen2ForCausalLM`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen2/modeling_qwen2.py) | `qwen2` -> default `CausalLM` | Original | [Qwen/Qwen2-0.5B](https://huggingface.co/Qwen/Qwen2-0.5B) | Dense precision set |
|  |  |  |  | Original | [Qwen/Qwen2-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2-0.5B-Instruct) |  |
|  |  |  |  | Original | [Qwen/Qwen2-1.5B](https://huggingface.co/Qwen/Qwen2-1.5B) |  |
|  |  |  |  | Original | [Qwen/Qwen2-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2-1.5B-Instruct) |  |
|  |  |  |  | Original | [Qwen/Qwen2-7B](https://huggingface.co/Qwen/Qwen2-7B) |  |
|  |  |  |  | Original | [Qwen/Qwen2-7B-Instruct](https://huggingface.co/Qwen/Qwen2-7B-Instruct) |  |
|  |  |  |  | Original | [Qwen/Qwen2-Math-1.5B](https://huggingface.co/Qwen/Qwen2-Math-1.5B) |  |
|  |  |  |  | Original | [Qwen/Qwen2-Math-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2-Math-1.5B-Instruct) |  |
|  |  |  |  | Original | [Qwen/Qwen2-Math-7B](https://huggingface.co/Qwen/Qwen2-Math-7B) |  |
|  |  |  |  | Original | [Qwen/Qwen2-Math-7B-Instruct](https://huggingface.co/Qwen/Qwen2-Math-7B-Instruct) |  |
|  |  |  |  | Original | [Qwen/Qwen2.5-0.5B](https://huggingface.co/Qwen/Qwen2.5-0.5B) |  |
|  |  |  |  | Original | [Qwen/Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct) |  |
|  |  |  |  | Original | [Qwen/Qwen2.5-1.5B](https://huggingface.co/Qwen/Qwen2.5-1.5B) |  |
|  |  |  |  | Original | [Qwen/Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct) |  |
|  |  |  |  | Original | [Qwen/Qwen2.5-3B](https://huggingface.co/Qwen/Qwen2.5-3B) |  |
|  |  |  |  | Original | [Qwen/Qwen2.5-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct) |  |
|  |  |  |  | Original | [Qwen/Qwen2.5-7B](https://huggingface.co/Qwen/Qwen2.5-7B) |  |
|  |  |  |  | Original | [Qwen/Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct) |  |
|  |  |  |  | Original | [Qwen/Qwen2.5-14B](https://huggingface.co/Qwen/Qwen2.5-14B) |  |
|  |  |  |  | Original | [Qwen/Qwen2.5-14B-Instruct](https://huggingface.co/Qwen/Qwen2.5-14B-Instruct) |  |
|  |  |  |  | Original | [Qwen/Qwen2.5-Coder-0.5B](https://huggingface.co/Qwen/Qwen2.5-Coder-0.5B) |  |
|  |  |  |  | Original | [Qwen/Qwen2.5-Coder-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-Coder-0.5B-Instruct) |  |
|  |  |  |  | Original | [Qwen/Qwen2.5-Coder-1.5B](https://huggingface.co/Qwen/Qwen2.5-Coder-1.5B) |  |
|  |  |  |  | Original | [Qwen/Qwen2.5-Coder-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-Coder-1.5B-Instruct) |  |
|  |  |  |  | Original | [Qwen/Qwen2.5-Coder-3B](https://huggingface.co/Qwen/Qwen2.5-Coder-3B) |  |
|  |  |  |  | Original | [Qwen/Qwen2.5-Coder-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-Coder-3B-Instruct) |  |
|  |  |  |  | Original | [Qwen/Qwen2.5-Coder-7B](https://huggingface.co/Qwen/Qwen2.5-Coder-7B) |  |
|  |  |  |  | Original | [Qwen/Qwen2.5-Coder-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct) |  |
|  |  |  |  | Original | [Qwen/Qwen2.5-Coder-14B](https://huggingface.co/Qwen/Qwen2.5-Coder-14B) |  |
|  |  |  |  | Original | [Qwen/Qwen2.5-Coder-14B-Instruct](https://huggingface.co/Qwen/Qwen2.5-Coder-14B-Instruct) |  |
|  |  |  |  | Original | [Qwen/Qwen2.5-Math-1.5B](https://huggingface.co/Qwen/Qwen2.5-Math-1.5B) |  |
|  |  |  |  | Original | [Qwen/Qwen2.5-Math-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-Math-1.5B-Instruct) |  |
|  |  |  |  | Original | [Qwen/Qwen2.5-Math-7B](https://huggingface.co/Qwen/Qwen2.5-Math-7B) |  |
|  |  |  |  | Original | [Qwen/Qwen2.5-Math-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-Math-7B-Instruct) |  |
|  |  |  |  | Original | [deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B) |  |
|  |  |  |  | Original | [deepseek-ai/DeepSeek-R1-Distill-Qwen-7B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B) |  |
|  |  |  |  | Original | [deepseek-ai/DeepSeek-R1-Distill-Qwen-14B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-14B) |  |
|  |  |  |  | Quantized | [Qwen/Qwen2-0.5B-Instruct-AWQ](https://huggingface.co/Qwen/Qwen2-0.5B-Instruct-AWQ) |  |
|  |  |  |  | Quantized | [Qwen/Qwen2-0.5B-Instruct-GPTQ-Int4](https://huggingface.co/Qwen/Qwen2-0.5B-Instruct-GPTQ-Int4) |  |
|  |  |  |  | Quantized | [Qwen/Qwen2-1.5B-Instruct-AWQ](https://huggingface.co/Qwen/Qwen2-1.5B-Instruct-AWQ) |  |
|  |  |  |  | Quantized | [Qwen/Qwen2-1.5B-Instruct-GPTQ-Int4](https://huggingface.co/Qwen/Qwen2-1.5B-Instruct-GPTQ-Int4) |  |
|  |  |  |  | Quantized | [Qwen/Qwen2-7B-Instruct-AWQ](https://huggingface.co/Qwen/Qwen2-7B-Instruct-AWQ) |  |
|  |  |  |  | Quantized | [Qwen/Qwen2-7B-Instruct-GPTQ-Int4](https://huggingface.co/Qwen/Qwen2-7B-Instruct-GPTQ-Int4) |  |
|  |  |  |  | Quantized | [Qwen/Qwen2.5-0.5B-Instruct-AWQ](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-AWQ) |  |
|  |  |  |  | Quantized | [Qwen/Qwen2.5-0.5B-Instruct-GPTQ-Int4](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GPTQ-Int4) |  |
|  |  |  |  | Quantized | [Qwen/Qwen2.5-1.5B-Instruct-AWQ](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-AWQ) |  |
|  |  |  |  | Quantized | [Qwen/Qwen2.5-1.5B-Instruct-GPTQ-Int4](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GPTQ-Int4) |  |
|  |  |  |  | Quantized | [Qwen/Qwen2.5-3B-Instruct-AWQ](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-AWQ) |  |
|  |  |  |  | Quantized | [Qwen/Qwen2.5-3B-Instruct-GPTQ-Int4](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GPTQ-Int4) |  |
|  |  |  |  | Quantized | [Qwen/Qwen2.5-7B-Instruct-AWQ](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct-AWQ) |  |
|  |  |  |  | Quantized | [Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4) |  |
|  |  |  |  | Quantized | [Qwen/Qwen2.5-14B-Instruct-AWQ](https://huggingface.co/Qwen/Qwen2.5-14B-Instruct-AWQ) |  |
|  |  |  |  | Quantized | [Qwen/Qwen2.5-14B-Instruct-GPTQ-Int4](https://huggingface.co/Qwen/Qwen2.5-14B-Instruct-GPTQ-Int4) |  |
|  |  |  |  | Quantized | [Qwen/Qwen2.5-Coder-0.5B-Instruct-AWQ](https://huggingface.co/Qwen/Qwen2.5-Coder-0.5B-Instruct-AWQ) |  |
|  |  |  |  | Quantized | [Qwen/Qwen2.5-Coder-0.5B-Instruct-GPTQ-Int4](https://huggingface.co/Qwen/Qwen2.5-Coder-0.5B-Instruct-GPTQ-Int4) |  |
|  |  |  |  | Quantized | [Qwen/Qwen2.5-Coder-1.5B-Instruct-AWQ](https://huggingface.co/Qwen/Qwen2.5-Coder-1.5B-Instruct-AWQ) |  |
|  |  |  |  | Quantized | [Qwen/Qwen2.5-Coder-1.5B-Instruct-GPTQ-Int4](https://huggingface.co/Qwen/Qwen2.5-Coder-1.5B-Instruct-GPTQ-Int4) |  |
|  |  |  |  | Quantized | [Qwen/Qwen2.5-Coder-3B-Instruct-AWQ](https://huggingface.co/Qwen/Qwen2.5-Coder-3B-Instruct-AWQ) |  |
|  |  |  |  | Quantized | [Qwen/Qwen2.5-Coder-3B-Instruct-GPTQ-Int4](https://huggingface.co/Qwen/Qwen2.5-Coder-3B-Instruct-GPTQ-Int4) |  |
|  |  |  |  | Quantized | [Qwen/Qwen2.5-Coder-7B-Instruct-AWQ](https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct-AWQ) |  |
|  |  |  |  | Quantized | [Qwen/Qwen2.5-Coder-7B-Instruct-GPTQ-Int4](https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct-GPTQ-Int4) |  |
|  |  |  |  | Quantized | [Qwen/Qwen2.5-Coder-14B-Instruct-AWQ](https://huggingface.co/Qwen/Qwen2.5-Coder-14B-Instruct-AWQ) |  |
|  |  |  |  | Quantized | [Qwen/Qwen2.5-Coder-14B-Instruct-GPTQ-Int4](https://huggingface.co/Qwen/Qwen2.5-Coder-14B-Instruct-GPTQ-Int4) |  |
| Dense LLM | Qwen3 dense | [`Qwen3ForCausalLM`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3/modeling_qwen3.py) | `qwen3` -> default `CausalLM` | Original | [Qwen/Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B) | Dense precision set |
|  |  |  |  | Original | [Qwen/Qwen3-0.6B-Base](https://huggingface.co/Qwen/Qwen3-0.6B-Base) |  |
|  |  |  |  | Original | [Qwen/Qwen3-1.7B](https://huggingface.co/Qwen/Qwen3-1.7B) |  |
|  |  |  |  | Original | [Qwen/Qwen3-1.7B-Base](https://huggingface.co/Qwen/Qwen3-1.7B-Base) |  |
|  |  |  |  | Original | [Qwen/Qwen3-4B](https://huggingface.co/Qwen/Qwen3-4B) |  |
|  |  |  |  | Original | [Qwen/Qwen3-4B-Base](https://huggingface.co/Qwen/Qwen3-4B-Base) |  |
|  |  |  |  | Original | [Qwen/Qwen3-4B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507) |  |
|  |  |  |  | Original | [Qwen/Qwen3-4B-Thinking-2507](https://huggingface.co/Qwen/Qwen3-4B-Thinking-2507) |  |
|  |  |  |  | Original | [Qwen/Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B) |  |
|  |  |  |  | Original | [Qwen/Qwen3-8B-Base](https://huggingface.co/Qwen/Qwen3-8B-Base) |  |
|  |  |  |  | Original | [Qwen/Qwen3-14B](https://huggingface.co/Qwen/Qwen3-14B) |  |
|  |  |  |  | Original | [Qwen/Qwen3-14B-Base](https://huggingface.co/Qwen/Qwen3-14B-Base) |  |
|  |  |  |  | Quantized | [Qwen/Qwen3-4B-AWQ](https://huggingface.co/Qwen/Qwen3-4B-AWQ) |  |
|  |  |  |  | Quantized | [Qwen/Qwen3-8B-AWQ](https://huggingface.co/Qwen/Qwen3-8B-AWQ) |  |
|  |  |  |  | Quantized | [Qwen/Qwen3-14B-AWQ](https://huggingface.co/Qwen/Qwen3-14B-AWQ) |  |
|  |  |  |  | Quantized | [nvidia/Qwen3-8B-FP8](https://huggingface.co/nvidia/Qwen3-8B-FP8) |  |
|  |  |  |  | Quantized | [nvidia/Qwen3-8B-NVFP4](https://huggingface.co/nvidia/Qwen3-8B-NVFP4) |  |
|  |  |  |  | Quantized | [nvidia/Qwen3-14B-FP8](https://huggingface.co/nvidia/Qwen3-14B-FP8) |  |
|  |  |  |  | Quantized | [nvidia/Qwen3-14B-NVFP4](https://huggingface.co/nvidia/Qwen3-14B-NVFP4) |  |
| Dense LLM / VLM | Qwen3.5 text and VLM | [Qwen3_5ForCausalLM](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_5/modeling_qwen3_5.py); [Qwen3_5ForConditionalGeneration](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_5/modeling_qwen3_5.py) | Text: `qwen3_5_text` -> `Qwen3_5CausalLM`; VLM: `qwen3_5` -> `Qwen3_5CausalLM` + `Qwen3_5VLVisualModel` | Original | [Qwen/Qwen3.5-0.8B](https://huggingface.co/Qwen/Qwen3.5-0.8B) | Dense precision set for text; VLM original checkpoints only |
|  |  |  |  | Original | [Qwen/Qwen3.5-0.8B-Base](https://huggingface.co/Qwen/Qwen3.5-0.8B-Base) |  |
|  |  |  |  | Original | [Qwen/Qwen3.5-2B](https://huggingface.co/Qwen/Qwen3.5-2B) |  |
|  |  |  |  | Original | [Qwen/Qwen3.5-2B-Base](https://huggingface.co/Qwen/Qwen3.5-2B-Base) |  |
|  |  |  |  | Original | [Qwen/Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B) |  |
|  |  |  |  | Original | [Qwen/Qwen3.5-4B-Base](https://huggingface.co/Qwen/Qwen3.5-4B-Base) |  |
|  |  |  |  | Original | [Qwen/Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B) |  |
|  |  |  |  | Original | [Qwen/Qwen3.5-9B-Base](https://huggingface.co/Qwen/Qwen3.5-9B-Base) |  |
|  |  |  |  | Original | [Qwen/Qwen3.5-27B](https://huggingface.co/Qwen/Qwen3.5-27B) |  |
| Dense LLM | Nemotron Nano dense | [`NemotronHForCausalLM`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/nemotron_h/modeling_nemotron_h.py) | `nemotron_h` -> `NemotronHCausalLM` | Original | [nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16) | BF16, FP8, NVFP4 |
|  |  |  |  | Original | [nvidia/NVIDIA-Nemotron-Nano-9B-v2](https://huggingface.co/nvidia/NVIDIA-Nemotron-Nano-9B-v2) |  |
|  |  |  |  | Quantized | [nvidia/NVIDIA-Nemotron-3-Nano-4B-FP8](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-4B-FP8) |  |
|  |  |  |  | Quantized | [nvidia/NVIDIA-Nemotron-3-Nano-4B-NVFP4](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-4B-NVFP4) |  |
|  |  |  |  | Quantized | [nvidia/NVIDIA-Nemotron-Nano-9B-v2-FP8](https://huggingface.co/nvidia/NVIDIA-Nemotron-Nano-9B-v2-FP8) |  |
|  |  |  |  | Quantized | [nvidia/NVIDIA-Nemotron-Nano-9B-v2-NVFP4](https://huggingface.co/nvidia/NVIDIA-Nemotron-Nano-9B-v2-NVFP4) |  |
| MoE | Qwen3-MoE | [`Qwen3MoeForCausalLM`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_moe/modeling_qwen3_moe.py) | `qwen3_moe` -> `Qwen3MoeCausalLM` | Quantized | [Qwen/Qwen3-30B-A3B-GPTQ-Int4](https://huggingface.co/Qwen/Qwen3-30B-A3B-GPTQ-Int4) | INT4 only |
| MoE | Nemotron3-MoE | [`NemotronHForCausalLM`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/nemotron_h/modeling_nemotron_h.py) | `nemotron_h` -> `NemotronHCausalLM` | Quantized | [nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4) | NVFP4 only |
| VLM | Qwen2.5-VL | [`Qwen2_5_VLForConditionalGeneration`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py) | `qwen2_5_vl` + `Qwen2_5VLVisualModel` | Original | [Qwen/Qwen2.5-VL-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct) | Dense precision set for LLM backbone |
|  |  |  |  | Original | [Qwen/Qwen2.5-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct) |  |
|  |  |  |  | Quantized | [Qwen/Qwen2.5-VL-3B-Instruct-AWQ](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct-AWQ) |  |
|  |  |  |  | Quantized | [Qwen/Qwen2.5-VL-7B-Instruct-AWQ](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct-AWQ) |  |
|  |  |  |  | Quantized | [nvidia/Qwen2.5-VL-7B-Instruct-FP8](https://huggingface.co/nvidia/Qwen2.5-VL-7B-Instruct-FP8) |  |
|  |  |  |  | Quantized | [nvidia/Qwen2.5-VL-7B-Instruct-NVFP4](https://huggingface.co/nvidia/Qwen2.5-VL-7B-Instruct-NVFP4) |  |
| VLM | Qwen3-VL / compatible | [`Qwen3VLForConditionalGeneration`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_vl/modeling_qwen3_vl.py) | `qwen3_vl` + `Qwen3VLVisualModel` | Original | [Qwen/Qwen3-VL-2B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct) | Dense precision set for LLM backbone |
|  |  |  |  | Original | [Qwen/Qwen3-VL-2B-Thinking](https://huggingface.co/Qwen/Qwen3-VL-2B-Thinking) |  |
|  |  |  |  | Original | [Qwen/Qwen3-VL-4B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct) |  |
|  |  |  |  | Original | [Qwen/Qwen3-VL-4B-Thinking](https://huggingface.co/Qwen/Qwen3-VL-4B-Thinking) |  |
|  |  |  |  | Original | [Qwen/Qwen3-VL-8B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct) |  |
|  |  |  |  | Original | [Qwen/Qwen3-VL-8B-Thinking](https://huggingface.co/Qwen/Qwen3-VL-8B-Thinking) |  |
|  |  |  |  | Original | [nvidia/Cosmos-Reason2-2B](https://huggingface.co/nvidia/Cosmos-Reason2-2B) |  |
|  |  |  |  | Original | [nvidia/Cosmos-Reason2-8B](https://huggingface.co/nvidia/Cosmos-Reason2-8B) |  |
|  |  |  |  | Quantized | [nvidia/Cosmos-Reason2-2B-FP8](https://huggingface.co/nvidia/Cosmos-Reason2-2B-FP8) |  |
|  |  |  |  | Quantized | [nvidia/Cosmos-Reason2-2B-NVFP4](https://huggingface.co/nvidia/Cosmos-Reason2-2B-NVFP4) |  |
|  |  |  |  | Quantized | [nvidia/Cosmos-Reason2-8B-FP8](https://huggingface.co/nvidia/Cosmos-Reason2-8B-FP8) |  |
|  |  |  |  | Quantized | [nvidia/Cosmos-Reason2-8B-NVFP4](https://huggingface.co/nvidia/Cosmos-Reason2-8B-NVFP4) |  |
| VLM | InternVL3 / InternVL3.5 HF format | [`InternVLForConditionalGeneration`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/internvl/modeling_internvl.py) | `internvl_chat` / `internvl` + InternVL visual models | Original | [OpenGVLab/InternVL3-1B-hf](https://huggingface.co/OpenGVLab/InternVL3-1B-hf) | Dense precision set for LLM backbone |
|  |  |  |  | Original | [OpenGVLab/InternVL3-2B-hf](https://huggingface.co/OpenGVLab/InternVL3-2B-hf) |  |
|  |  |  |  | Original | [OpenGVLab/InternVL3-8B-hf](https://huggingface.co/OpenGVLab/InternVL3-8B-hf) |  |
|  |  |  |  | Original | [OpenGVLab/InternVL3-9B](https://huggingface.co/OpenGVLab/InternVL3-9B) |  |
|  |  |  |  | Original | [OpenGVLab/InternVL3-9B-Instruct](https://huggingface.co/OpenGVLab/InternVL3-9B-Instruct) |  |
|  |  |  |  | Original | [OpenGVLab/InternVL3-14B-hf](https://huggingface.co/OpenGVLab/InternVL3-14B-hf) |  |
|  |  |  |  | Original | [OpenGVLab/InternVL3_5-1B-HF](https://huggingface.co/OpenGVLab/InternVL3_5-1B-HF) |  |
|  |  |  |  | Original | [OpenGVLab/InternVL3_5-2B-HF](https://huggingface.co/OpenGVLab/InternVL3_5-2B-HF) |  |
|  |  |  |  | Original | [OpenGVLab/InternVL3_5-4B-HF](https://huggingface.co/OpenGVLab/InternVL3_5-4B-HF) |  |
|  |  |  |  | Original | [OpenGVLab/InternVL3_5-8B-HF](https://huggingface.co/OpenGVLab/InternVL3_5-8B-HF) |  |
|  |  |  |  | Original | [OpenGVLab/InternVL3_5-14B-HF](https://huggingface.co/OpenGVLab/InternVL3_5-14B-HF) |  |
|  |  |  |  | Quantized | [OpenGVLab/InternVL3-1B-AWQ](https://huggingface.co/OpenGVLab/InternVL3-1B-AWQ) |  |
|  |  |  |  | Quantized | [OpenGVLab/InternVL3-2B-AWQ](https://huggingface.co/OpenGVLab/InternVL3-2B-AWQ) |  |
|  |  |  |  | Quantized | [OpenGVLab/InternVL3-8B-AWQ](https://huggingface.co/OpenGVLab/InternVL3-8B-AWQ) |  |
|  |  |  |  | Quantized | [OpenGVLab/InternVL3-14B-AWQ](https://huggingface.co/OpenGVLab/InternVL3-14B-AWQ) |  |
| VLM | Phi-4-Multimodal | [`Phi4MultimodalForCausalLM`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/phi4_multimodal/modeling_phi4_multimodal.py) | `phi4mm` / `phi4_multimodal` + `Phi4MMVisualModel` | Original | [microsoft/Phi-4-multimodal-instruct](https://huggingface.co/microsoft/Phi-4-multimodal-instruct) | Merge vision LoRA, then dense precision set for the LLM backbone |
| Audio / Speech | Qwen3-ASR | Checkpoint architecture `Qwen3ASRForConditionalGeneration`; text backbone compatible with [`Qwen3ForCausalLM`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3/modeling_qwen3.py) | `Qwen3ASRLanguageModel` + `QwenAudioEncoder` | Original | [Qwen/Qwen3-ASR-0.6B](https://huggingface.co/Qwen/Qwen3-ASR-0.6B) | FP16 |
|  |  |  |  | Original | [Qwen/Qwen3-ASR-1.7B](https://huggingface.co/Qwen/Qwen3-ASR-1.7B) |  |
| TTS | Qwen3-TTS | Checkpoint architecture `Qwen3TTSForConditionalGeneration`; talker/code-predictor decoders compatible with [`Qwen3ForCausalLM`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3/modeling_qwen3.py) | `TalkerCausalLM` + `CodePredictorCausalLM` | Original | [Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice) | FP16 |
| Omni | Nemotron-Omni | Checkpoint architecture `NemotronH_Nano_Omni_Reasoning_V3`; LLM is Nemotron-H compatible with [`NemotronHForCausalLM`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/nemotron_h/modeling_nemotron_h.py) | `NemotronHCausalLM` + `NemotronOmniVisualModel` + `NemotronOmniAudioModel` | Quantized | [nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-NVFP4](https://huggingface.co/nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-NVFP4) | NVFP4 only |

Qwen3-ASR and Qwen3-TTS use checkpoint architecture names that are not present in the installed `transformers==5.3.0` package, so TensorRT Edge-LLM handles their speech/audio/talker components with local model implementations.

## EAGLE3 Draft Models

EAGLE3 draft checkpoints are detected by `draft_vocab_size` in `config.json` and exported with `Eagle3DraftModel`. Draft checkpoints can be quantized with `experimental.quantization` using the same ModelOpt methods exposed by the draft quantization CLI: `fp8`, `int4_awq`, `nvfp4`, `mxfp8`, and `int8_sq` for the backbone; `fp8`, `int4_awq`, `nvfp4`, and `mxfp8` for the LM head; and `fp8` for KV cache.

| Draft checkpoint | Base model | Draft config class |
|------------------|------------|--------------------|
| [yuhuili/EAGLE3-LLaMA3.1-Instruct-8B](https://huggingface.co/yuhuili/EAGLE3-LLaMA3.1-Instruct-8B) | [meta-llama/Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) | `LlamaForCausalLM`-style draft |
| [AngelSlim/Qwen3-1.7B_eagle3](https://huggingface.co/AngelSlim/Qwen3-1.7B_eagle3) | [Qwen/Qwen3-1.7B](https://huggingface.co/Qwen/Qwen3-1.7B) | `LlamaForCausalLMEagle3`-style draft |
| [AngelSlim/Qwen3-4B_eagle3](https://huggingface.co/AngelSlim/Qwen3-4B_eagle3) | [Qwen/Qwen3-4B](https://huggingface.co/Qwen/Qwen3-4B) | `Eagle3LlamaForCausalLM`-style draft |
| [Tengyunw/qwen3_8b_eagle3](https://huggingface.co/Tengyunw/qwen3_8b_eagle3) | [Qwen/Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B) | `LlamaForCausalLMEagle3`-style draft |
| [AngelSlim/Qwen3-8B_eagle3](https://huggingface.co/AngelSlim/Qwen3-8B_eagle3) | [Qwen/Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B) | `LlamaForCausalLMEagle3`-style draft |
| [Rayzl/qwen2.5-vl-7b-eagle3-sgl](https://huggingface.co/Rayzl/qwen2.5-vl-7b-eagle3-sgl) | [Qwen/Qwen2.5-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct) | `LlamaForCausalLMEagle3`-style draft |
