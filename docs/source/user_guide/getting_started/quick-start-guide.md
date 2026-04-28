# Quick Start Guide

> **Repository:** [github.com/NVIDIA/TensorRT-Edge-LLM](https://github.com/NVIDIA/TensorRT-Edge-LLM)

This quick start guide will get you up and running with TensorRT Edge-LLM in ~15 minutes.

**Prerequisites:** Complete the [Installation Guide](installation.md) on the machine where you will run TensorRT Edge-LLM. For the manual export and device-transfer path, set up both the x86 host and edge device.

---

## Recommended: High-Level API or Server

For Jetson Thor and x86 development, use the high-level Python API (HLAPI) or the OpenAI-compatible server. Build the project once with Python bindings enabled, then let the HLAPI export, build, load, and run the model from a HuggingFace checkpoint.

For x86 development:

```bash
cd ~/TensorRT-Edge-LLM

mkdir -p build
cd build
cmake .. \
  -DTRT_PACKAGE_DIR=$TRT_PACKAGE_DIR \
  -DCUDA_CTK_VERSION=<YOUR_CUDA_VERSION> \
  -DBUILD_PYTHON_BINDINGS=ON
make -j$(nproc)
cd ..
```

For Jetson Thor, follow the CMake pattern used by CI and enable CuTe DSL prebuilt kernels:

```bash
cd ~/TensorRT-Edge-LLM

mkdir -p build
cd build
cmake .. \
  -DTRT_PACKAGE_DIR=/usr \
  -DCUDA_CTK_VERSION=13.0 \
  -DCMAKE_TOOLCHAIN_FILE=cmake/aarch64_linux_toolchain.cmake \
  -DEMBEDDED_TARGET=jetson-thor \
  -DENABLE_CUTE_DSL=ALL \
  -DBUILD_PYTHON_BINDINGS=ON
make -j$(nproc)
cd ..
```

After either build:

```bash
export PYTHONPATH=$PWD:$PWD/experimental:$PYTHONPATH

pip install pybind11 fastapi uvicorn
```

Run a prompt with the HLAPI:

```bash
python - <<'PY'
from experimental.server import LLM, SamplingParams

llm = LLM(model="Qwen/Qwen3-0.6B")
outputs = llm.generate(
    ["What is the capital of the United States?"],
    SamplingParams(max_tokens=128),
)
print(outputs[0].text)
PY
```

Or launch an OpenAI-compatible server:

```bash
python -m experimental.server \
  --model Qwen/Qwen3-0.6B \
  --port 8000
```

Query the server from another terminal:

```bash
curl -sN http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "What is the capital of the United States?"}], "max_tokens": 128}'
```

For more options, including loading existing ONNX or engine directories, see [Experimental High-Level Python API and Server](../examples/experimental-server.md).

---

## Manual Export and C++ Runtime Path

Use this path when you need explicit control over ONNX export, engine build flags, file transfer, or the low-level C++ examples.

### Part 1: Export on x86 Host

This part runs on a standard x86 host with an NVIDIA GPU. **DriveOS users:** this process does not need to run in DriveOS Docker; use your regular x86 development machine.

#### Recommended: Checkpoint-Based Loader

The checkpoint-based loader (`llm_loader`) is the recommended export path. It reads pre-quantized HuggingFace checkpoints directly and exports to ONNX in a single command.

Let's use [Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B) as a lightweight example:

```bash
# Set up workspace directory
export WORKSPACE_DIR=$HOME/tensorrt-edgellm-workspace
export MODEL_NAME=Qwen3-0.6B
mkdir -p $WORKSPACE_DIR
cd $WORKSPACE_DIR

# Set PYTHONPATH to include the quantization package and loader
export PYTHONPATH=/path/to/TensorRT-Edge-LLM:/path/to/TensorRT-Edge-LLM/experimental:$PYTHONPATH

# Export FP16 model to ONNX (single command)
python -m llm_loader.export_all_cli \
    Qwen/Qwen3-0.6B \
    $MODEL_NAME/onnx

# Or export a pre-quantized NVFP4 checkpoint (no separate quantization step)
# See https://huggingface.co/nvidia/Qwen3-8B-NVFP4
python -m llm_loader.export_all_cli \
    /path/to/nvidia/Qwen3-8B-NVFP4 \
    Qwen3-8B-NVFP4/onnx

# Or export a pre-quantized INT4-AWQ checkpoint
# See https://huggingface.co/Qwen/Qwen3-4B-AWQ
python -m llm_loader.export_all_cli \
    /path/to/Qwen/Qwen3-4B-AWQ \
    Qwen3-4B-AWQ/onnx
```

For pre-quantized checkpoints (FP8, INT4 AWQ/GPTQ, NVFP4), simply point the loader at the quantized checkpoint directory — no separate quantization step is needed. See [Checkpoint-Based Model Loader](../../developer_guide/software-design/llm-loader.md) for full details.

To create a quantized checkpoint from an FP16 model, use [Quantization](../features/quantization.md), then export the generated checkpoint with `llm_loader`.

#### Alternative: Legacy Pipeline

> **Deprecated:** The legacy `tensorrt_edgellm` pipeline is kept for compatibility. New workflows should use `llm_loader`. The `tensorrt_edgellm/` folder will be removed in 0.8.0, with full feature parity provided by the `experimental/quantization` -> `experimental/llm_loader` workflow for all models and features. See the [migration guide](../../developer_guide/software-design/llm-loader.md#migrating-from-the-legacy-pipeline).

```bash
# Step 1: Quantize to FP8 (downloads model automatically)
tensorrt-edgellm-quantize-llm \
    --model_dir Qwen/Qwen3-0.6B \
    --output_dir $MODEL_NAME/quantized \
    --quantization fp8

# Step 2: Export to ONNX
tensorrt-edgellm-export-llm \
    --model_dir $MODEL_NAME/quantized \
    --output_dir $MODEL_NAME/onnx
```

> **Troubleshooting:** If you encounter issues during export, see the [Python Export Pipeline - Common Issues and Solutions](../../developer_guide/software-design/python-export-pipeline.md#common-issues-and-solutions).

#### Transfer to Device

Transfer the ONNX folder to your Thor device:

```bash
# From x86 host - transfer to device
scp -r $MODEL_NAME/onnx <device_user>@<device_ip>:~/tensorrt-edgellm-workspace/$MODEL_NAME/
```

> **Note:** Replace `<device_user>` and `<device_ip>` with your actual device credentials (e.g., `nvidia@192.168.1.100`). If the directory doesn't exist on the device, create it first: `ssh <device_user>@<device_ip> "mkdir -p ~/tensorrt-edgellm-workspace/$MODEL_NAME"`

---

### Part 2: Build and Run on Edge Device

#### Build TensorRT Engine

On your Thor device:

```bash
# Set up workspace directory
export WORKSPACE_DIR=$HOME/tensorrt-edgellm-workspace
export MODEL_NAME=Qwen3-0.6B
cd ~/TensorRT-Edge-LLM

# Build engine
./build/examples/llm/llm_build \
    --onnxDir $WORKSPACE_DIR/$MODEL_NAME/onnx \
    --engineDir $WORKSPACE_DIR/$MODEL_NAME/engines \
    --maxBatchSize 1 \
    --maxInputLen 1024 \
    --maxKVCacheCapacity 4096
```

Build time: ~2-5 minutes

#### Run Inference

Create an input file with a sample question:

```bash
cat > $WORKSPACE_DIR/input.json << 'EOF'
{
    "batch_size": 1,
    "temperature": 1.0,
    "top_p": 1.0,
    "top_k": 50,
    "max_generate_length": 128,
    "requests": [
        {
            "messages": [
                {
                    "role": "user",
                    "content": "What is the capital of United States?"
                }
            ]
        }
    ]
}
EOF
```

> **Tip:** You can also use example input files from `~/TensorRT-Edge-LLM/tests/test_cases/` (e.g., `llm_basic.json`) instead of creating your own.

Run inference:

```bash
cd ~/TensorRT-Edge-LLM

./build/examples/llm/llm_inference \
    --engineDir $WORKSPACE_DIR/$MODEL_NAME/engines \
    --inputFile $WORKSPACE_DIR/input.json \
    --outputFile $WORKSPACE_DIR/output.json
```

Verify the output:

```bash
# View the model response
cat $WORKSPACE_DIR/output.json
```

You should see a JSON response with the model's answer, similar to:

```json
{
  "responses": [
    {
      "text": "The capital of the United States is Washington, D.C.",
      "finish_reason": "stop"
    }
  ]
}
```

**Success!** 🎉 You've successfully run LLM inference on your edge device!

---

## Next Steps

**For more advanced workflows, see the example guides:**
- **[VLM Inference](../examples/vlm.md)** - Vision-language models with image understanding
- **[Speculative Decoding](../examples/speculative-decoding.md)** - Speculative decoding for LLM and VLM
- **[Phi-4-Multimodal](../examples/phi4.md)** - Phi-4 Multimodal
- **[ASR](../examples/asr.md)** - Automatic speech recognition
- **[MoE](../examples/moe.md)** - Mixture of Experts models (CPU-only export, Qwen3-30B-A3B-GPTQ-Int4)
- **[TTS](../examples/tts.md)** - Text-to-speech synthesis

**Checkpoint-Based Loader:** For detailed documentation on the recommended export pipeline, pre-quantized checkpoint support, and migration from the legacy tools, see [Checkpoint-Based Model Loader](../../developer_guide/software-design/llm-loader.md).

**Quantization:** To create quantized checkpoints for `llm_loader`, see [Quantization](../features/quantization.md).

**Experimental Python API and Server:** To use the vLLM-style high-level Python API or an OpenAI-compatible chat server, see [Experimental High-Level Python API and Server](../examples/experimental-server.md).

**Input Format:** Our format matches closely with the OpenAI API format. See [Input Format Guide](../format/input-format.md) for detailed specifications. Example input files are available in `tests/test_cases/` (e.g., `llm_basic.json`, `vlm_basic.json`).
