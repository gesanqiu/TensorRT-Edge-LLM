# ASR (Automatic Speech Recognition)

Complete workflow for speech recognition with audio understanding capabilities.

**Example model:** Qwen3-ASR-0.6B

> **Prerequisites:** Complete the [Installation Guide](../getting_started/installation.md) before proceeding.

---

## Part 0: Install ASR Dependency (x86 Host)

The export pipeline loads the Qwen3-ASR model via the `qwen-asr` package. Install it before exporting:

> **Warning:** Installing `qwen-asr` may break package versions in your current environment (e.g. `transformers`, `torch`). **Use a dedicated virtual environment for Qwen3-ASR export only** — do not share it with other model workflows. We need this special workflow until Qwen3-ASR gets merged into HuggingFace transformers.

```bash
cd TensorRT-Edge-LLM
python3 -m venv venv-qwen3-asr
source venv-qwen3-asr/bin/activate
pip3 install .              # install TensorRT Edge-LLM export dependencies
pip3 install qwen-asr       # install qwen3-asr and its required dependencies.
```

---

## Step 1: Export (x86 Host)

```bash
export WORKSPACE_DIR=$HOME/tensorrt-edgellm-workspace
export MODEL_NAME=Qwen3-ASR-0.6B
mkdir -p $WORKSPACE_DIR
cd $WORKSPACE_DIR

# Export audio encoder
tensorrt-edgellm-export-audio \
  --model_dir Qwen/Qwen3-ASR-0.6B \
  --output_dir $MODEL_NAME/onnx/audio

# Export language model
tensorrt-edgellm-export-llm \
  --model_dir Qwen/Qwen3-ASR-0.6B \
  --output_dir $MODEL_NAME/onnx/llm
```

## Step 2: Transfer to Device

```bash
# Transfer ONNX to device
scp -r $MODEL_NAME/onnx \
  <device_user>@<device_ip>:~/tensorrt-edgellm-workspace/$MODEL_NAME/
```

## Step 3: Build Engines (Thor Device)

```bash
# Set up workspace directory on device
export WORKSPACE_DIR=$HOME/tensorrt-edgellm-workspace
export MODEL_NAME=Qwen3-ASR-0.6B
cd ~/TensorRT-Edge-LLM

# Build language model engine
./build/examples/llm/llm_build \
  --onnxDir $WORKSPACE_DIR/$MODEL_NAME/onnx/llm \
  --engineDir $WORKSPACE_DIR/$MODEL_NAME/engines/llm \
  --maxBatchSize 1 \
  --maxInputLen 1024 \
  --maxKVCacheCapacity 4096

# Build audio encoder engine
./build/examples/multimodal/audio_build \
  --onnxDir $WORKSPACE_DIR/$MODEL_NAME/onnx/audio \
  --engineDir $WORKSPACE_DIR/$MODEL_NAME/engines/audio \
  --minTimeSteps 1000 \
  --maxTimeSteps 3000
```

## Step 4: Preprocess Audio Input (x86 Host or Thor Device)

Audio files must be converted to mel-spectrogram safetensors format before inference:

```bash
cd ~/TensorRT-Edge-LLM

# Convert WAV to safetensors mel-spectrogram format
python -m tensorrt_edgellm.scripts.preprocess_audio \
  --input /path/to/audio.wav \
  --output $WORKSPACE_DIR/audio_input.safetensors
```

**Note:** Supported audio formats include `.wav`, `.mp3`, `.flac`, `.ogg`, and `.m4a`. If preprocessing is done on the x86 host, transfer the output safetensors file to the device before running inference.

## Step 5: Run Inference (Thor Device)

Create an input file `$WORKSPACE_DIR/input_asr.json` (replace `/path/to/audio_input.safetensors` with the actual preprocessed audio file path):

```json
{
    "batch_size": 1,
    "temperature": 1.0,
    "top_p": 1.0,
    "top_k": 50,
    "max_generate_length": 256,
    "requests": [
        {
            "messages": [
                {
                    "role": "system",
                    "content": ""
                },
                {
                    "role": "user",
                    "content": [{"type": "audio", "audio": "/path/to/audio_input.safetensors"}]
                }
            ]
        }
    ]
}
```

Run inference:

```bash
cd ~/TensorRT-Edge-LLM

./build/examples/llm/llm_inference \
  --engineDir $WORKSPACE_DIR/$MODEL_NAME/engines/llm \
  --multimodalEngineDir $WORKSPACE_DIR/$MODEL_NAME/engines/audio \
  --inputFile $WORKSPACE_DIR/input_asr.json \
  --outputFile $WORKSPACE_DIR/output_asr.json
```

Check `output_asr.json` for the speech recognition transcription.
