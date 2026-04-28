# TTS (Text-to-Speech)

This guide covers the full pipeline for running Qwen3-TTS: export on x86 host, engine build on device, and inference.

**Supported model:** [Qwen3-TTS-12Hz-1.7B-CustomVoice](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice)

> **Note:** Unlike Qwen3-Omni, Qwen3-TTS has no Thinker or visual encoder. The text embedding is self-contained in the Talker and exported as `text_embedding.safetensors`.

> **Prerequisites:** Complete the [Installation Guide](../getting_started/installation.md) before proceeding.

---

## Part 0: Install TTS Dependency (x86 Host)

The export pipeline loads the Qwen3-TTS model via the `qwen-tts` package. Install it before exporting:

> **Warning:** Installing `qwen-tts` may break package versions in your current environment (e.g. `transformers`, `torch`). **Use a dedicated virtual environment for Qwen3-TTS export only** — do not share it with other model workflows. We need this special workflow until Qwen3-TTS gets merged into HuggingFace transformers.

```bash
cd TensorRT-Edge-LLM
python3 -m venv venv-qwen3-tts
source venv-qwen3-tts/bin/activate
pip3 install .              # install TensorRT Edge-LLM export dependencies
pip3 install qwen-tts       # install qwen3-tts and its required dependencies.
```

---

## Part 1: Export on x86 Host

Qwen3-TTS has three components. Export them separately.

### Export Talker and CodePredictor (LLM components)

```bash
export WORKSPACE_DIR=$HOME/tensorrt-edgellm-workspace
export TTS_MODEL=Qwen3-TTS-12Hz-1.7B-CustomVoice
export ONNX_OUTPUT_DIR=$WORKSPACE_DIR/$TTS_MODEL/onnx
export TTS_CHAT_TEMPLATE=./tensorrt_edgellm/chat_templates/templates/qwen3tts.json

# Exports talker/ and code_predictor/ subdirectories
tensorrt-edgellm-export-llm \
    --model_dir Qwen/$TTS_MODEL \
    --output_dir $ONNX_OUTPUT_DIR \
    --chat_template $TTS_CHAT_TEMPLATE \
    --export_models talker,code_predictor
```

### Export Code2Wav vocoder

```bash
# Exports tokenizer_decoder/ subdirectory
tensorrt-edgellm-export-audio \
    --model_dir Qwen/$TTS_MODEL \
    --output_dir $ONNX_OUTPUT_DIR \
    --export_models tokenizer_decoder
```

### Expected Export Output

```
$ONNX_OUTPUT_DIR/
├── llm/
│   ├── talker/
│   │   ├── model.onnx + onnx_model.data   # Talker ONNX
│   │   ├── config.json                     # model_type: qwen3_tts_talker
│   │   ├── embedding.safetensors           # codec embedding
│   │   ├── text_embedding.safetensors      # TTS-only (no Thinker)
│   │   └── text_projection.safetensors
│   ├── code_predictor/
│   │   ├── model.onnx + onnx_model.data   # CodePredictor ONNX
│   │   ├── config.json
│   │   ├── codec_embeddings.safetensors    # 15 embeddings
│   │   ├── lm_heads.safetensors           # 15 lm_heads
│   │   └── small_to_mtp_projection.safetensors  # if not Identity
│   ├── tokenizer_config.json              # at top level (no thinker/)
│   ├── processed_chat_template.json       # chat template for runtime
│   └── tokenizer files                    # tokenizer vocab/merges copied from model
└── audio/
    ├── tokenizer_decoder/
    │   ├── model.onnx + onnx_model.data   # Code2Wav vocoder
    │   └── config.json
    └── speaker_encoder/                    # Base models only
        ├── model.onnx + onnx_model.data
        └── config.json
```

### Transfer to Device

```bash
scp -r $ONNX_OUTPUT_DIR <user>@<device>:~/tensorrt-edgellm-workspace/$TTS_MODEL/
```

---

## Part 2: Build Engines

Three engine builds are required. Run these on the edge device.

```bash
cd ~/TensorRT-Edge-LLM
export WORKSPACE_DIR=$HOME/tensorrt-edgellm-workspace
export TTS_MODEL=Qwen3-TTS-12Hz-1.7B-CustomVoice
export ONNX=$WORKSPACE_DIR/$TTS_MODEL/onnx
export ENG=$WORKSPACE_DIR/$TTS_MODEL/engines

# 1. Build Talker LLM engine
./build/examples/llm/llm_build \
    --onnxDir $ONNX/llm/talker \
    --engineDir $ENG/talker \
    --maxInputLen 4096 \
    --maxKVCacheCapacity 4096 \
    --maxBatchSize 1

# 2. Build CodePredictor LLM engine
./build/examples/llm/llm_build \
    --onnxDir $ONNX/llm/code_predictor \
    --engineDir $ENG/code_predictor \
    --maxInputLen 4096 \
    --maxKVCacheCapacity 4096 \
    --maxBatchSize 1

# 3. Build Code2Wav engine
./build/examples/multimodal/audio_build \
    --onnxDir $ONNX/audio/tokenizer_decoder \
    --engineDir $ENG/code2wav
```

> **Note:** `--maxBatchSize` must be set to **1**. The Qwen3-TTS ONNX export uses a fixed batch size of 1; larger values are not supported.

Build time: < 5 minutes

### Copy Tokenizer and Chat Template Files to Engine Folder

The runtime loads the tokenizer and chat template from the engine directory. Copy the required files from the ONNX export output:

```bash
cp $ONNX/llm/*.json $ENG/   # includes tokenizer_config.json, processed_chat_template.json, etc.
```

---

## Part 3: Run Inference

### Input File Format

Each request specifies a `messages` array and an optional per-request `speaker`. If omitted, the top-level `speaker` default is used.

```json
{
    "talker_temperature": 0.9,
    "talker_top_k": 50,
    "repetition_penalty": 1.05,
    "speaker": "ryan",
    "requests": [
        {
            "messages": [{"role": "assistant", "content": "Hello, how can I help you today?"}]
        },
        {
            "speaker": "serena",
            "messages": [{"role": "assistant", "content": "The weather is sunny and warm."}]
        }
    ]
}
```

**Available speakers:** `ryan`, `serena`, `aiden`, `vivian`, `dylan`, `eric`, `uncle_fu`, `ono_anna`, `sohee`

**Sampling parameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `talker_temperature` | 0.9 | Sampling temperature |
| `talker_top_k` | 50 | Top-K sampling |
| `talker_top_p` | 1.0 | Top-P sampling |
| `repetition_penalty` | 1.05 | Penalize repeated codec tokens |
| `max_audio_length` | 4096 | Max codec frames per request |
| `speaker` | config default | Top-level speaker fallback |

### Run

```bash
cd /path/to/TensorRT-Edge-LLM

./build/examples/omni/qwen3_tts_inference \
    --talkerEngineDir   $ENG/talker \
    --code2wavEngineDir $ENG/code2wav \
    --tokenizerDir      $ENG \
    --inputFile         input.json \
    --outputFile        output.json \
    --outputAudioDir    ./audio_output
```

Generated `.wav` files are named `audio_req{N}.wav` (one per request). The output JSON records per-request metadata: audio file path, sample count, duration, and RVQ code file path.

### Output JSON Example

```json
{
  "responses": [
    {
      "request_idx": 0,
      "messages": [{"role": "assistant", "content": "Hello, how can I help you today?"}],
      "audio_file": "./audio_output/audio_req0.wav",
      "audio_samples": 120960,
      "audio_sample_rate": 24000,
      "audio_duration_ms": 5040,
      "rvq_file": "./audio_output/rvq_req0.safetensors"
    }
  ]
}
```

---

## Notes

- `--code2wavEngineDir` is optional: auto-detected as `parent(talkerEngineDir)/code2wav` if not set.
- RVQ code files (`.safetensors`) are saved alongside audio when `--outputAudioDir` is set and can be used to re-synthesize audio without re-running the TTS model.
