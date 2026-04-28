# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Audio model export functionality for TensorRT Edge-LLM.

This module provides functions to export audio components of multimodal models
(Qwen3-Omni, Qwen3-TTS, Qwen3-ASR) to ONNX format.
"""

import json
import os
from typing import Optional

import torch

from ..llm_models.model_utils import load_hf_model
from .config_export import export_audio_config, export_code2wav_config


def audio_export(model_dir: str,
                 output_dir: str,
                 dtype: str,
                 device: str = "cuda",
                 export_models: str = None,
                 quantization: Optional[str] = None,
                 dataset_dir: str = "openslr/librispeech_asr") -> str:
    """
    Export audio model using the appropriate wrapper based on model architecture.
    
    This function loads a multimodal model, extracts its audio component, wraps it
    in the appropriate model wrapper, applies quantization if requested, and exports
    it to ONNX format.
    
    Args:
        model_dir: Directory containing the torch model
        output_dir: Directory to save the exported ONNX model
        dtype: Data type for export (currently only "fp16" supported)
        device: Device to load the model on (default: "cuda", options: cpu, cuda, cuda:0, cuda:1, etc.)
        export_models: Comma-separated list of models to export for Qwen3-Omni (e.g., 'audio_encoder', 'code2wav', or both. Default is to export both models)
        quantization: Quantization method for audio encoder ("fp8" or None).
            For Qwen3-Omni, this only applies to `audio_encoder`; `code2wav`
            is always exported in FP16.
        dataset_dir: HuggingFace dataset identifier for quantization calibration data
    Returns:
        str: Path to the output directory where the exported model is saved
    
    Raises:
        ValueError: If unsupported dtype is provided
        ValueError: If unsupported model type is detected
    """
    # Validate input parameters
    assert dtype == "fp16", f"Only fp16 is supported for dtype. You passed: {dtype}"
    assert quantization in [
        "fp8", None
    ], f"Only fp8 or None is supported for quantization. You passed: {quantization}"
    if not os.path.isdir(model_dir):
        print(f"Warning: model_dir '{model_dir}' is not a local directory. "
              "The model will be downloaded from the HuggingFace Hub.")

    # Load the model and processor
    try:
        model, _, _ = load_hf_model(model_dir, dtype, device)
    except Exception as e:
        raise ValueError(f"Could not load model from {model_dir}. Error: {e}")

    model_type = model.config.model_type
    torch_dtype = torch.float16

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Detect model architecture and use appropriate wrapper
    if model_type == 'qwen3_omni':
        print(f"Exporting Qwen3-Omni audio model from {model_dir}")

        # Parse export_models parameter
        valid_models = {'audio_encoder', 'code2wav'}
        if export_models is None:
            models_to_export = valid_models
        else:
            models_to_export = set(m.strip() for m in export_models.split(','))
            invalid_models = models_to_export - valid_models
            if invalid_models:
                raise ValueError(f"Invalid export_models: {invalid_models}. "
                                 f"Valid options are: {valid_models}")

        # Export audio_encoder if requested
        if 'audio_encoder' in models_to_export:
            from tensorrt_edgellm.audio_models.qwen3_omni_model import (
                Qwen3OmniAudioEncoderPatch, export_qwen3_omni_audio)

            wrapped_model = Qwen3OmniAudioEncoderPatch._from_config(
                model.thinker.audio_tower.config,
                torch_dtype=torch_dtype,
            )
            wrapped_model.load_state_dict(
                model.thinker.audio_tower.state_dict())
            wrapped_model.eval().to(device)

            if quantization == "fp8":
                from tensorrt_edgellm.quantization.audio_quantization import \
                    quantize_audio
                wrapped_model = quantize_audio(wrapped_model, quantization,
                                               dataset_dir)

            audio_encoder_output_dir = os.path.join(output_dir,
                                                    'audio_encoder')
            export_qwen3_omni_audio(wrapped_model, audio_encoder_output_dir,
                                    torch_dtype)
            print(f"Exported audio_encoder to {audio_encoder_output_dir}")

            # Export model configuration to JSON
            config_dict = export_audio_config(model.thinker.config)
            with open(os.path.join(audio_encoder_output_dir, "config.json"),
                      "w") as f:
                json.dump(config_dict, f, indent=2)

        # Export code2wav if requested
        if 'code2wav' in models_to_export:
            from tensorrt_edgellm.audio_models.qwen3_omni_model import (
                Qwen3OmniCode2WavModelPatch, export_qwen3_omni_code2wav)

            wrapped_code2wav = Qwen3OmniCode2WavModelPatch._from_config(
                model.code2wav.config,
                torch_dtype=torch_dtype,
            )
            wrapped_code2wav.load_state_dict(model.code2wav.state_dict())
            wrapped_code2wav.eval().to(device)
            code2wav_output_dir = os.path.join(output_dir, 'code2wav')
            export_qwen3_omni_code2wav(wrapped_code2wav, code2wav_output_dir)
            print(f"Exported code2wav to {code2wav_output_dir}")

            # Export model configuration to JSON
            config_dict = export_code2wav_config(model.config)
            with open(os.path.join(code2wav_output_dir, "config.json"),
                      "w") as f:
                json.dump(config_dict, f, indent=2)
    elif model_type == 'qwen3_tts':
        if quantization is not None:
            raise ValueError(
                "Quantization is not supported for Qwen3-TTS audio models.")
        print(f"Exporting Qwen3-TTS audio models from {model_dir}")

        valid_models = {'tokenizer_decoder', 'speaker_encoder'}
        if export_models is None:
            models_to_export = {'tokenizer_decoder'}
        else:
            models_to_export = set(m.strip() for m in export_models.split(','))
            invalid_models = models_to_export - valid_models
            if invalid_models:
                raise ValueError(f"Invalid export_models: {invalid_models}. "
                                 f"Valid options are: {valid_models}")

        if 'tokenizer_decoder' in models_to_export:
            from qwen_tts.core.tokenizer_12hz.configuration_qwen3_tts_tokenizer_v2 import \
                Qwen3TTSTokenizerV2Config
            # speech_tokenizer is None after our load_hf_model (we skip the
            # overridden from_pretrained to avoid feature_extractor issues).
            # Load the tokenizer model directly from the HF sub-directory.
            from qwen_tts.core.tokenizer_12hz.modeling_qwen3_tts_tokenizer_v2 import \
                Qwen3TTSTokenizerV2Model
            from transformers import AutoConfig, AutoModel

            from tensorrt_edgellm.audio_models.qwen3_tts_audio_model import \
                export_qwen3_tts_tokenizer_decoder
            AutoConfig.register("qwen3_tts_tokenizer_12hz",
                                Qwen3TTSTokenizerV2Config)
            AutoModel.register(Qwen3TTSTokenizerV2Config,
                               Qwen3TTSTokenizerV2Model)
            tokenizer_subdir = os.path.join(model_dir, "speech_tokenizer")
            if not os.path.isdir(tokenizer_subdir):
                raise ValueError(
                    "Qwen3-TTS export requires a local speech_tokenizer directory at "
                    f"{tokenizer_subdir}")
            tokenizer_model = Qwen3TTSTokenizerV2Model.from_pretrained(
                tokenizer_subdir, torch_dtype=torch_dtype).to(device)
            decoder = tokenizer_model.decoder

            decoder_output_dir = os.path.join(output_dir, 'tokenizer_decoder')
            export_qwen3_tts_tokenizer_decoder(decoder, decoder_output_dir,
                                               torch_dtype)

        if 'speaker_encoder' in models_to_export:
            if model.speaker_encoder is None:
                print(
                    "Warning: speaker_encoder is None (tts_model_type != 'base'), skipping"
                )
            else:
                from tensorrt_edgellm.audio_models.qwen3_tts_audio_model import \
                    export_qwen3_tts_speaker_encoder
                spk_output_dir = os.path.join(output_dir, 'speaker_encoder')
                export_qwen3_tts_speaker_encoder(
                    model.speaker_encoder, model.config.speaker_encoder_config,
                    spk_output_dir, torch_dtype)

    elif model_type == 'qwen3_asr':
        print(f"Exporting Qwen3-ASR audio model from {model_dir}")
        from tensorrt_edgellm.audio_models.qwen3_asr_model import (
            Qwen3ASRModelPatch, export_qwen3_asr_audio)
        wrapped_asr = Qwen3ASRModelPatch._from_config(
            model.thinker.audio_tower.config,
            torch_dtype=torch_dtype,
        )
        wrapped_asr.load_state_dict(model.thinker.audio_tower.state_dict())
        wrapped_asr.eval().to(device)

        if quantization == "fp8":
            from tensorrt_edgellm.quantization.audio_quantization import \
                quantize_audio
            wrapped_asr = quantize_audio(wrapped_asr, quantization,
                                         dataset_dir)

        export_qwen3_asr_audio(wrapped_asr, output_dir, torch_dtype)
        print(f"Exported ASR audio encoder to {output_dir}")

        # Export model configuration to JSON
        config_dict = export_audio_config(model.thinker.config)
        with open(os.path.join(output_dir, "config.json"), "w") as f:
            json.dump(config_dict, f, indent=2)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    print(
        f"Audio export completed for {model_type} with dtype={dtype}, quantization={quantization}, device={device}"
    )
    print(f"Exported to: {output_dir}")
    return output_dir


def export_code2wav(model_dir: str,
                    output_dir: str,
                    dtype: str,
                    device: str = "cuda") -> str:
    """
    Export Code2Wav vocoder for Qwen3-Omni audio generation.
    
    Code2Wav is a CNN-based neural vocoder that converts RVQ audio codes
    to waveform. It is part of the Talker audio generation pipeline.
    
    Args:
        model_dir: Directory containing the torch model
        output_dir: Directory to save the exported ONNX model
        dtype: Data type for export (currently only "fp16" supported)
        device: Device to load the model on (default: "cuda", options: cpu, cuda, cuda:0, cuda:1, etc.)
    
    Returns:
        str: Path to the output directory where the exported model is saved
    
    Raises:
        ValueError: If unsupported dtype is provided
        ValueError: If unsupported model type is detected
    """
    # Validate input parameters
    assert dtype == "fp16", f"Only fp16 is supported for dtype. You passed: {dtype}"

    # Load the model and processor
    try:
        model, _, _ = load_hf_model(model_dir, dtype, device)
    except Exception as e:
        raise ValueError(f"Could not load model from {model_dir}. Error: {e}")

    model_type = model.config.model_type
    torch_dtype = torch.float16

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Detect model architecture and use appropriate wrapper
    if model_type == 'qwen3_omni':
        print(f"Exporting Qwen3-Omni Code2Wav from {model_dir}")

        # Check if code2wav is available
        if not hasattr(model, 'code2wav'):
            raise ValueError(
                "Model does not have code2wav. "
                "Make sure the model was loaded with enable_audio_output=True")

        # Create Qwen3-Omni Code2Wav wrapper model
        from tensorrt_edgellm.audio_models.qwen3_omni_model import (
            Qwen3OmniCode2WavModelPatch, export_qwen3_omni_code2wav)

        wrapped_model = Qwen3OmniCode2WavModelPatch._from_config(
            model.code2wav.config,
            torch_dtype=torch_dtype,
        )
        wrapped_model.load_state_dict(model.code2wav.state_dict())
        wrapped_model.eval().to(device)

        export_qwen3_omni_code2wav(wrapped_model, output_dir)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    # Export model configuration to JSON
    from .config_export import export_code2wav_config
    config_dict = export_code2wav_config(model.config)
    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(config_dict, f, indent=2)

    print(
        f"Code2Wav export completed for {model_type} with dtype={dtype}, device={device}"
    )
    print(f"Exported to: {output_dir}")
    return output_dir
