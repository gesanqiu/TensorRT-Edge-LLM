# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Shared audio helpers for accuracy benchmarks."""

import os
from pathlib import Path

OMNI_SYSTEM_PROMPT = (
    "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, "
    "capable of perceiving auditory and visual inputs, as well as generating "
    "text and speech.")

_feature_extractor = None


def get_whisper_feature_extractor():
    """Lazy-load the Whisper feature extractor used by benchmark preprocessors."""
    global _feature_extractor
    if _feature_extractor is None:
        from transformers import WhisperFeatureExtractor

        _feature_extractor = WhisperFeatureExtractor(
            feature_size=128,
            sampling_rate=16000,
            hop_length=160,
            n_fft=400,
            return_attention_mask=True,
            padding_value=0.0,
        )
    return _feature_extractor


def save_audio_array_to_safetensors(audio, output_path: str) -> str:
    """Convert a 16 kHz mono waveform array into a mel safetensors file."""
    import numpy as np
    import torch
    from safetensors.torch import save_file

    audio = np.asarray(audio, dtype=np.float32)
    fe = get_whisper_feature_extractor()
    inputs = fe(
        audio,
        sampling_rate=16000,
        return_tensors="np",
        return_attention_mask=True,
        padding=False,
        truncation=False,
    )
    mel = inputs["input_features"][0][np.newaxis, ...]
    tensor = torch.from_numpy(mel).to(torch.float16)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    save_file({"mel_spectrogram": tensor}, output_path)
    return output_path


def preprocess_audio_to_safetensors(wav_path: str, output_dir: str) -> str:
    """Load a wav file, extract mel features, and save them as safetensors."""
    import librosa

    stem = Path(wav_path).stem
    output_path = os.path.join(output_dir, f"{stem}.safetensors")
    if os.path.exists(output_path):
        return output_path

    audio, _ = librosa.load(wav_path, sr=16000, mono=True)
    return save_audio_array_to_safetensors(audio, output_path)
