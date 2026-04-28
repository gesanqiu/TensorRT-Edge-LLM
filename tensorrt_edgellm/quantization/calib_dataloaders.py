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
"""
Calibration dataloaders for quantization.

This module centralises calibration data preparation for LLM backbone
quantization.  Each input modality (text, audio, …) has its own loader so
that the quantization orchestration modules stay focused on algorithm
application rather than data preprocessing.
"""

import io
import os

from datasets import Audio, load_dataset
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer


def get_text_calib_dataloader(
    tokenizer: AutoTokenizer,
    dataset_dir: str,
    batch_size: int,
    num_samples: int,
    max_length: int,
) -> DataLoader:
    """
    Create a text calibration dataloader for LLM quantization.

    Args:
        tokenizer: HuggingFace tokenizer for text processing.
        dataset_dir: Dataset name or local directory path.
        batch_size: Batch size for the dataloader.
        num_samples: Number of samples to use for calibration.
        max_length: Maximum sequence length for tokenization.

    Returns:
        DataLoader yielding batches of ``input_ids`` tensors.

    Raises:
        NotImplementedError: If dataset format is not supported.
    """
    print(f"Loading calibration dataset from {dataset_dir}")
    if "cnn_dailymail" in dataset_dir:
        dataset = load_dataset(dataset_dir, name="3.0.0", split="train")
        dataset = dataset["article"][:num_samples]
    elif os.path.isdir(dataset_dir):
        print(
            f"Recognized local dataset repo {dataset_dir} for calibration; "
            "assuming the calibration data are in the train split and text column."
        )
        dataset = load_dataset(dataset_dir, split="train")
        dataset = dataset["text"][:num_samples]
    else:
        raise NotImplementedError(
            f"Unsupported dataset name or local repo directory: {dataset_dir}."
        )

    # Use tokenizer __call__ for transformers v5-compatible batch tokenization.
    batch_encoded = tokenizer(dataset,
                              return_tensors="pt",
                              padding=True,
                              truncation=True,
                              max_length=max_length)

    return DataLoader(batch_encoded["input_ids"],
                      batch_size=batch_size,
                      shuffle=False)


class _AudioLLMCalibDataset(Dataset):
    """Lazy dataset that preprocesses LibriSpeech audio into thinker inputs."""

    def __init__(self, data, processor):
        self.data = data
        self.processor = processor

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        import soundfile as sf

        raw = self.data[idx]
        audio_bytes = raw["audio"]["bytes"]
        audio, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)

        if sr != 16000:
            import librosa
            audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)

        messages = [
            {
                "role": "system",
                "content": ""
            },
            {
                "role": "user",
                "content": [{
                    "type": "audio",
                    "audio": ""
                }]
            },
        ]
        text = self.processor.apply_chat_template(messages,
                                                  add_generation_prompt=True,
                                                  tokenize=False)
        inputs = self.processor(text=text, audio=audio, return_tensors="pt")

        return {
            "input_ids": inputs["input_ids"].squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0),
            "input_features": inputs["input_features"].squeeze(0),
            "feature_attention_mask":
            inputs["feature_attention_mask"].squeeze(0),
        }


def get_audio_llm_calib_dataloader(
    model_dir: str,
    dataset_dir: str = "openslr/librispeech_asr",
    num_samples: int = 512,
) -> DataLoader:
    """
    Create an audio calibration dataloader for LLM quantization of ASR models.

    Loads audio samples from LibriSpeech, preprocesses them through the
    Qwen3-ASR processor (Whisper feature extraction + tokenization), and
    returns a DataLoader yielding dicts compatible with the thinker forward.

    Args:
        model_dir: Path to the ASR model (used to load the processor).
        dataset_dir: HuggingFace dataset identifier
            (default: openslr/librispeech_asr).
        num_samples: Number of calibration samples.

    Returns:
        DataLoader yielding dicts with keys: input_ids, attention_mask,
        input_features, feature_attention_mask.
    """
    from qwen_asr.core.transformers_backend.processing_qwen3_asr import \
        Qwen3ASRProcessor

    print(f"Loading audio calibration dataset from {dataset_dir}")
    processor = Qwen3ASRProcessor.from_pretrained(model_dir)

    dataset_stream = load_dataset(dataset_dir,
                                  "clean",
                                  split="test",
                                  streaming=True)
    dataset_stream = dataset_stream.cast_column("audio", Audio(decode=False))
    dataset = list(dataset_stream.take(num_samples))

    calib_dataset = _AudioLLMCalibDataset(dataset, processor)

    # batch_size=1 because audio samples have variable lengths
    return DataLoader(calib_dataset, batch_size=1, shuffle=False)
