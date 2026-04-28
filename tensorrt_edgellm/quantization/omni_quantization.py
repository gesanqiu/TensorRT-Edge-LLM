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
"""Qwen3-Omni multimodal calibration for quantization.

Calibrates the Thinker with real audio + image + text inputs so that
every LLM layer sees realistic activation distributions.  The Thinker's
hidden states at ``accept_hidden_layer`` are then projected through
``hidden_projection`` / ``text_projection`` and forwarded through the
Talker, giving it a realistic calibration signal instead of random noise.

Typical usage (called automatically by ``quantize_llm`` when the model
is detected as Qwen3-Omni)::

    python -m tensorrt_edgellm.scripts.quantize_llm \\
        --model_dir /path/to/Qwen3-Omni-4B-Instruct \\
        --output_dir /path/to/output \\
        --quantization nvfp4
"""

import io
import os
from typing import List

import torch
from datasets import (Audio, concatenate_datasets, get_dataset_config_names,
                      load_dataset)
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm

try:
    import soundfile as sf
except ImportError as e:
    raise ImportError("Omni multimodal calibration requires soundfile. "
                      "Install with: pip install soundfile") from e

try:
    import librosa
except ImportError:
    librosa = None

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class OmniMultimodalCalibDataset(Dataset):
    """Mixed-modality calibration dataset for Qwen3-Omni.

    Each item is a dict ready for ``thinker(**item)`` containing a mix of
    audio-only, image-only, and text-only samples so that the Thinker sees
    all modality combinations during calibration.
    """

    def __init__(self, processor, audio_data: List, image_data: List,
                 text_data: List[str]):
        self.processor = processor
        self.samples: List[dict] = []
        for item in audio_data:
            self.samples.append({"type": "audio", "raw": item})
        for item in image_data:
            self.samples.append({"type": "image", "raw": item})
        for text in text_data:
            self.samples.append({"type": "text", "raw": text})

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        kind = sample["type"]
        if kind == "audio":
            return self._process_audio(sample["raw"])
        elif kind == "image":
            return self._process_image(sample["raw"])
        return self._process_text(sample["raw"])

    # -- per-modality helpers ------------------------------------------------

    def _process_audio(self, raw):
        audio_bytes = raw["audio"]["bytes"]
        audio, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if sr != 16000:
            if librosa is None:
                raise ImportError("Audio resampling requires librosa. "
                                  "Install with: pip install librosa")
            audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)

        messages = [{
            "role":
            "user",
            "content": [
                {
                    "type": "audio",
                    "audio": "placeholder"
                },
                {
                    "type": "text",
                    "text": "Describe what you hear."
                },
            ]
        }]
        text = self.processor.apply_chat_template(messages,
                                                  add_generation_prompt=True,
                                                  tokenize=False)
        inputs = self.processor(text=text, audio=[audio], return_tensors="pt")
        return {k: v.squeeze(0) for k, v in inputs.items()}

    def _process_image(self, raw):
        images = [
            v.convert("RGB") for k, v in raw.items()
            if "image" in k and isinstance(v, Image.Image)
        ]
        if not images:
            return self._process_text("Describe the scene.")

        messages = [{
            "role":
            "user",
            "content": [
                {
                    "type": "image"
                },
                {
                    "type": "text",
                    "text": "Describe what you see."
                },
            ]
        }]
        text = self.processor.apply_chat_template(messages,
                                                  add_generation_prompt=True,
                                                  tokenize=False)
        inputs = self.processor(text=text,
                                images=images[:1],
                                return_tensors="pt")
        return {k: v.squeeze(0) for k, v in inputs.items()}

    def _process_text(self, content):
        messages = [{"role": "user", "content": content}]
        text = self.processor.apply_chat_template(messages,
                                                  add_generation_prompt=True,
                                                  tokenize=False)
        inputs = self.processor(text=text, return_tensors="pt")
        return {k: v.squeeze(0) for k, v in inputs.items()}


# ---------------------------------------------------------------------------
# Dataset loader
# ---------------------------------------------------------------------------


def get_omni_multimodal_calib_dataset(
    processor,
    audio_dataset_dir: str = "openslr/librispeech_asr",
    visual_dataset_dir: str = "lmms-lab/MMMU",
    text_dataset_dir: str = "cnn_dailymail",
    num_audio_samples: int = 150,
    num_image_samples: int = 150,
    num_text_samples: int = 200,
) -> OmniMultimodalCalibDataset:
    """Build a mixed multimodal calibration dataset for Qwen3-Omni."""

    print(f"[Omni calib] Loading audio from {audio_dataset_dir}")
    audio_stream = load_dataset(audio_dataset_dir,
                                "clean",
                                split="test",
                                streaming=True)
    audio_stream = audio_stream.cast_column("audio", Audio(decode=False))
    audio_data = list(audio_stream.take(num_audio_samples))

    print(f"[Omni calib] Loading images from {visual_dataset_dir}")
    if "lmms-lab/MMMU" in visual_dataset_dir:
        image_dataset = load_dataset(visual_dataset_dir, split="dev")
    elif "MMMU" in visual_dataset_dir:
        configs = get_dataset_config_names(visual_dataset_dir)
        image_dataset = concatenate_datasets([
            load_dataset(visual_dataset_dir, c, split="dev") for c in configs
        ])
    else:
        image_dataset = load_dataset(visual_dataset_dir, split="dev")
    image_data = list(
        image_dataset.select(range(min(num_image_samples,
                                       len(image_dataset)))))

    print(f"[Omni calib] Loading text from {text_dataset_dir}")
    if "cnn_dailymail" in text_dataset_dir:
        ds = load_dataset(text_dataset_dir, name="3.0.0", split="train")
        text_data = ds["article"][:num_text_samples]
    elif os.path.isdir(text_dataset_dir):
        ds = load_dataset(text_dataset_dir, split="train")
        text_data = ds["text"][:num_text_samples]
    else:
        text_data = ["Describe the weather today."] * num_text_samples

    return OmniMultimodalCalibDataset(processor, audio_data, image_data,
                                      text_data)


# ---------------------------------------------------------------------------
# Calibration loop
# ---------------------------------------------------------------------------


def omni_multimodal_calib_loop(
    model,
    calib_dataset: OmniMultimodalCalibDataset,
    accept_hidden_layer: int = 14,
):
    """Calibration forward-loop for ``mtq.quantize`` on a Qwen3-Omni model.

    For each sample the Thinker is run with ``output_hidden_states=True``.
    The hidden states at *accept_hidden_layer* are projected into the Talker
    space so the Talker sees realistic ``inputs_embeds``.

    Args:
        model: ``Qwen3OmniForConditionalGeneration`` (the top-level HF model).
        calib_dataset: Mixed-modality dataset.
        accept_hidden_layer: Which Thinker layer feeds the Talker (default 14).
    """
    device = next(model.parameters()).device
    has_talker = hasattr(model, "has_talker") and model.has_talker

    skipped = 0
    for i in tqdm(range(len(calib_dataset)),
                  desc="Calibrating Omni (multimodal)"):
        try:
            data = calib_dataset[i]
        except Exception as e:
            skipped += 1
            if skipped <= 3:
                print(f"Skipping sample {i}: {e}")
            continue
        data = {
            k:
            v.unsqueeze(0).to(
                device,
                dtype=model.thinker.dtype if v.is_floating_point() else None)
            for k, v in data.items()
        }

        thinker_out = model.thinker(**data, output_hidden_states=True)

        if not has_talker:
            continue

        all_hidden = thinker_out.hidden_states
        if all_hidden is None or len(all_hidden) <= accept_hidden_layer:
            continue

        thinker_hidden = all_hidden[accept_hidden_layer]
        thinker_embed = all_hidden[0]

        input_ids = data.get("input_ids")
        audio_tok = getattr(model.config.thinker_config, "audio_token_id", -1)
        image_tok = getattr(model.config.thinker_config, "image_token_id", -1)
        mm_mask = torch.zeros_like(input_ids, dtype=torch.bool)
        if audio_tok >= 0:
            mm_mask |= (input_ids == audio_tok)
        if image_tok >= 0:
            mm_mask |= (input_ids == image_tok)

        seq_len = thinker_hidden.shape[1]
        talker_dim = model.talker.config.text_config.hidden_size
        inputs_embeds = torch.empty(1,
                                    seq_len,
                                    talker_dim,
                                    dtype=model.talker.dtype,
                                    device=device)

        if mm_mask.any():
            inputs_embeds[mm_mask] = model.talker.hidden_projection(
                thinker_hidden[mm_mask])
        text_mask = ~mm_mask
        if text_mask.any():
            inputs_embeds[text_mask] = model.talker.text_projection(
                thinker_embed[text_mask])

        tc = model.talker.config.text_config
        talker_ids = torch.randint(0,
                                   tc.vocab_size, (1, seq_len),
                                   device=device)
        attn_mask = torch.ones(1, seq_len, dtype=torch.long, device=device)

        model.talker(inputs_embeds=inputs_embeds,
                     attention_mask=attn_mask,
                     talker_input_ids=talker_ids)

    if skipped:
        print(
            f"Omni calibration: skipped {skipped}/{len(calib_dataset)} samples"
        )
