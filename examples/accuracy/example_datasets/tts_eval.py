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
TTS evaluation datasets for TensorRT Edge-LLM.

Generates input JSON for llm_inference (with --enableAudioOutput) and a
reference JSON for calculate_tts_score.py.

Supports two benchmark datasets:

  1. Seed-TTS-eval (ByteDance)
     Official zero-shot TTS benchmark used in Qwen Omni's technical report.
     Test sets: test-zh (2000 samples), test-en (1000 samples), test-hard.
     Download: https://drive.google.com/file/d/1GlSjVfSHkW3-leKKBlfrjuuTGqQ_xaLP

  2. MiniMax Multilingual Test Set
     24-language TTS benchmark with 100 sentences per language.
     HuggingFace: MiniMaxAI/TTS-Multilingual-Test-Set

Output:
    {output_dir}/tts_eval_dataset.json  - input for llm_inference
    {output_dir}/tts_eval_reference.json - reference for calculate_tts_score.py
"""

import json
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from audio_eval_utils import OMNI_SYSTEM_PROMPT
from edgellm_dataset import DatasetConfig

_SEED_TTS_LANGUAGE_ALIASES = {
    "zh": "zh",
    "chinese": "zh",
    "en": "en",
    "english": "en",
}

_MINIMAX_LANGUAGE_TO_CODE = {
    "chinese": "zh",
    "english": "en",
    "cantonese": "zh",
    "japanese": "ja",
    "korean": "ko",
    "arabic": "ar",
    "spanish": "es",
    "turkish": "tr",
    "indonesian": "id",
    "portuguese": "pt",
    "french": "fr",
    "italian": "it",
    "dutch": "nl",
    "vietnamese": "vi",
    "german": "de",
    "russian": "ru",
    "ukrainian": "uk",
    "thai": "th",
    "polish": "pl",
    "romanian": "ro",
    "greek": "el",
    "czech": "cs",
    "finnish": "fi",
    "hindi": "hi",
}

_MINIMAX_LANGUAGE_ALIASES = {
    **{
        language: language
        for language in _MINIMAX_LANGUAGE_TO_CODE
    },
    **{
        code: language
        for language, code in _MINIMAX_LANGUAGE_TO_CODE.items() if code != "zh"
    },
    "zh": "chinese",
}


def _normalize_seed_tts_language(language: str) -> str:
    """Normalize Seed-TTS-eval language to the short code used downstream."""
    normalized = language.strip().lower()
    language_code = _SEED_TTS_LANGUAGE_ALIASES.get(normalized)
    if language_code is None:
        raise ValueError(
            "Unsupported SeedTTSEval language "
            f"'{language}'. Use 'zh'/'chinese' or 'en'/'english'.")
    return language_code


def _normalize_minimax_language(language: str) -> str:
    """Normalize MiniMax language to the canonical dataset file name."""
    normalized = language.strip().lower()
    dataset_language = _MINIMAX_LANGUAGE_ALIASES.get(normalized)
    if dataset_language is None:
        supported = ", ".join(sorted(_MINIMAX_LANGUAGE_TO_CODE))
        raise ValueError(
            "Unsupported MiniMaxMultilingual language "
            f"'{language}'. Use a short code like 'en'/'zh' or one of: "
            f"{supported}.")
    return dataset_language


def _save_dataset(
    requests: List[Dict[str, Any]],
    references: List[Dict[str, Any]],
    output_dir: str,
    config: DatasetConfig,
) -> Tuple[str, str]:
    """Save input and reference JSONs."""
    input_data = {
        "batch_size": config.batch_size,
        "temperature": config.temperature,
        "top_p": config.top_p,
        "top_k": config.top_k,
        "max_generate_length": config.max_generate_length,
        # Keep talker decoding fixed to benchmark defaults so prepared
        # datasets stay comparable across runs.
        "repetition_penalty": 1.1,
        "talker_temperature": 0.9,
        "talker_top_k": 40,
        "talker_top_p": 0.8,
        "requests": requests,
    }
    ref_data = {"requests": references}

    input_path = os.path.join(output_dir, "tts_eval_dataset.json")
    ref_path = os.path.join(output_dir, "tts_eval_reference.json")

    with open(input_path, 'w', encoding='utf-8') as f:
        json.dump(input_data, f, indent=4, ensure_ascii=False)
    with open(ref_path, 'w', encoding='utf-8') as f:
        json.dump(ref_data, f, indent=4, ensure_ascii=False)

    print(f"Saved {len(requests)} requests to {input_path}")
    print(f"Saved {len(references)} references to {ref_path}")
    return input_path, ref_path


# ---------------------------------------------------------------------------
# Seed-TTS-eval
# ---------------------------------------------------------------------------


def convert_seed_tts_eval(
    meta_file: str,
    output_dir: str,
    config: DatasetConfig,
    language: str = "zh",
    max_samples: Optional[int] = None,
) -> Tuple[str, str]:
    """
    Convert Seed-TTS-eval meta file to Edge LLM format.

    Meta file format (pipe-separated):
        sample_id | prompt_text | prompt_audio | synth_text

    Args:
        meta_file: Path to meta.lst / hardcase.lst.
        output_dir: Output directory.
        config: DatasetConfig with inference parameters.
        language: 'zh' or 'en'.
        max_samples: Limit samples (for quick testing).

    Returns:
        Tuple of (input_json_path, reference_json_path).
    """
    os.makedirs(output_dir, exist_ok=True)
    language = _normalize_seed_tts_language(language)
    meta_dir = os.path.dirname(os.path.abspath(meta_file))

    requests = []
    references = []

    with open(meta_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    if max_samples:
        lines = lines[:max_samples]

    for idx, line in enumerate(lines):
        parts = [p.strip() for p in line.strip().split('|')]
        if len(parts) < 4:
            print(f"Warning: skipping malformed line {idx}: {line.strip()}")
            continue

        filename = parts[0]
        _ = parts[1]  # prompt_text (unused by benchmark conversion)
        prompt_audio_rel = parts[2]
        synth_text = parts[3]

        prompt_audio_abs = (
            os.path.join(meta_dir, prompt_audio_rel) if prompt_audio_rel
            and not os.path.isabs(prompt_audio_rel) else prompt_audio_rel)

        # Benchmark prompt is fixed regardless of the runtime being tested:
        # evaluate read-aloud fidelity with the same text-only instruction.
        if language == "zh":
            user_text = (f"请严格按照原文朗读以下内容，不要解释、"
                         f"不要添加任何额外内容：\n{synth_text}")
        else:
            user_text = (f"Read the following text exactly as written. "
                         f"Do not explain or add anything:\n{synth_text}")

        user_content = [{"type": "text", "text": user_text}]

        requests.append({
            "messages": [
                {
                    "role": "system",
                    "content": OMNI_SYSTEM_PROMPT
                },
                {
                    "role": "user",
                    "content": user_content
                },
            ],
        })

        ref_entry = {"reference": synth_text, "id": filename}

        # Seed-TTS-eval hardcase.lst uses sample IDs like "raokouling-0000" in
        # the first column, while the real reference audio path is stored in the
        # third column. Prefer the explicit audio path when present, and keep the
        # legacy fallback for layouts where the wav is named after the sample ID.
        ref_audio_path = None
        if prompt_audio_abs and os.path.exists(prompt_audio_abs):
            ref_audio_path = prompt_audio_abs
        else:
            legacy_gt_audio_path = os.path.join(meta_dir, "wavs",
                                                f"{filename}.wav")
            if os.path.exists(legacy_gt_audio_path):
                ref_audio_path = legacy_gt_audio_path

        if ref_audio_path is not None:
            ref_entry["reference_audio"] = os.path.abspath(ref_audio_path)
        references.append(ref_entry)

    return _save_dataset(requests, references, output_dir, config)


# ---------------------------------------------------------------------------
# MiniMax Multilingual Test Set
# ---------------------------------------------------------------------------


def convert_minimax_multilingual(
    dataset_dir: str,
    output_dir: str,
    config: DatasetConfig,
    language: str = "en",
    max_samples: Optional[int] = None,
) -> Tuple[str, str]:
    """
    Convert MiniMax Multilingual TTS Test Set to Edge LLM format.

    Dataset structure (HuggingFace: MiniMaxAI/TTS-Multilingual-Test-Set):
        speaker/  - two audio files per language (male + female)
        text/     - {language}.txt with lines: cloning_audio_filename|text

    Args:
        dataset_dir: Path to downloaded dataset root.
        output_dir: Output directory.
        config: DatasetConfig with inference parameters.
        language: Language code or name (e.g. 'en', 'english', 'ja',
            'japanese').
        max_samples: Limit samples.

    Returns:
        Tuple of (input_json_path, reference_json_path).
    """
    os.makedirs(output_dir, exist_ok=True)

    dataset_language = _normalize_minimax_language(language)
    text_file = os.path.join(dataset_dir, "text", f"{dataset_language}.txt")
    if not os.path.exists(text_file):
        raise FileNotFoundError(f"Text file not found: {text_file}")

    with open(text_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    if max_samples:
        lines = lines[:max_samples]

    lang_code = _minimax_lang_to_code(dataset_language)

    requests = []
    references = []

    for idx, line in enumerate(lines):
        line = line.strip()
        if not line or '|' not in line:
            continue

        parts = line.split('|', 1)
        speaker_name = parts[0].strip()
        synth_text = parts[1].strip()

        if lang_code == "zh":
            user_text = (f"请严格按照原文朗读以下内容，不要解释、"
                         f"不要添加任何额外内容：\n{synth_text}")
        else:
            user_text = (f"Read the following text exactly as written. "
                         f"Do not explain or add anything:\n{synth_text}")

        user_content = [{"type": "text", "text": user_text}]

        requests.append({
            "messages": [
                {
                    "role": "system",
                    "content": OMNI_SYSTEM_PROMPT
                },
                {
                    "role": "user",
                    "content": user_content
                },
            ],
        })

        references.append({
            "reference": synth_text,
            "id": f"{dataset_language}_{idx:04d}",
            "speaker": speaker_name,
        })

    return _save_dataset(requests, references, output_dir, config)


def _minimax_lang_to_code(language: str) -> str:
    """Map MiniMax language name or short code to Seed-TTS-eval language code."""
    dataset_language = _normalize_minimax_language(language)
    return _MINIMAX_LANGUAGE_TO_CODE[dataset_language]


# ---------------------------------------------------------------------------
# Entry point for prepare_dataset.py
# ---------------------------------------------------------------------------


def convert_seed_tts_eval_dataset(
    config: DatasetConfig,
    dataset_name_or_dir: str,
    output_dir: str,
    language: str = "zh",
    max_samples: Optional[int] = None,
) -> None:
    """
    Entry point for Seed-TTS-eval via prepare_dataset.py.

    Args:
        config: DatasetConfig from prepare_dataset.py.
        dataset_name_or_dir: Path to meta.lst or hardcase.lst.
        output_dir: Output directory.
        language: 'zh' or 'en'.
        max_samples: Limit samples.
    """
    if not dataset_name_or_dir or not dataset_name_or_dir.endswith(".lst"):
        raise ValueError(
            "SeedTTSEval requires --dataset_name_or_dir pointing to a "
            ".lst file (e.g. zh/meta.lst, en/meta.lst, zh/hardcase.lst)")
    convert_seed_tts_eval(dataset_name_or_dir,
                          output_dir,
                          config,
                          language,
                          max_samples=max_samples)


def convert_minimax_multilingual_dataset(
    config: DatasetConfig,
    dataset_name_or_dir: str,
    output_dir: str,
    language: str = "en",
    max_samples: Optional[int] = None,
) -> None:
    """
    Entry point for MiniMax Multilingual via prepare_dataset.py.

    Args:
        config: DatasetConfig from prepare_dataset.py.
        dataset_name_or_dir: Path to downloaded dataset root directory.
        output_dir: Output directory.
        language: Language code or name (e.g. 'en', 'english', 'zh',
            'chinese').
        max_samples: Limit samples.
    """
    if not dataset_name_or_dir or not os.path.isdir(dataset_name_or_dir):
        raise ValueError(
            f"MiniMaxMultilingual dataset root not found: "
            f"{dataset_name_or_dir!r}. Expected a directory containing "
            f"speaker/ and text/ subdirectories.")
    convert_minimax_multilingual(dataset_name_or_dir,
                                 output_dir,
                                 config,
                                 language,
                                 max_samples=max_samples)
