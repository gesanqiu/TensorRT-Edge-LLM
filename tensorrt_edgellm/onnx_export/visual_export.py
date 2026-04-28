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
Visual model export functionality for TensorRT Edge-LLM.

This module provides functions to export visual components of multimodal models
to ONNX format with optional quantization support.
"""

import json
import os
import shutil
from typing import Optional

import torch

from tensorrt_edgellm.quantization.visual_quantization import quantize_visual
# Import visual model wrappers
from tensorrt_edgellm.visual_models.internvl3_model import (
    InternVLVisionModel, export_internvl3_visual)
from tensorrt_edgellm.visual_models.phi4mm_model import (Phi4MMVisionModel,
                                                         export_phi4mm_visual)
from tensorrt_edgellm.visual_models.qwen2_5_vl_model import (
    Qwen2_5_VisionTransformerPretrainedModelPatch, export_qwen2_5_vl_visual)
from tensorrt_edgellm.visual_models.qwen2_vl_model import (
    Qwen2VisionTransformerPretrainedModelPatch, export_qwen2_vl_visual)
from tensorrt_edgellm.visual_models.qwen3_vl_model import (
    Qwen3VLVisionModelPatch, export_qwen3_vl_visual)

from ..llm_models.model_utils import load_hf_model
from .config_export import export_vision_config


def _export_qwen_visual(model, model_type: str, model_dir: str,
                        output_dir: str, torch_dtype: torch.dtype, device: str,
                        quantization: Optional[str], processor,
                        dataset_dir: str) -> None:
    """
    Export visual models in the Qwen family (qwen2/qwen2.5/qwen3/qwen3-omni/qwen3.5).
    """
    visual_model = model.model.visual if model_type != 'qwen3_omni' else model.thinker.visual

    # Quantize the original visual model first
    if quantization == "fp8":
        # Qwen2.5-VL 3B VIT has overflow issue with FP16 and only happens in /blocks.31/mlp/down_proj
        # Apply workaround to cast /blocks.31/mlp/down_proj to FP32 to avoid overflow in quantization.
        if model_type == 'qwen2_5_vl' and visual_model.config.out_hidden_size == 2048:
            from tensorrt_edgellm.visual_models.qwen2_5_vl_model import (
                Qwen2_5_VLMLPPatch, Qwen2_5_VLPatchMergerWAR)
            last_block = visual_model.blocks[-1]
            last_block.mlp = Qwen2_5_VLMLPPatch(visual_model.config,
                                                last_block.mlp)
            visual_model.merger = Qwen2_5_VLPatchMergerWAR(
                visual_model.config, visual_model.merger)

        visual_model = quantize_visual(visual_model, quantization, processor,
                                       dataset_dir)

    print(f"Exporting {model_type} visual model from {model_dir}")

    if model_type == 'qwen2_vl':
        wrapped_model = Qwen2VisionTransformerPretrainedModelPatch(
            visual_model)
        wrapped_model.eval().to(device)
        export_qwen2_vl_visual(wrapped_model, output_dir, torch_dtype)
    elif model_type == 'qwen2_5_vl':
        wrapped_model = Qwen2_5_VisionTransformerPretrainedModelPatch(
            visual_model)
        wrapped_model.eval().to(device)
        export_qwen2_5_vl_visual(wrapped_model, output_dir, torch_dtype)
    else:
        # qwen3_vl, qwen3_omni and qwen3_5 share the same visual wrapper/export path
        wrapped_model = Qwen3VLVisionModelPatch(visual_model)
        wrapped_model.eval().to(device)
        export_qwen3_vl_visual(wrapped_model, output_dir, torch_dtype)


def visual_export(model_dir: str,
                  output_dir: str,
                  dtype: str,
                  quantization: Optional[str],
                  dataset_dir: Optional[str] = "lmms-lab/MMMU",
                  device: str = "cuda") -> str:
    """
    Export visual model using the appropriate wrapper based on model architecture.
    
    This function loads a multimodal model, extracts its visual component, wraps it
    in the appropriate model wrapper, applies quantization if requested, and exports
    it to ONNX format.
    
    Args:
        model_dir: Directory containing the torch model
        output_dir: Directory to save the exported ONNX model
        dtype: Data type for export (currently only "fp16" supported)
        quantization: Quantization type ("fp8" or None)
        device: Device to load the model on (default: "cuda", options: cpu, cuda, cuda:0, cuda:1, etc.)
    
    Returns:
        str: Path to the output directory where the exported model is saved
    
    Raises:
        ValueError: If unsupported dtype or quantization is provided
        ValueError: If unsupported model type is detected
    """
    # Convert dtype string to torch dtype
    if dtype == "fp16":
        torch_dtype = torch.float16
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")

    # Validate quantization mode
    assert quantization in [
        "fp8", None
    ], f"Only fp8 or None is supported for quantization. You passed: {quantization}"

    # Load the model and processor
    try:
        model, _, processor = load_hf_model(model_dir, dtype, device)
    except Exception as e:
        raise ValueError(f"Could not load model from {model_dir}. Error: {e}")

    # Get visual model from the multimodal model
    model_type = model.config.model_type

    # For Alpamayo 1, extract the VLM
    if model_type == 'alpamayo_r1':
        model = model.vlm
        model_type = model.config.model_type

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    if model_type in [
            'qwen2_vl',
            'qwen2_5_vl',
            'qwen3_vl',
            'qwen3_omni',
            'qwen3_5',
    ]:
        _export_qwen_visual(model,
                            model_type,
                            model_dir,
                            output_dir,
                            torch_dtype=torch_dtype,
                            device=device,
                            quantization=quantization,
                            processor=processor,
                            dataset_dir=dataset_dir)

    elif model_type == 'internvl':
        print(f"Exporting InternVL3 visual model from {model_dir}")
        # Create InternVL3 wrapper model
        wrapped_model = InternVLVisionModel(model)
        wrapped_model.eval().to(device)

        # Apply quantization to wrapped model if requested
        if quantization == "fp8":
            wrapped_model = quantize_visual(wrapped_model, quantization,
                                            processor.image_processor,
                                            dataset_dir)

        # Export using the wrapper's export function
        export_internvl3_visual(wrapped_model, output_dir, torch_dtype)

    elif model_type == 'phi4mm':
        print(f"Exporting Phi4MM visual model from {model_dir}")
        # Create Phi4MM wrapper model
        wrapped_model = Phi4MMVisionModel(model)
        wrapped_model.eval().to(device)

        # Apply quantization to wrapped model if requested
        if quantization == "fp8":
            wrapped_model = quantize_visual(wrapped_model, quantization,
                                            processor.image_processor,
                                            dataset_dir)
        export_phi4mm_visual(wrapped_model, output_dir, torch_dtype)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    # Export model configuration to JSON
    config_dict = export_vision_config(
        model.config.thinker_config if model_type ==
        'qwen3_omni' else model.config)
    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(config_dict, f, indent=2)

    # Export processor configuration to JSON if exists
    if processor is not None:
        # FIXME: If not commented out, the export will fail for internvl
        # # Phi4MMProcessor may not define audio_tokenizer, but transformers'
        # # save_pretrained expects the attribute.
        # if not hasattr(processor, "audio_tokenizer"):
        #     processor.audio_tokenizer = None
        processor.save_pretrained(output_dir)

        # Transformers v5 may save processor metadata as processor_config.json.
        # Keep preprocessor_config.json for C++ runtime compatibility.
        processor_config_path = os.path.join(output_dir,
                                             "processor_config.json")
        preprocessor_config_path = os.path.join(output_dir,
                                                "preprocessor_config.json")
        if os.path.exists(processor_config_path
                          ) and not os.path.exists(preprocessor_config_path):
            shutil.copyfile(processor_config_path, preprocessor_config_path)

    print(
        f"Visual export completed for {model_type} with dtype={dtype}, quantization={quantization}, device={device}"
    )
    print(f"Exported to: {output_dir}")
    return output_dir
