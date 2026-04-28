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
Auto-dispatch model factory and parameter utilities.

``AutoModel.from_pretrained`` reads a checkpoint config, picks the right model
class, constructs it, and loads weights — the primary entry point for callers.

Custom model classes can be registered via :func:`register_model` to override
the default :class:`~models.default.modeling_default.CausalLM` for a given
``model_type`` string.
"""

from typing import Dict, Type

import torch.nn as nn

from .checkpoint.loader import load_weights
from .config import ModelConfig

__all__ = ["AutoModel", "register_model", "dtype_summary", "param_count"]

_MODEL_REGISTRY: Dict[str, Type[nn.Module]] = {}


def register_model(model_type: str, model_class: Type[nn.Module]) -> None:
    """Register *model_class* as the handler for *model_type*.

    When :meth:`AutoModel.from_pretrained` encounters a checkpoint whose
    ``model_type`` field equals *model_type*, it instantiates *model_class*
    instead of the built-in :class:`~models.default.modeling_default.CausalLM`.

    Args:
        model_type:  Value of ``model_type`` in the checkpoint ``config.json``.
        model_class: ``nn.Module`` subclass; must accept a single
                     :class:`~config.ModelConfig` as its constructor argument.
    """
    _MODEL_REGISTRY[model_type] = model_class


class AutoModel:
    """HuggingFace-style factory that dispatches on ``model_type``."""

    @classmethod
    def from_pretrained(cls,
                        model_dir: str,
                        device: str = "cpu",
                        key_remap=None,
                        key_prefix: "str | None" = None,
                        eagle_base: bool = False) -> nn.Module:
        """Construct and load a model from *model_dir*.

        Reads ``config.json`` via :class:`~config.ModelConfig`, looks up the
        model class in the registry (falling back to the built-in
        :class:`~models.default.modeling_default.CausalLM`), instantiates it,
        moves it to *device*, and loads safetensors weights.

        Args:
            model_dir:      Local HF checkpoint directory.
            device:         Target device (e.g. ``"cpu"``, ``"cuda:0"``).
            key_remap:      Optional callable ``(key: str) -> Optional[str]``.
                            Passed through to :func:`load_weights` for checkpoint
                            key remapping (e.g. TTS talker ``codec_embedding``
                            → ``embed_tokens``).
            key_prefix:     Explicit checkpoint key prefix to strip (e.g.
                            ``"talker."``).  Passed through to :func:`load_weights`.
            eagle_base:     When True, export as EAGLE3 base model with extra
                            tree-attention inputs and hidden_states output.

        Returns:
            Loaded ``nn.Module`` in eval mode.
        """
        from .models.default.modeling_default import CausalLM

        config = ModelConfig.from_pretrained(model_dir)
        if eagle_base:
            config.eagle_base = True

        # EAGLE3 draft: auto-detect from draft_vocab_size
        if config.is_eagle3_draft:
            from .models.eagle3.modeling_eagle3_draft import Eagle3DraftModel
            model_class = Eagle3DraftModel
            # Set up key remapping: midlayer -> layers.0, skip t2d
            if key_remap is None:
                key_remap = _eagle3_key_remap
        else:
            model_class = _MODEL_REGISTRY.get(config.model_type, CausalLM)

        model = model_class(config)
        model.to(device)
        load_weights(model,
                     model_dir,
                     device=device,
                     key_remap=key_remap,
                     key_prefix=key_prefix)

        return model


def param_count(model: nn.Module) -> int:
    """Return total parameter element count (trainable and frozen)."""
    return sum(p.numel() for p in model.parameters())


def dtype_summary(model: nn.Module) -> Dict[str, int]:
    """Map dtype name -> number of parameter elements."""
    out: Dict[str, int] = {}
    for p in model.parameters():
        name = str(p.dtype).replace("torch.", "")
        out[name] = out.get(name, 0) + p.numel()
    return dict(sorted(out.items(), key=lambda x: -x[1]))


# ---------------------------------------------------------------------------
# EAGLE3 helpers
# ---------------------------------------------------------------------------


def _eagle3_key_remap(key: str) -> "str | None":
    """Remap EAGLE3 draft checkpoint keys.

    Handles all known EAGLE3 draft checkpoint variations:
    - ``t2d`` keys are skipped (but ``d2t`` is kept).
    - ``target_model.*`` keys are skipped (multi-target training artifact).
    - ``midlayer.*`` -> ``layers.0.*``
    - ``qkv_proj.{q,k,v}_proj`` -> ``{q,k,v}_proj`` (flatten old pipeline
      ``EdgeLLMAttention`` wrapper nesting, used by quantized checkpoints).
    - ``._pre_quant_scale`` -> ``.pre_quant_scale`` (modelopt internal naming;
      normally stripped by ``postprocess_state_dict()`` but not by per-module
      export via ``_export_quantized_weight()``).
    """
    if "t2d" in key and "d2t" not in key:
        return None  # skip t2d
    if key.startswith("target_model."):
        return None  # skip multi-target training artifact
    key = key.replace("midlayer.", "layers.0.")
    key = key.replace("qkv_proj.q_proj", "q_proj")
    key = key.replace("qkv_proj.k_proj", "k_proj")
    key = key.replace("qkv_proj.v_proj", "v_proj")
    key = key.replace("._pre_quant_scale", ".pre_quant_scale")
    return key
