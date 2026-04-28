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
Processes the chat template to create a JSON file with chat template data for 
the following: 

Roles: 
- System
- User
- Assistant

Messages:
- Role
- Content
  - Type
    - text
    - image
    - video

The JSON file is saved to the exported ONNX model directory.

This implementation uses the HF tokenizer's apply_chat_template method with test cases
to extract the actual prefix/suffix patterns used by the model, rather than trying
to parse the Jinja template directly.
"""

import json
import os
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from ..llm_models.model_utils import (_is_alpamayo_1_model,
                                      _is_qwen3_asr_model,
                                      _is_qwen3_omni_model, is_vlm)


@dataclass
class Message:
    role: str
    content: str | List[Dict[str, str]] = field(default_factory=list)


@dataclass
class SystemMessage(Message):
    role: str = "system"
    content: str = '__SENTINEL_SYS_a7f3e2b1__'


@dataclass
class UserMessage(Message):
    role: str = "user"
    content: str = '__SENTINEL_USR_c9d4f6e8__'


@dataclass
class MultimodalUserMessage(Message):
    role: str = "user"
    content: List[Dict[str, str]] = field(
        default_factory=lambda: [{
            'type': 'text',
            'text': '<placeholder_user_text>'
        }])

    def add_text_content(self, text: str):
        self.content.append({"type": "text", "text": text})

    def add_image_content(self, image: str):
        self.content.append({"type": "image", "image": image})

    def add_video_content(self, video: str):
        self.content.append({"type": "video", "video": video})

    def add_audio_content(self, audio: str):
        self.content.append({"type": "audio", "audio": audio})


@dataclass
class AssistantMessage(Message):
    role: str = "assistant"
    content: str = '<placeholder_assistant_text>'
    # TODO: Add tool calling


# TODO: Add ToolMessage


def _format_messages(tokenizer: Any,
                     messages: List[Message],
                     add_generation_prompt: bool = False,
                     enable_thinking: Optional[bool] = None) -> str:
    """
    Format the messages using the tokenizer's chat template.
    
    Args:
        tokenizer: HuggingFace loaded tokenizer
        messages: List of messages
        add_generation_prompt: Whether to add generation prompt
        enable_thinking: Optional parameter for models that support thinking mode
                        None = use model default behavior
                        False = disable thinking
                        True = explicitly enable thinking
        
    Returns:
        Formatted text
        
    Raises:
        ValueError: If unable to format messages
    """
    # Convert dataclass messages to dictionaries
    message_dicts = [asdict(msg) for msg in messages]

    # Build kwargs for apply_chat_template
    kwargs = {
        'tokenize': False,
        'add_generation_prompt': add_generation_prompt
    }

    # Only add enable_thinking if explicitly set (Qwen3-specific)
    if enable_thinking is not None:
        kwargs['enable_thinking'] = enable_thinking

    try:
        return tokenizer.apply_chat_template(message_dicts, **kwargs)
    except Exception:
        # Fallback: convert list content to string for tokenizers that don't support multimodal
        message_dicts = []
        for msg in messages:
            content = msg.content
            # If content is a list, extract the first text element
            if isinstance(content, list):
                for item in content:
                    if isinstance(item, dict) and item.get('type') == 'text':
                        content = item.get('text', '')
                        break
            message_dicts.append({"role": msg.role, "content": content})

        try:
            return tokenizer.apply_chat_template(message_dicts, **kwargs)
        except Exception as e:
            raise ValueError(
                f"Unable to format messages using HuggingFace tokenizer's apply_chat_template method. "
                f"Messages need to be in the format: role: <str>, content: <str|list of dicts>. "
                f"Check INPUT_FORMAT.md for more details. Error: {e}") from e


def _extract_prefix_suffix(text: str, placeholder: str) -> Tuple[str, str]:
    """
    Extract prefix and suffix from the differential text by finding the placeholder content.
    
    Args:
        text: The text to extract the prefix and suffix from
        placeholder   : The placeholder content to search for in the formatted output
    
    Returns:
        Tuple of (prefix, suffix) strings
    """
    content_start = text.find(placeholder)

    if content_start == -1:
        return "", ""

    prefix = text[:content_start]
    suffix = text[content_start + len(placeholder):]

    return prefix, suffix


def _extract_content_pattern(tokenizer: Any, system_prompt: SystemMessage,
                             content_type: str, placeholder: str,
                             text_only_formatted: str,
                             placeholder_text: str) -> Optional[str]:
    """
    Extract the pattern for a specific content type (image/video/audio) by comparing
    with text-only message.
    
    Args:
        tokenizer: The loaded tokenizer
        system_prompt: System message to use
        content_type: Type of content ('image', 'video', or 'audio')
        placeholder: Placeholder string for the content
        text_only_formatted: Formatted text-only message
        placeholder_text: The text placeholder used
        
    Returns:
        Extracted pattern string or None if failed or tokenizer does not support multimodal content
    """
    # Create user message with the content type
    user_with_content = MultimodalUserMessage()
    if content_type == 'image':
        user_with_content.add_image_content(placeholder)
    elif content_type == 'video':
        user_with_content.add_video_content(placeholder)
    elif content_type == 'audio':
        user_with_content.add_audio_content(placeholder)
    else:
        return None

    with_content_formatted = _format_messages(
        tokenizer, [system_prompt, user_with_content])

    # Extract the differential - what was added for this content type
    if placeholder_text in text_only_formatted and placeholder_text in with_content_formatted:
        # Find position after the placeholder in both
        text_pos = text_only_formatted.find(placeholder_text) + len(
            placeholder_text)
        content_pos = with_content_formatted.find(placeholder_text) + len(
            placeholder_text)

        # Get what comes after the placeholder in both
        text_only_suffix = text_only_formatted[text_pos:]
        with_content_suffix = with_content_formatted[content_pos:]

        # The pattern is what was added (the difference)
        if text_only_suffix and with_content_suffix.endswith(text_only_suffix):
            pattern = with_content_suffix[:-len(text_only_suffix)]
        else:
            pattern = with_content_suffix

        # Strip dynamic prefixes like "Image 1:" or "Video 1:"
        pattern = re.sub(rf'^{content_type.capitalize()} \d+:\s*', '', pattern)

        return pattern if pattern else None

    return None


def validate_chat_template(chat_template_path: str) -> Dict[str, Any]:
    """
    Validate that a chat template JSON file follows the required format.
    
    Args:
        chat_template_path: Path to the chat template JSON file
        
    Raises:
        ValueError: If the chat template is invalid
        FileNotFoundError: If the chat template file doesn't exist
    """
    print(f"Validating chat template: {chat_template_path}")

    if not os.path.exists(chat_template_path):
        raise FileNotFoundError(
            f"Chat template file not found: {chat_template_path}")

    try:
        with open(chat_template_path, 'r') as f:
            template = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in chat template file: {e}")

    def check_type(value, expected_type, field_name):
        if not isinstance(value, expected_type):
            raise ValueError(
                f"'{field_name}' must be {expected_type.__name__}, got {type(value).__name__}"
            )

    def check_required_keys(data, keys, parent=""):
        prefix = f"{parent}." if parent else ""
        missing = [k for k in keys if k not in data]
        if missing:
            raise ValueError(
                f"Missing required keys: {[prefix + k for k in missing]}")

    # Validate roles
    required_roles = ["system", "user", "assistant"]
    check_required_keys(template, ["roles"])
    check_type(template["roles"], dict, "roles")
    check_required_keys(template["roles"], required_roles, "roles")

    for role in required_roles:
        role_data = template["roles"][role]
        check_type(role_data, dict, f"roles.{role}")
        check_required_keys(role_data, ["prefix", "suffix"], f"roles.{role}")
        check_type(role_data["prefix"], str, f"roles.{role}.prefix")
        check_type(role_data["suffix"], str, f"roles.{role}.suffix")

    # Validate optional string fields
    for field in ["generation_prompt", "default_system_prompt", "model_path"]:
        if field in template:
            check_type(template[field], str, field)

    # Validate content_types if present
    if "content_types" in template:
        check_type(template["content_types"], dict, "content_types")
        for content_type, content_data in template["content_types"].items():
            check_type(content_data, dict, f"content_types.{content_type}")
            check_required_keys(content_data, ["format"],
                                f"content_types.{content_type}")
            check_type(content_data["format"], str,
                       f"content_types.{content_type}.format")

    print("Chat template validation successful!")


def process_chat_template(model_dir: str, model_tokenizer: Any,
                          model_processor: Any, output_dir: str) -> None:
    """
    Process the chat template from model's tokenizer and create a JSON file
    with parsed template information.
    
    This function uses the tokenizer's apply_chat_template method with various
    test cases to extract the actual prefix/suffix patterns. 

    Args:
        model_dir: The directory containing the model
        model_tokenizer: The tokenizer to use to process the chat template
        model_processor: The processor to use to process the chat template
        output_dir: Path to save the chat_template.json file
    
    Returns:
        None
    """
    print(f"Processing chat template for {model_dir}")

    tokenizer = None
    loaders = [model_processor, model_tokenizer
               ] if is_vlm(model_dir) else [model_tokenizer, model_processor]
    for ldr in loaders:
        if getattr(ldr, 'chat_template', None):
            print(f"Successfully loaded chat template")
            tokenizer = ldr
            break
        else:
            print(f"No chat template found")
            tokenizer = None

    if tokenizer is None:
        print("Skipping chat template processing - no chat template available")
        return

    print("Extracting patterns from chat template...")

    # Extract system and user role patterns (base case)
    system_prompt = SystemMessage()
    user_prompt = UserMessage()
    try:
        system_formatted = _format_messages(tokenizer, [system_prompt])
        system_prefix, system_suffix = _extract_prefix_suffix(
            system_formatted, system_prompt.content)

        user_formatted = _format_messages(tokenizer,
                                          [system_prompt, user_prompt])
        user_prefix, user_suffix = _extract_prefix_suffix(
            user_formatted[len(system_formatted):], user_prompt.content)

    except ValueError as e:
        # Some chat templates (e.g. qwen3.5) reject system-only messages and require
        # at least one user query. In that case we infer boundaries from
        # [system, user] and [user] probes.
        user_only_formatted = _format_messages(tokenizer, [user_prompt])
        user_prefix_base, user_suffix_base = _extract_prefix_suffix(
            user_only_formatted, user_prompt.content)

        user_formatted = _format_messages(tokenizer,
                                          [system_prompt, user_prompt])
        system_content_start = user_formatted.find(system_prompt.content)
        user_content_start = user_formatted.find(
            user_prompt.content,
            system_content_start + len(system_prompt.content))
        if system_content_start == -1 or user_content_start == -1:
            raise ValueError(
                "Unable to infer system and user boundaries from tokenizer chat template output."
            ) from e

        system_prefix = user_formatted[:system_content_start]
        between = user_formatted[system_content_start +
                                 len(system_prompt.content):user_content_start]
        if user_prefix_base and between.endswith(user_prefix_base):
            system_suffix = between[:-len(user_prefix_base)]
            user_prefix = user_prefix_base
        elif user_prefix_base and user_prefix_base in between:
            split_pos = between.rfind(user_prefix_base)
            system_suffix = between[:split_pos]
            user_prefix = between[split_pos:]
        else:
            # Conservative fallback: preserve all boundary text as user prefix.
            system_suffix = ""
            user_prefix = between
        user_suffix = user_formatted[user_content_start +
                                     len(user_prompt.content):]
        if not user_suffix and user_suffix_base:
            user_suffix = user_suffix_base

    # Some models (e.g. Qwen3-ASR) inject extra role blocks into the
    # system-only output (an empty user turn).  This causes system_suffix
    # to contain markers that belong to the user role.  Strip them so
    # the C++ runtime doesn't emit a spurious empty user block.
    if user_prefix and user_prefix in system_suffix:
        system_suffix = system_suffix[:system_suffix.find(user_prefix)]
    elif not user_prefix and system_suffix:
        # User extraction failed (e.g. template ignores text-only user
        # messages).  If system_suffix has an embedded user block it will
        # look like  SUFFIX + user_prefix + SUFFIX  where SUFFIX is the
        # end-of-turn marker that appears at both ends.  Decompose it.
        for length in range(1, len(system_suffix) // 2 + 1):
            candidate = system_suffix[:length]
            if (system_suffix.endswith(candidate)
                    and len(system_suffix) > 2 * len(candidate)):
                user_prefix = system_suffix[length:-length]
                user_suffix = candidate
                system_suffix = candidate
                print(
                    f"Extracted user role patterns from system suffix: "
                    f"prefix={repr(user_prefix)}, suffix={repr(user_suffix)}")
                break

    # Extract assistant role patterns (compare with user case)
    assistant_prompt = AssistantMessage()
    assistant_formatted = _format_messages(
        tokenizer, [system_prompt, user_prompt, assistant_prompt])
    assistant_prefix, assistant_suffix = _extract_prefix_suffix(
        assistant_formatted[len(user_formatted):], assistant_prompt.content)

    # Extract the default generation prompt (model's natural behavior).
    # Do NOT pass enable_thinking — the default prompt must match what
    # the model produces with no flags, which is what the C++ runtime
    # uses for non-thinking inference.  Passing enable_thinking=False
    # on Qwen3 models injects a <think></think> block that breaks the
    # Talker prefill token layout.
    generation_formatted = _format_messages(tokenizer,
                                            [system_prompt, user_prompt],
                                            add_generation_prompt=True)
    generation_prompt = generation_formatted[len(user_formatted):]

    # Extract generation prompt with thinking enabled (if supported by model)
    generation_prompt_thinking = None
    try:
        thinking_formatted = _format_messages(tokenizer,
                                              [system_prompt, user_prompt],
                                              add_generation_prompt=True,
                                              enable_thinking=False)
        generation_prompt_thinking = thinking_formatted[len(user_formatted):]

        # Only keep if different (model supports thinking mode)
        if generation_prompt_thinking != generation_prompt:
            print(
                "Detected thinking mode support, extracted both generation prompts"
            )
        else:
            generation_prompt_thinking = None
    except Exception:
        # Model doesn't support thinking mode
        pass

    # Build content types
    content_types = {}

    # Only extract multimodal patterns if this is a VLM model
    if is_vlm(model_dir):
        print("Detected VLM model, extracting multimodal content patterns...")
        # Get base text-only formatted message for comparison
        user_text_only = MultimodalUserMessage()
        text_only_formatted = _format_messages(tokenizer,
                                               [system_prompt, user_text_only])
        placeholder_text = user_text_only.content[0]['text']

        # Extract image pattern
        image_pattern = _extract_content_pattern(tokenizer, system_prompt,
                                                 'image',
                                                 '<placeholder_image_path>',
                                                 text_only_formatted,
                                                 placeholder_text)
        if image_pattern:
            content_types['image'] = {'format': image_pattern}

        # Extract video pattern
        video_pattern = _extract_content_pattern(tokenizer, system_prompt,
                                                 'video',
                                                 '<placeholder_video_path>',
                                                 text_only_formatted,
                                                 placeholder_text)
        if video_pattern:
            content_types['video'] = {'format': video_pattern}

    # Check for Omni models (audio + vision + text)
    elif _is_qwen3_omni_model(model_dir) or _is_qwen3_asr_model(model_dir):
        print(
            "Detected Omni-modal model (audio + vision), using special token placeholders..."
        )
        # For Omni models, use special tokens as placeholders that the C++ multimodal runners expect
        # These are single tokens that Qwen3OmniAudioRunner and QwenViTRunner will find and replace
        content_types['audio'] = {'format': '<|audio_pad|>'}
        content_types['image'] = {'format': '<|image_pad|>'}
        content_types['video'] = {'format': '<|video_pad|>'}
        print(
            "  Using special token placeholders: <|audio_pad|>, <|image_pad|>, <|video_pad|>"
        )
        print(
            "  Note: These will be expanded by Qwen3OmniAudioRunner/VisionRunner during inference"
        )

    else:
        print(
            "Text-only LLM detected, skipping multimodal content pattern extraction"
        )

    if _is_alpamayo_1_model(model_dir):
        print(
            "Detected Alpamayo 1 model, adding <|cot_start|> to generation prompt"
        )
        generation_prompt = generation_prompt + "<|cot_start|>"

    # Extract default system prompt by testing without system message
    user_only_prompt = UserMessage()
    user_only_formatted = _format_messages(tokenizer, [user_only_prompt])

    # Extract default system prompt
    default_system_prompt = ""
    # Check if a default system prompt was added
    # The system message should appear in user_only_formatted if there's a default
    system_start = user_only_formatted.find(system_prefix)
    if system_start != -1:
        # Extract the system content between prefix and suffix
        content_start = system_start + len(system_prefix)
        content_end = user_only_formatted.find(system_suffix, content_start)
        if content_end != -1:
            default_system_prompt = user_only_formatted[
                content_start:content_end]
            # Remove the placeholder if it appears
            if default_system_prompt == system_prompt.content:
                default_system_prompt = ""

    # Build the final JSON structure
    chat_template_data = {
        "model_path": model_dir,
        "roles": {
            "system": {
                "prefix": system_prefix,
                "suffix": system_suffix
            },
            "user": {
                "prefix": user_prefix,
                "suffix": user_suffix
            },
            "assistant": {
                "prefix": assistant_prefix,
                "suffix": assistant_suffix
            }
        },
        "content_types": content_types,
        "generation_prompt": generation_prompt,
        "default_system_prompt": default_system_prompt
    }

    # Add thinking mode generation prompt if model supports it.
    # Sanity check: generation_prompt (non-thinking default) must not contain
    # <think>; if it does the two prompts were extracted in the wrong order
    # (happens with Qwen3 Jinja2 templates where enable_thinking=False
    # *adds* the <think> block).  Swap them to fix.
    if generation_prompt_thinking is not None:
        if '<think>' in generation_prompt and '<think>' not in generation_prompt_thinking:
            generation_prompt, generation_prompt_thinking = (
                generation_prompt_thinking, generation_prompt)
            chat_template_data["generation_prompt"] = generation_prompt
        chat_template_data[
            "generation_prompt_thinking"] = generation_prompt_thinking

    # Save to output directory
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "processed_chat_template.json")

    with open(output_path, 'w') as f:
        json.dump(chat_template_data, f, indent=2)

    print(f"Chat template saved to {output_path}")
