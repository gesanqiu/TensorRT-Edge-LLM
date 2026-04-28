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
End-to-end tests for the experimental Python server (pybind11 runtime).

Tests the full pipeline: build pybind extension -> load TRT engine via
Python API -> run inference -> validate output.
"""
import logging
import os
import shlex
from typing import Dict, Optional

import pytest
from conftest import EnvironmentConfig, RemoteConfig
from pytest_helpers import run_command, timer_context

from .config import ModelType, TaskType, TestConfig


def test_build_pybind(env_config: EnvironmentConfig,
                      remote_config: Optional[RemoteConfig],
                      test_logger: logging.Logger):
    """Build the pybind extension by reconfiguring the existing build directory.

    Uses subdirectory mode (BUILD_PYTHON_BINDINGS=ON) so all cmake
    variables (CUDA, TRT, toolchain) are inherited from the prior
    test_build_project configuration.
    """
    build_dir = env_config.build_dir

    pybind_venv = 'venv/pybind'
    install_cmd = (f'python3 -m venv {pybind_venv}'
                   f' && {pybind_venv}/bin/pip install -q pybind11')
    result = run_command(cmd=['bash', '-c', install_cmd],
                         remote_config=remote_config,
                         timeout=120,
                         logger=test_logger)
    if not result['success']:
        pytest.fail(f"Failed to install pybind11: {result.get('error')}")

    pybind11_dir_expr = (
        f'$({pybind_venv}/bin/python'
        ' -c "import pybind11; print(pybind11.get_cmake_dir())")')
    build_cmd = (f'PYBIND11_DIR={pybind11_dir_expr}'
                 f' && cd {build_dir}'
                 f' && cmake .. -DBUILD_PYTHON_BINDINGS=ON'
                 f' -Dpybind11_DIR=$PYBIND11_DIR'
                 f' && make -j$(nproc) _edgellm_runtime')

    with timer_context("Building pybind extension", test_logger):
        result = run_command(cmd=['bash', '-c', build_cmd],
                             remote_config=remote_config,
                             timeout=600,
                             logger=test_logger)

    if not result['success']:
        pytest.fail(f"Pybind build failed: {result.get('error')}")

    pybind_output_dir = f'{build_dir}/pybind'
    result = run_command(
        cmd=['bash', '-c', f'ls {pybind_output_dir}/*_edgellm_runtime*.so'],
        remote_config=remote_config,
        timeout=10,
        logger=test_logger)
    if not result['success']:
        pytest.fail(
            f"_edgellm_runtime.so not found in {pybind_output_dir} after build"
        )


class TestServerPipeline:
    """E2E tests for inference via the experimental Python server API."""

    def test_server_inference(self, test_param: str,
                              executable_files: Dict[str, str],
                              remote_config: Optional[RemoteConfig],
                              test_logger: logging.Logger,
                              env_config: EnvironmentConfig) -> None:
        """Test inference using the pybind Python runtime directly."""
        is_vlm = "-mnit" in test_param
        model_type = ModelType.VLM if is_vlm else ModelType.LLM
        config = TestConfig.from_param_string(test_param, model_type,
                                              TaskType.INFERENCE, env_config)

        engine_dir = config.get_llm_engine_dir()
        multimodal_engine_dir = config.get_visual_engine_dir(
        ) if is_vlm else ""
        test_logger.info("Using engine dir: %s", engine_dir)
        if multimodal_engine_dir:
            test_logger.info("Using visual engine dir: %s",
                             multimodal_engine_dir)

        pybind_build_dir = os.path.join(env_config.build_dir, "pybind")
        prompt = "Please introduce the company NVIDIA and its CEO."
        max_tokens = 128

        script = f"""\
import sys, os
sys.path.insert(0, {pybind_build_dir!r})
import importlib.util
so_files = [f for f in os.listdir({pybind_build_dir!r}) if '_edgellm_runtime' in f and f.endswith('.so')]
if not so_files:
    raise RuntimeError('_edgellm_runtime.so not found in ' + {pybind_build_dir!r})
spec = importlib.util.spec_from_file_location('_edgellm_runtime', os.path.join({pybind_build_dir!r}, so_files[0]))
rt = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rt)

engine_dir = {engine_dir!r}
multimodal_engine_dir = {multimodal_engine_dir!r}
runtime = rt.LLMRuntime(engine_dir, multimodal_engine_dir, {{}})
runtime.capture_decoding_cuda_graph()

request = rt.LLMGenerationRequest()
msg = rt.Message()
msg.role = 'user'
msg.contents = [rt.MessageContent('text', {prompt!r})]
req = rt.Request(messages=[msg])
req.image_buffers = []
request.requests = [req]
request.temperature = 0.7
request.top_p = 0.9
request.top_k = 50
request.max_generate_length = {max_tokens}
request.apply_chat_template = True
request.add_generation_prompt = True
request.enable_thinking = False
request.disable_spec_decode = False

response = runtime.handle_request(request)
text = response.output_texts[0] if response.output_texts else ''
ids = response.output_ids[0] if response.output_ids else []
print(f'OUTPUT_TEXT_LEN={{len(text)}}')
print(f'OUTPUT_IDS_LEN={{len(ids)}}')
print(f'OUTPUT_TEXT={{text[:200]}}')
assert len(text) > 0, 'Empty output text'
assert len(ids) > 0, 'Empty output token ids'
print('SERVER_INFERENCE_PASSED')
"""
        script_escaped = shlex.quote(script)
        cmd = ['bash', '-c', f'python3 -c {script_escaped}']

        env_vars = None
        if env_config.trt_package_dir:
            trt_lib = f"{env_config.trt_package_dir}/lib"
            env_vars = {"LD_LIBRARY_PATH": f"$LD_LIBRARY_PATH:{trt_lib}"}

        with timer_context(
                f"Server inference for {config.model_name}",
                test_logger,
        ):
            result = run_command(cmd=cmd,
                                 remote_config=remote_config,
                                 timeout=600,
                                 logger=test_logger,
                                 env_vars=env_vars)

        if not result['success']:
            pytest.fail(
                f"Server inference failed: {result.get('error', 'Unknown')}")

        output = result.get('output', '')
        if 'SERVER_INFERENCE_PASSED' not in output:
            pytest.fail(
                f"Server inference did not produce expected output. Output:\n{output}"
            )

    def test_server_streaming(self, test_param: str,
                              executable_files: Dict[str, str],
                              remote_config: Optional[RemoteConfig],
                              test_logger: logging.Logger,
                              env_config: EnvironmentConfig) -> None:
        """Test streaming inference using StreamChannel via pybind."""
        config = TestConfig.from_param_string(test_param, ModelType.LLM,
                                              TaskType.INFERENCE, env_config)

        engine_dir = config.get_llm_engine_dir()
        test_logger.info("Using engine dir: %s", engine_dir)

        pybind_build_dir = os.path.join(env_config.build_dir, "pybind")
        prompt = "Count from 1 to 10."
        max_tokens = 128

        script = f"""\
import sys, os, threading
sys.path.insert(0, {pybind_build_dir!r})
import importlib.util
so_files = [f for f in os.listdir({pybind_build_dir!r}) if '_edgellm_runtime' in f and f.endswith('.so')]
if not so_files:
    raise RuntimeError('_edgellm_runtime.so not found in ' + {pybind_build_dir!r})
spec = importlib.util.spec_from_file_location('_edgellm_runtime', os.path.join({pybind_build_dir!r}, so_files[0]))
rt = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rt)

engine_dir = {engine_dir!r}
runtime = rt.LLMRuntime(engine_dir, '', {{}})
runtime.capture_decoding_cuda_graph()

channel = rt.StreamChannel.create()
channel.set_skip_special_tokens(True)

request = rt.LLMGenerationRequest()
msg = rt.Message()
msg.role = 'user'
msg.contents = [rt.MessageContent('text', {prompt!r})]
req = rt.Request(messages=[msg])
req.image_buffers = []
request.requests = [req]
request.stream_channels = [channel]
request.temperature = 0.7
request.top_p = 0.9
request.top_k = 50
request.max_generate_length = {max_tokens}
request.apply_chat_template = True
request.add_generation_prompt = True
request.enable_thinking = False
request.disable_spec_decode = False

def run_inference():
    runtime.handle_request(request)

worker = threading.Thread(target=run_inference, daemon=True)
worker.start()

chunks = []
while True:
    chunk = channel.wait_pop(timeout_ms=500)
    if chunk is None:
        if channel.is_finished() or channel.is_cancelled():
            break
        continue
    chunks.append(chunk)
    if chunk.finished:
        break

worker.join(timeout=10)

total_text = ''.join(c.text for c in chunks)
total_ids = sum(len(c.token_ids) for c in chunks)
print(f'STREAM_CHUNKS={{len(chunks)}}')
print(f'STREAM_TEXT_LEN={{len(total_text)}}')
print(f'STREAM_IDS={{total_ids}}')
print(f'STREAM_TEXT={{total_text[:200]}}')
assert len(chunks) > 1, f'Expected multiple chunks, got {{len(chunks)}}'
assert len(total_text) > 0, 'Empty streamed text'
assert any(c.finished for c in chunks), 'No terminal chunk received'
print('SERVER_STREAMING_PASSED')
"""
        script_escaped = shlex.quote(script)
        cmd = ['bash', '-c', f'python3 -c {script_escaped}']

        env_vars = None
        if env_config.trt_package_dir:
            trt_lib = f"{env_config.trt_package_dir}/lib"
            env_vars = {"LD_LIBRARY_PATH": f"$LD_LIBRARY_PATH:{trt_lib}"}

        with timer_context(
                f"Server streaming for {config.model_name}",
                test_logger,
        ):
            result = run_command(cmd=cmd,
                                 remote_config=remote_config,
                                 timeout=600,
                                 logger=test_logger,
                                 env_vars=env_vars)

        if not result['success']:
            pytest.fail(
                f"Server streaming failed: {result.get('error', 'Unknown')}")

        output = result.get('output', '')
        if 'SERVER_STREAMING_PASSED' not in output:
            pytest.fail(
                f"Server streaming did not produce expected output. Output:\n{output}"
            )


class TestHLAPI:
    """E2E tests for the high-level LLM Python API with pre-built engines."""

    @staticmethod
    def _build_hlapi_env_setup(trt_package_dir: str = "") -> str:
        """Return inline script preamble that sets up sys.path and LD_LIBRARY_PATH."""
        parts = [
            "import sys, os",
            "sys.path.insert(0, os.getcwd())",
        ]
        if trt_package_dir:
            parts.append("os.environ.setdefault('LD_LIBRARY_PATH', '')")
            parts.append(
                f"os.environ['LD_LIBRARY_PATH'] += ':{trt_package_dir}/lib'")
        return "\n".join(parts)

    def test_hlapi_generate(self, test_param: str, executable_files: Dict[str,
                                                                          str],
                            remote_config: Optional[RemoteConfig],
                            test_logger: logging.Logger,
                            env_config: EnvironmentConfig) -> None:
        """Test LLM.generate() with a pre-built engine directory."""
        is_vlm = "-mnit" in test_param
        model_type = ModelType.VLM if is_vlm else ModelType.LLM
        config = TestConfig.from_param_string(test_param, model_type,
                                              TaskType.INFERENCE, env_config)

        engine_dir = config.get_llm_engine_dir()
        visual_engine_dir = config.get_visual_engine_dir() if is_vlm else ""
        test_logger.info("HLAPI generate: engine=%s visual=%s", engine_dir,
                         visual_engine_dir or "(none)")

        prompt = "Please introduce the company NVIDIA and its CEO."
        max_tokens = 128

        setup = self._build_hlapi_env_setup(env_config.trt_package_dir or "")

        script = f"""\
{setup}
from experimental.server import LLM, SamplingParams

llm = LLM(engine_dir={engine_dir!r}, visual_engine_dir={visual_engine_dir!r})
outputs = llm.generate(
    [{prompt!r}],
    SamplingParams(temperature=0.7, max_tokens={max_tokens}),
)
text = outputs[0].text
ids = outputs[0].token_ids
print(f'HLAPI_TEXT_LEN={{len(text)}}')
print(f'HLAPI_IDS_LEN={{len(ids)}}')
print(f'HLAPI_TEXT={{text[:200]}}')
assert len(text) > 0, 'Empty output text'
assert len(ids) > 0, 'Empty output token ids'
print('HLAPI_GENERATE_PASSED')
"""
        script_escaped = shlex.quote(script)
        cmd = ['bash', '-c', f'python3 -c {script_escaped}']

        env_vars = None
        if env_config.trt_package_dir:
            trt_lib = f"{env_config.trt_package_dir}/lib"
            env_vars = {"LD_LIBRARY_PATH": f"$LD_LIBRARY_PATH:{trt_lib}"}

        with timer_context(
                f"HLAPI generate for {config.model_name}",
                test_logger,
        ):
            result = run_command(cmd=cmd,
                                 remote_config=remote_config,
                                 timeout=600,
                                 logger=test_logger,
                                 env_vars=env_vars)

        if not result['success']:
            pytest.fail(
                f"HLAPI generate failed: {result.get('error', 'Unknown')}")

        output = result.get('output', '')
        if 'HLAPI_GENERATE_PASSED' not in output:
            pytest.fail(
                f"HLAPI generate did not produce expected output. Output:\n{output}"
            )

    def test_hlapi_streaming(self, test_param: str,
                             executable_files: Dict[str, str],
                             remote_config: Optional[RemoteConfig],
                             test_logger: logging.Logger,
                             env_config: EnvironmentConfig) -> None:
        """Test LLM.generate_stream() with a pre-built engine directory."""
        config = TestConfig.from_param_string(test_param, ModelType.LLM,
                                              TaskType.INFERENCE, env_config)

        engine_dir = config.get_llm_engine_dir()
        test_logger.info("HLAPI streaming: engine=%s", engine_dir)

        prompt = "Count from 1 to 10."
        max_tokens = 128

        setup = self._build_hlapi_env_setup(env_config.trt_package_dir or "")

        script = f"""\
{setup}
from experimental.server import LLM, SamplingParams

llm = LLM(engine_dir={engine_dir!r})
chunks = list(llm.generate_stream(
    [{{"role": "user", "content": {prompt!r}}}],
    SamplingParams(temperature=0.7, max_tokens={max_tokens}),
))
total_text = ''.join(c.text for c in chunks)
print(f'HLAPI_STREAM_CHUNKS={{len(chunks)}}')
print(f'HLAPI_STREAM_TEXT_LEN={{len(total_text)}}')
print(f'HLAPI_STREAM_TEXT={{total_text[:200]}}')
assert len(chunks) > 1, f'Expected multiple chunks, got {{len(chunks)}}'
assert len(total_text) > 0, 'Empty streamed text'
assert any(c.finished for c in chunks), 'No terminal chunk received'
print('HLAPI_STREAMING_PASSED')
"""
        script_escaped = shlex.quote(script)
        cmd = ['bash', '-c', f'python3 -c {script_escaped}']

        env_vars = None
        if env_config.trt_package_dir:
            trt_lib = f"{env_config.trt_package_dir}/lib"
            env_vars = {"LD_LIBRARY_PATH": f"$LD_LIBRARY_PATH:{trt_lib}"}

        with timer_context(
                f"HLAPI streaming for {config.model_name}",
                test_logger,
        ):
            result = run_command(cmd=cmd,
                                 remote_config=remote_config,
                                 timeout=600,
                                 logger=test_logger,
                                 env_vars=env_vars)

        if not result['success']:
            pytest.fail(
                f"HLAPI streaming failed: {result.get('error', 'Unknown')}")

        output = result.get('output', '')
        if 'HLAPI_STREAMING_PASSED' not in output:
            pytest.fail(
                f"HLAPI streaming did not produce expected output. Output:\n{output}"
            )
