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
Test suite for LLM pipeline functionality.

Tests engine building, chat inference, benchmarking, and batch inference
using parameterized configurations.
"""
import logging
from typing import Dict, Optional

import pytest
from conftest import EnvironmentConfig, RemoteConfig
from pytest_helpers import timer_context

from .config import ModelType, TaskType, TestConfig
from .utils.command_execution import (check_result_failures,
                                      execute_build_test,
                                      execute_e2e_bench_test,
                                      execute_inference_test,
                                      execute_kernel_bench_test)


class TestLLMPipeline:
    """Test suite for LLM pipeline functionality."""

    def test_engine_build(self, test_param: str, executable_files: Dict[str,
                                                                        str],
                          remote_config: Optional[RemoteConfig],
                          test_logger: logging.Logger,
                          env_config: EnvironmentConfig) -> None:
        """Test TensorRT engine building for LLM models."""
        config = TestConfig.from_param_string(test_param, ModelType.LLM,
                                              TaskType.BUILD, env_config)

        with timer_context(f"LLM build for {config.model_name}", test_logger):
            result = execute_build_test(config, executable_files,
                                        remote_config, test_logger, env_config)
            if not result['success']:
                pytest.fail(f"Build failed: {result['error']}")

    def test_e2e_bench(self, test_param: str, executable_files: Dict[str, str],
                       remote_config: Optional[RemoteConfig],
                       test_logger: logging.Logger,
                       env_config: EnvironmentConfig) -> None:
        """Test end-to-end benchmarking for LLM models."""
        config = TestConfig.from_param_string(test_param, ModelType.LLM,
                                              TaskType.E2E_BENCH, env_config)

        with timer_context(f"LLM e2e_bench for {config.model_name}",
                           test_logger):
            result = execute_e2e_bench_test(config, executable_files,
                                            remote_config, test_logger,
                                            env_config)
            if not result['success']:
                pytest.fail(f"e2e_bench failed: {result['error']}")
            check_result_failures(result)

    def test_kernel_bench(self, test_param: str, executable_files: Dict[str,
                                                                        str],
                          remote_config: Optional[RemoteConfig],
                          test_logger: logging.Logger,
                          env_config: EnvironmentConfig) -> None:
        """Test kernel-level benchmarking with llm_bench."""
        config = TestConfig.from_param_string(test_param, ModelType.LLM,
                                              TaskType.KERNEL_BENCH,
                                              env_config)

        with timer_context(f"kernel_bench for {config.model_name}",
                           test_logger):
            result = execute_kernel_bench_test(config, executable_files,
                                               remote_config, test_logger,
                                               env_config)
            if not result['success']:
                pytest.fail(f"kernel_bench failed: {result['error']}")

    def test_inference(self, test_param: str, executable_files: Dict[str, str],
                       remote_config: Optional[RemoteConfig],
                       test_logger: logging.Logger,
                       env_config: EnvironmentConfig) -> None:
        """Test batch inference for LLM models."""
        config = TestConfig.from_param_string(test_param, ModelType.LLM,
                                              TaskType.INFERENCE, env_config)

        with timer_context(f"LLM inference for {config.model_name}",
                           test_logger):
            result = execute_inference_test(config, executable_files,
                                            remote_config, test_logger,
                                            env_config)
            if not result['success']:
                pytest.fail(f"Inference failed: {result['error']}")
            check_result_failures(result)
