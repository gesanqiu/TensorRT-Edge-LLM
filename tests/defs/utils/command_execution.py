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
Test execution functions for TensorRT Edge-LLM tests.

This module contains the simplified test execution functions for build, inference, 
and benchmark tests. Each function is focused on its specific task without 
unnecessary abstraction layers.
"""

import os
from typing import Any, Dict, Optional

import pytest
from conftest import EnvironmentConfig, RemoteConfig
from pytest_helpers import check_file_exists, run_command, run_with_trt_env

from ..config import TaskType, TestConfig
from .accuracy import check_accuracy_with_dataset
from .baseline import (get_baseline, map_accuracy_result_to_csv,
                       parse_perf_from_output, save_to_baseline)
from .command_generation import (generate_build_commands,
                                 generate_e2e_bench_commands,
                                 generate_inference_commands,
                                 generate_kernel_bench_commands)


def check_result_failures(result: Dict[str, Any]) -> None:
    """Check baseline regressions first, then static threshold failures.

    Called by pipeline tests after execute_*_test returns successfully.
    """
    failures = []
    if result.get('baseline_regressions'):
        failures.append("Baseline regression:\n  " +
                        "\n  ".join(result['baseline_regressions']))
    if result.get('threshold_failure'):
        failures.append(result['threshold_failure'])
    if failures:
        pytest.fail("\n\n".join(failures))


def _try_save_baseline(config: TestConfig, test_func: str,
                       result: Dict[str, Any], logger) -> None:
    """Save current result to baseline CSV when BASELINE_CSV is set but has no entry."""
    csv_path = os.environ.get('BASELINE_CSV', 'logs/baseline.csv')
    if not result.get('success', False):
        return
    if result.get('threshold_failure'):
        return
    save_to_baseline(csv_path, config.model_type.value, test_func,
                     config.param_str, result)
    if logger:
        logger.info(
            "No baseline entry for [%s]. "
            "Saved current result to %s", config.param_str, csv_path)


def _check_baseline_regression(config: TestConfig,
                               test_func: str,
                               result: Dict[str, Any],
                               logger,
                               check_perf: bool = False) -> bool:
    """Check accuracy (and optionally perf) regression against baseline CSV.

    Returns True if baseline entry was found (regardless of pass/fail).
    When baseline is found, threshold_failure is cleared since baseline takes priority.
    If no baseline exists, saves the current result for future runs.

    Args:
        check_perf: only True for benchmark tests; inference skips perf comparison.
    """
    baseline = get_baseline()
    if baseline is None:
        _try_save_baseline(config, test_func, result, logger)
        return False

    entry = baseline.find_by_param(config.param_str,
                                   test_func,
                                   model_type_value=config.model_type.value)
    if entry is None:
        _try_save_baseline(config, test_func, result, logger)
        return False

    regressions = []
    all_summaries = []

    current_acc = map_accuracy_result_to_csv(result)
    if current_acc:
        acc_reg, acc_sum = baseline.check_accuracy_regression(
            entry, current_acc)
        regressions.extend(acc_reg)
        all_summaries.extend(acc_sum)

    if check_perf:
        raw_output = result.get('output', '')
        current_perf = parse_perf_from_output(raw_output)
        # Merge accuracy metrics into perf dict; check_perf_regression
        # only looks at columns in PERF_LOWER/HIGHER_IS_BETTER, so extras
        # (e.g. rouge scores) are naturally ignored.
        current_perf.update(current_acc)
        if current_perf:
            perf_reg, perf_sum = baseline.check_perf_regression(
                entry, current_perf)
            regressions.extend(perf_reg)
            all_summaries.extend(perf_sum)

    if logger and all_summaries:
        logger.info("Baseline comparison:\n  " + "\n  ".join(all_summaries))

    if regressions:
        result['baseline_regressions'] = regressions
        if logger:
            logger.warning("Baseline regressions detected:\n  " +
                           "\n  ".join(regressions))

    # Baseline found → it takes priority, discard static threshold result
    result.pop('threshold_failure', None)
    return True


def execute_build_test(
        config: TestConfig, executable_files: Dict[str, str],
        remote_config: Optional[RemoteConfig], logger,
        env_config: Optional[EnvironmentConfig]) -> Dict[str, Any]:
    """Execute build test for any model type"""

    # Generate all build commands
    commands = generate_build_commands(config, executable_files)

    all_outputs = []

    engine_file_map = {
        executable_files['llm_build']: "llm.engine",
        executable_files['visual_build']: "visual.engine",
    }

    for i, (cmd, timeout) in enumerate(commands):
        task_name = f"Build step {i+1}/{len(commands)}"
        if logger:
            logger.info(f"Starting {task_name}: {' '.join(cmd)}")

        engine_filename = engine_file_map.get(cmd[0])
        if engine_filename:
            engine_dir = next(
                (arg.split('=', 1)[1]
                 for arg in cmd if arg.startswith('--engineDir=')), None)
            if engine_dir and check_file_exists(
                    os.path.join(engine_dir, engine_filename), remote_config,
                    logger):
                if logger:
                    logger.info(
                        f"{engine_filename} already exists in {engine_dir}. Skipping."
                    )
                all_outputs.append(
                    f"{engine_filename} already exists - skipped")
                continue

        result = run_with_trt_env(cmd, remote_config, timeout, logger,
                                  env_config)
        all_outputs.append(result['output'])

        if not result['success']:
            return {
                'success': False,
                'error':
                f"{task_name} failed: {result.get('error', 'Unknown error')}",
                'output': '\n'.join(all_outputs),
                'test_type': TaskType.BUILD.value
            }

    return {
        'success': True,
        'error': None,
        'output': '\n'.join(all_outputs),
        'test_type': TaskType.BUILD.value
    }


def execute_e2e_bench_test(
        config: TestConfig, executable_files: Dict[str, str],
        remote_config: Optional[RemoteConfig], logger,
        env_config: Optional[EnvironmentConfig]) -> Dict[str, Any]:
    """Execute end-to-end benchmark test for any model type"""

    # Handle LoRA weights replacement if needed
    if config.max_lora_rank > 0:
        # Edit the test case file to replace $LORA_WEIGHTS_DIR with the lora weights directory
        test_case_file = config.get_test_case_file()
        result = run_command([
            'sed', '-i',
            f's|$LORA_WEIGHTS_DIR|{config.get_lora_weights_dir()}|g',
            test_case_file
        ], remote_config, 300, logger)
        if not result['success']:
            result['test_type'] = TaskType.E2E_BENCH.value
            return result

    # Generate all e2e benchmark commands
    commands = generate_e2e_bench_commands(config, executable_files)

    all_outputs = []

    for i, (cmd, timeout) in enumerate(commands):
        task_name = f"Benchmark step {i+1}/{len(commands)}"
        if logger:
            logger.info(f"Starting {task_name}: {' '.join(cmd)}")

        result = run_with_trt_env(cmd, remote_config, timeout, logger,
                                  env_config)
        all_outputs.append(result['output'])

        if not result['success']:
            return {
                'success': False,
                'error':
                f"{task_name} failed: {result.get('error', 'Unknown error')}",
                'output': '\n'.join(all_outputs),
                'test_type': TaskType.E2E_BENCH.value
            }

    # Calculate metrics based on dataset type
    final_result = {
        'success': True,
        'error': None,
        'output': '\n'.join(all_outputs),
        'test_type': TaskType.E2E_BENCH.value
    }

    try:
        # Use model-specific reference if available, fallback to generic test case file
        reference_file = config.get_reference_json_file(
        ) or config.get_test_case_file()
        # Pass file paths directly to the accuracy checker (runs on host only)
        metrics_result = check_accuracy_with_dataset(
            config.get_output_json_file(), reference_file, config.test_case,
            logger)

        # Merge metrics result into final result
        final_result.update(metrics_result)

    except Exception as e:
        final_result['error'] = f"Failed to calculate metrics: {str(e)}"
        final_result['success'] = False

    if final_result['success']:
        _check_baseline_regression(config,
                                   'test_e2e_bench',
                                   final_result,
                                   logger,
                                   check_perf=True)

    return final_result


def execute_inference_test(
        config: TestConfig, executable_files: Dict[str, str],
        remote_config: Optional[RemoteConfig], logger,
        env_config: Optional[EnvironmentConfig]) -> Dict[str, Any]:
    """Execute inference test for any model type"""

    # Handle LoRA weights replacement if needed
    if config.max_lora_rank > 0:
        # Edit the test case file to replace $LORA_WEIGHTS_DIR with the lora weights directory
        test_case_file = config.get_test_case_file()
        result = run_command([
            'sed', '-i',
            f's|$LORA_WEIGHTS_DIR|{config.get_lora_weights_dir()}|g',
            test_case_file
        ], remote_config, 300, logger)
        if not result['success']:
            result['test_type'] = TaskType.INFERENCE.value
            return result

    # Generate all inference commands
    commands = generate_inference_commands(config, executable_files)

    all_outputs = []

    for i, (cmd, timeout) in enumerate(commands):
        task_name = f"Inference step {i+1}/{len(commands)}"
        if logger:
            logger.info(f"Starting {task_name}: {' '.join(cmd)}")

        result = run_with_trt_env(cmd, remote_config, timeout, logger,
                                  env_config)
        all_outputs.append(result['output'])

        if not result['success']:
            return {
                'success': False,
                'error':
                f"{task_name} failed: {result.get('error', 'Unknown error')}",
                'output': '\n'.join(all_outputs),
                'test_type': TaskType.INFERENCE.value
            }

    # Calculate metrics based on dataset type
    final_result = {
        'success': True,
        'error': None,
        'output': '\n'.join(all_outputs),
        'test_type': TaskType.INFERENCE.value
    }

    try:
        # Use model-specific reference if available, fallback to generic test case file
        reference_file = config.get_reference_json_file(
        ) or config.get_test_case_file()
        # Pass file paths directly to the accuracy checker (runs on host only)
        metrics_result = check_accuracy_with_dataset(
            config.get_output_json_file(), reference_file, config.test_case,
            logger)

        # Merge metrics result into final result
        final_result.update(metrics_result)

    except Exception as e:
        final_result['error'] = f"Failed to calculate metrics: {str(e)}"
        final_result['success'] = False

    if final_result['success']:
        _check_baseline_regression(config, 'test_inference', final_result,
                                   logger)

    return final_result


def execute_kernel_bench_test(
        config: TestConfig, executable_files: Dict[str, str],
        remote_config: Optional[RemoteConfig], logger,
        env_config: Optional[EnvironmentConfig]) -> Dict[str, Any]:
    """Execute kernel_bench test - validates that the kernel benchmark runs successfully"""

    commands = generate_kernel_bench_commands(config, executable_files)

    all_outputs = []

    for i, (cmd, timeout) in enumerate(commands):
        task_name = f"kernel_bench step {i+1}/{len(commands)}"
        if logger:
            logger.info(f"Starting {task_name}: {' '.join(cmd)}")

        result = run_with_trt_env(cmd, remote_config, timeout, logger,
                                  env_config)
        all_outputs.append(result['output'])

        if not result['success']:
            return {
                'success': False,
                'error':
                f"{task_name} failed: {result.get('error', 'Unknown error')}",
                'output': '\n'.join(all_outputs),
                'test_type': TaskType.KERNEL_BENCH.value
            }

    return {
        'success': True,
        'error': None,
        'output': '\n'.join(all_outputs),
        'test_type': TaskType.KERNEL_BENCH.value
    }
