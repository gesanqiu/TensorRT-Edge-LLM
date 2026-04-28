/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "builder/actionBuilder.h"
#include "common/logger.h"

#include <cstdlib>
#include <fstream>
#include <getopt.h>
#include <iostream>
#include <string>

using namespace trt_edgellm;

enum ActionBuildOptionId : int
{
    HELP = 701,
    ONNX_DIR = 702,
    ENGINE_DIR = 703,
    DEBUG = 704,
    MIN_SEQ_LEN = 705,
    MAX_SEQ_LEN = 706,
    OPT_SEQ_LEN = 707
};

struct ActionBuildArgs
{
    std::string onnxDir;
    std::string engineDir;
    bool help{false};
    bool debug{false};
    int32_t minSeqLen{2808};
    int32_t maxSeqLen{3424};
    int32_t optSeqLen{0}; // Set to midpoint in parseActionBuildArgs if not passed
};

static void printUsage(char const* programName)
{
    std::cerr << "Usage: " << programName
              << " [--help] <--onnxDir str> <--engineDir str> [--debug] [--minSeqLen int] [--maxSeqLen int]"
              << std::endl;
    std::cerr << "Options:" << std::endl;
    std::cerr << "  --help       Display this help message" << std::endl;
    std::cerr << "  --onnxDir    Directory containing the action expert ONNX (model.onnx and config.json). Required."
              << std::endl;
    std::cerr << "  --engineDir  Output directory. Engine will be saved to <engineDir>/action/action.engine"
              << std::endl;
    std::cerr << "  --debug      Enable verbose logging" << std::endl;
    std::cerr << "  --minSeqLen  Minimum number of tokens in the past KV cache input (default: 2808). " << std::endl;
    std::cerr << "  --maxSeqLen  Maximum number of tokens in the past KV cache input (default: 3424). " << std::endl;
    std::cerr << "  --optSeqLen  Optimal number of tokens in the past KV cache input to optimize the engine for "
                 "(default: midpoint of min/max)."
              << std::endl;
}

static bool parseActionBuildArgs(ActionBuildArgs& args, int argc, char* argv[])
{
    static struct option options[] = {{"help", no_argument, 0, ActionBuildOptionId::HELP},
        {"onnxDir", required_argument, 0, ActionBuildOptionId::ONNX_DIR},
        {"engineDir", required_argument, 0, ActionBuildOptionId::ENGINE_DIR},
        {"debug", no_argument, 0, ActionBuildOptionId::DEBUG},
        {"minSeqLen", required_argument, 0, ActionBuildOptionId::MIN_SEQ_LEN},
        {"maxSeqLen", required_argument, 0, ActionBuildOptionId::MAX_SEQ_LEN},
        {"optSeqLen", required_argument, 0, ActionBuildOptionId::OPT_SEQ_LEN}, {0, 0, 0, 0}};

    int opt;
    while ((opt = getopt_long(argc, argv, "", options, nullptr)) != -1)
    {
        switch (opt)
        {
        case ActionBuildOptionId::HELP: args.help = true; return true;
        case ActionBuildOptionId::ONNX_DIR:
            if (optarg)
                args.onnxDir = optarg;
            else
            {
                LOG_ERROR("--onnxDir requires an argument.");
                return false;
            }
            break;
        case ActionBuildOptionId::ENGINE_DIR:
            if (optarg)
                args.engineDir = optarg;
            else
            {
                LOG_ERROR("--engineDir requires an argument.");
                return false;
            }
            break;
        case ActionBuildOptionId::DEBUG: args.debug = true; break;
        case ActionBuildOptionId::MIN_SEQ_LEN:
            if (optarg)
                args.minSeqLen = std::stoi(optarg);
            break;
        case ActionBuildOptionId::MAX_SEQ_LEN:
            if (optarg)
                args.maxSeqLen = std::stoi(optarg);
            break;
        case ActionBuildOptionId::OPT_SEQ_LEN:
            if (optarg)
                args.optSeqLen = std::stoi(optarg);
            break;
        default: LOG_ERROR("Invalid argument"); return false;
        }
    }
    // Default opt to midpoint when not passed
    if (args.optSeqLen == 0)
    {
        args.optSeqLen = (args.minSeqLen + args.maxSeqLen) / 2;
    }
    return true;
}

int main(int argc, char** argv)
{
    ActionBuildArgs args;
    if ((argc < 2) || (!parseActionBuildArgs(args, argc, argv)))
    {
        LOG_ERROR("Unable to parse builder args.");
        printUsage(argv[0]);
        return EXIT_FAILURE;
    }
    if (args.help)
    {
        printUsage(argv[0]);
        return EXIT_SUCCESS;
    }

    if (args.debug)
    {
        gLogger.setLevel(nvinfer1::ILogger::Severity::kVERBOSE);
    }
    else
    {
        gLogger.setLevel(nvinfer1::ILogger::Severity::kINFO);
    }

    std::string configPath = args.onnxDir + "/config.json";
    std::ifstream configFile(configPath);
    if (!configFile.good())
    {
        LOG_ERROR("config.json not found in onnx directory: %s", args.onnxDir.c_str());
        return EXIT_FAILURE;
    }
    configFile.close();

    if (args.minSeqLen > args.maxSeqLen)
    {
        LOG_ERROR("minSeqLen (%ld) must be <= maxSeqLen (%ld)", args.minSeqLen, args.maxSeqLen);
        return EXIT_FAILURE;
    }
    if (args.optSeqLen < args.minSeqLen || args.optSeqLen > args.maxSeqLen)
    {
        LOG_ERROR("optSeqLen (%ld) must be between minSeqLen and maxSeqLen", args.optSeqLen);
        return EXIT_FAILURE;
    }

    builder::ActionBuilderConfig config;
    config.minSeqLen = args.minSeqLen;
    config.maxSeqLen = args.maxSeqLen;
    config.optSeqLen = args.optSeqLen;

    std::string actualEngineDir = args.engineDir + "/action";

    builder::ActionBuilder actionBuilder(args.onnxDir, actualEngineDir, config);
    if (!actionBuilder.build())
    {
        LOG_ERROR("Failed to build Action engine.");
        return EXIT_FAILURE;
    }

    LOG_INFO("Action engine built successfully.");
    return EXIT_SUCCESS;
}
