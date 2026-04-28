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

#pragma once

#include <NvInfer.h>
#include <filesystem>
#include <nlohmann/json.hpp>
#include <sstream>
#include <string>

using Json = nlohmann::json;

namespace trt_edgellm
{

namespace builder
{

//! Configuration structure for action expert model building.
//! Contains parameters needed to build the TensorRT engine from the action expert ONNX
//! (e.g. Alpamayo 1 flow-matching decoder).
struct ActionBuilderConfig
{
    //! Minimum sequence length for dynamic dimensions (e.g. seq_length, past_kv sequence dim).
    //! Must match the range used when exporting the action ONNX (e.g. 2808 for Alpamayo).
    int32_t minSeqLen{2808};

    //! Maximum sequence length for dynamic dimensions.
    //! Must match the export range (e.g. 3424 for Alpamayo).
    int32_t maxSeqLen{3424};

    //! Optimal sequence length for TensorRT to optimize for.
    //! When loading from JSON, if omitted, defaults to (minSeqLen + maxSeqLen) / 2 in fromJson().
    int32_t optSeqLen{(2808 + 3424) / 2};

    //! Convert configuration to JSON format for serialization.
    //! @return JSON object containing all configuration parameters
    Json toJson() const
    {
        Json json;
        json["min_seq_len"] = minSeqLen;
        json["max_seq_len"] = maxSeqLen;
        json["opt_seq_len"] = optSeqLen;
        return json;
    }

    //! Create configuration from JSON format.
    //! If opt_seq_len is omitted, sets optSeqLen to (minSeqLen + maxSeqLen) / 2.
    //! @param json JSON object containing configuration parameters
    //! @return ActionBuilderConfig object with parsed parameters
    static ActionBuilderConfig fromJson(Json const& json)
    {
        ActionBuilderConfig config;
        if (json.contains("min_seq_len"))
        {
            config.minSeqLen = json["min_seq_len"];
        }
        if (json.contains("max_seq_len"))
        {
            config.maxSeqLen = json["max_seq_len"];
        }
        if (json.contains("opt_seq_len"))
        {
            config.optSeqLen = json["opt_seq_len"];
        }
        else
        {
            config.optSeqLen = (config.minSeqLen + config.maxSeqLen) / 2;
        }
        return config;
    }

    //! Convert configuration to human-readable string format.
    //! @return String representation of the configuration for debugging/logging
    std::string toString() const
    {
        std::ostringstream oss;
        oss << "ActionBuilderConfig:\n";
        oss << "  minSeqLen: " << minSeqLen << "\n";
        oss << "  maxSeqLen: " << maxSeqLen << "\n";
        oss << "  optSeqLen: " << optSeqLen << "\n";
        return oss.str();
    }
};

//! Builder class for action expert TensorRT engines.
//! Handles the complete process of building TensorRT engines from ONNX models
//! for action experts used in multimodal models (e.g., Alpamayo 1).
class ActionBuilder
{
public:
    //! Constructor for ActionBuilder.
    //! @param onnxDir Directory containing the ONNX model and configuration files
    //! @param engineDir Directory where the built engine and related files will be saved
    //! @param config Configuration object specifying build parameters
    ActionBuilder(std::filesystem::path const& onnxDir, std::filesystem::path const& engineDir,
        ActionBuilderConfig const& config);

    //! Destructor.
    ~ActionBuilder() noexcept = default;

    //! Build the TensorRT engine from the action expert ONNX.
    //! This method performs the complete build process including:
    //! - Loading and parsing the ONNX model
    //! - Setting up optimization profiles
    //! - Building the TensorRT engine
    //! - Copying necessary files to the engine directory
    //! @return true if build was successful, false otherwise
    bool build();

private:
    std::filesystem::path mOnnxDir;     //!< Directory containing ONNX model files
    std::filesystem::path mEngineDir;   //!< Directory for saving built engine
    ActionBuilderConfig mBuilderConfig; //!< Build configuration

    //! Parse the model configuration from config.json
    //! @return true if parsing was successful, false otherwise
    bool parseConfig();

    //! Set up optimization profile for action expert.
    //! Creates optimization profile with appropriate dynamic shapes for action inputs.
    //! @param builder TensorRT builder object (must not be null)
    //! @param config TensorRT builder config object (must not be null)
    //! @param network TensorRT network definition (must not be null)
    //! @return true if setup was successful, false otherwise
    bool setupActionOptimizationProfile(
        nvinfer1::IBuilder& builder, nvinfer1::IBuilderConfig& config, nvinfer1::INetworkDefinition const& network);

    //! Copy and save the model configuration with builder config.
    //! Creates a config.json file in the engine directory with both original model config
    //! and builder configuration parameters.
    //! @return true if copying was successful, false otherwise
    bool copyConfig();

    Json mModelConfig; //!< Parsed model configuration (from config.json)
};

} // namespace builder
} // namespace trt_edgellm
