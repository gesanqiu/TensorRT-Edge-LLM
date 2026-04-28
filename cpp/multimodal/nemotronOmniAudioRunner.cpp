/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "nemotronOmniAudioRunner.h"
#include "common/checkMacros.h"
#include "common/mmapReader.h"
#include "common/safetensorsUtils.h"
#include "profiling/metrics.h"
#include "profiling/timer.h"
#include <fstream>
#include <nlohmann/json.hpp>

using Json = nlohmann::json;

namespace trt_edgellm
{
namespace rt
{

NemotronOmniAudioRunner::NemotronOmniAudioRunner(std::string const& engineDir, cudaStream_t stream)
    : MultimodalRunner()
{
    // Audio runner does not use the visual engine from base class.
    // Load the audio engine instead.
    mRuntime = std::unique_ptr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(gLogger));

    std::string const enginePath = engineDir + "/audio_encoder.engine";
    auto mmapReader = std::make_unique<file_io::MmapReader>(enginePath);
    mAudioEngine = std::unique_ptr<nvinfer1::ICudaEngine>(
        mRuntime->deserializeCudaEngine(mmapReader->getData(), mmapReader->getSize()));

    mAudioContext = std::unique_ptr<nvinfer1::IExecutionContext>(
        mAudioEngine->createExecutionContext(nvinfer1::ExecutionContextAllocationStrategy::kUSER_MANAGED));
    if (!mAudioContext->setOptimizationProfileAsync(0, stream))
    {
        throw std::runtime_error("Failed to set optimization profile for audio engine");
    }

    if (!validateAndFillConfig(engineDir))
    {
        throw std::runtime_error("NemotronOmniAudioRunner: Failed to validate config");
    }
    if (!allocateBuffer(stream))
    {
        throw std::runtime_error("NemotronOmniAudioRunner: Failed to allocate buffer");
    }
}

bool NemotronOmniAudioRunner::validateAndFillConfig(std::string const& engineDir)
{
    Json jsonConfig;
    std::string const configPath = engineDir + "/config.json";
    std::ifstream configFileStream(configPath);
    if (!configFileStream.is_open())
    {
        LOG_ERROR("Failed to open config file: %s", configPath.c_str());
        return false;
    }

    try
    {
        jsonConfig = Json::parse(configFileStream);
        configFileStream.close();
    }
    catch (Json::parse_error const& e)
    {
        LOG_ERROR("Failed to parse config file: %s", e.what());
        return false;
    }

    mModelType = multimodal::ModelType::NEMOTRON_OMNI_AUDIO_ENCODER;

    if (!jsonConfig.contains("sound_config"))
    {
        LOG_ERROR("sound_config not found in config.json");
        return false;
    }
    auto const& soundConfig = jsonConfig["sound_config"];

    if (!soundConfig.contains("num_mel_bins"))
    {
        LOG_ERROR("sound_config.num_mel_bins not found in config.json");
        return false;
    }
    mConfig.melBins = soundConfig["num_mel_bins"].get<int32_t>();

    if (soundConfig.contains("subsampling_factor"))
    {
        mConfig.subsamplingFactor = soundConfig["subsampling_factor"].get<int32_t>();
    }
    else
    {
        LOG_ERROR("sound_config.subsampling_factor not found in config.json");
        return false;
    }

    if (soundConfig.contains("sampling_rate"))
    {
        mConfig.samplingRate = soundConfig["sampling_rate"].get<int32_t>();
    }
    else
    {
        LOG_ERROR("sound_config.sampling_rate not found in config.json");
        return false;
    }

    if (!jsonConfig.contains("sound_context_token_id"))
    {
        LOG_ERROR("sound_context_token_id not found in config.json");
        return false;
    }
    mConfig.soundContextTokenId = jsonConfig["sound_context_token_id"].get<int32_t>();

    if (jsonConfig.contains("llm_config") && jsonConfig["llm_config"].contains("vocab_size"))
    {
        mConfig.vocabSize = jsonConfig["llm_config"]["vocab_size"].get<int32_t>();
    }

    nvinfer1::Dims const inputShapeMax
        = mAudioEngine->getProfileShape("input_features", 0, nvinfer1::OptProfileSelector::kMAX);
    mMaxSeqLen = inputShapeMax.d[1];

    nvinfer1::Dims const outputShape = mAudioEngine->getTensorShape("last_hidden_state");
    mConfig.audioFeatureDim = outputShape.d[2];

    LOG_INFO("NemotronOmniAudioRunner: melBins=%d, maxSeqLen=%ld, hiddenDim=%d, soundTokenId=%d", mConfig.melBins,
        mMaxSeqLen, mConfig.audioFeatureDim, mConfig.soundContextTokenId);

    return true;
}

bool NemotronOmniAudioRunner::allocateBuffer(cudaStream_t stream)
{
    bool status{true};

    mInputFeatures = rt::Tensor({1, mMaxSeqLen, mConfig.melBins}, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF,
        "NemotronOmniAudioRunner::mInputFeatures");
    status &= mAudioContext->setTensorAddress("input_features", mInputFeatures.rawPointer());

    mAttentionMask = rt::Tensor(
        {1, mMaxSeqLen}, rt::DeviceType::kGPU, nvinfer1::DataType::kINT64, "NemotronOmniAudioRunner::mAttentionMask");
    status &= mAudioContext->setTensorAddress("attention_mask", mAttentionMask.rawPointer());

    int64_t const maxEncodedSeqLen = mMaxSeqLen / mConfig.subsamplingFactor;
    mAudioEmbedding = rt::Tensor({1, maxEncodedSeqLen, mConfig.audioFeatureDim}, rt::DeviceType::kGPU,
        nvinfer1::DataType::kHALF, "NemotronOmniAudioRunner::mAudioEmbedding");
    status &= mAudioContext->setTensorAddress("last_hidden_state", mAudioEmbedding.rawPointer());

    if (!status)
    {
        LOG_ERROR("Failed to set tensor addresses for audio engine");
        return false;
    }

    return true;
}

bool NemotronOmniAudioRunner::loadMelSpectrogramFromFile(
    std::string const& filePath, std::string const& format, rt::Tensor& melSpectrogram, cudaStream_t stream)
{
    if (format != "safetensors")
    {
        LOG_ERROR("Only safetensors format is supported. Got: %s", format.c_str());
        return false;
    }

    std::vector<rt::Tensor> tensors;
    if (!safetensors::loadSafetensors(filePath, tensors, stream))
    {
        LOG_ERROR("Failed to load mel-spectrogram from: %s", filePath.c_str());
        return false;
    }

    check::check(tensors.size() == 1, "Mel-spectrogram safetensors should contain exactly one tensor");
    check::check(tensors[0].getDataType() == nvinfer1::DataType::kHALF, "Mel-spectrogram must be FP16");

    melSpectrogram = std::move(tensors[0]);
    return true;
}

bool NemotronOmniAudioRunner::preprocessAudio(std::vector<rt::audioUtils::AudioData> const& audioBuffers,
    std::vector<int64_t>& audioTokenLengths, cudaStream_t stream)
{
    if (audioBuffers.empty())
    {
        return true;
    }

    if (audioBuffers.size() > 1)
    {
        LOG_WARNING(
            "Nemotron-Omni audio: only single audio clip supported, ignoring %zu extra clips", audioBuffers.size() - 1);
    }
    auto const& audio = audioBuffers[0];

    // Load pre-computed mel-spectrogram from file
    if (audio.melSpectrogramPath.empty())
    {
        LOG_ERROR("Nemotron-Omni audio runner requires mel-spectrogram input (melSpectrogramPath)");
        return false;
    }

    rt::Tensor melSpectrogram;
    if (!loadMelSpectrogramFromFile(audio.melSpectrogramPath, audio.melSpectrogramFormat, melSpectrogram, stream))
    {
        LOG_ERROR("Failed to load mel-spectrogram from %s", audio.melSpectrogramPath.c_str());
        return false;
    }

    // melSpectrogram shape: [1, time_steps, mel_bins] for Parakeet
    int64_t const rawSeqLen = melSpectrogram.getShape()[1];

    // Pad to a multiple of subsamplingFactor (3× stride-2 CNN requires divisible input)
    auto alignUp = [](int64_t x, int64_t a) { return (x + a - 1) / a * a; };
    int64_t const seqLen = alignUp(rawSeqLen, mConfig.subsamplingFactor);
    int64_t const encodedSeqLen = seqLen / mConfig.subsamplingFactor;
    audioTokenLengths.push_back(encodedSeqLen);

    // Copy mel-spectrogram to input tensor (zero-padded if needed)
    check::check(mInputFeatures.reshape({1, seqLen, static_cast<int64_t>(mConfig.melBins)}), "Tensor reshape failed");
    if (seqLen > rawSeqLen)
    {
        CUDA_CHECK(cudaMemsetAsync(mInputFeatures.rawPointer(), 0, mInputFeatures.getMemoryCapacity(), stream));
    }
    CUDA_CHECK(cudaMemcpyAsync(mInputFeatures.rawPointer(), melSpectrogram.rawPointer(),
        melSpectrogram.getMemoryCapacity(), cudaMemcpyDeviceToDevice, stream));

    // Attention mask: 1 for valid frames, 0 for padding
    std::vector<int64_t> maskHost(seqLen, 0);
    std::fill_n(maskHost.begin(), rawSeqLen, 1);
    check::check(mAttentionMask.reshape({1, seqLen}), "Tensor reshape failed");
    CUDA_CHECK(cudaMemcpyAsync(
        mAttentionMask.rawPointer(), maskHost.data(), seqLen * sizeof(int64_t), cudaMemcpyHostToDevice, stream));

    // Reshape output
    check::check(mAudioEmbedding.reshape({1, encodedSeqLen, static_cast<int64_t>(mConfig.audioFeatureDim)}),
        "Tensor reshape failed");

    mMultimodalMetrics.recordRun(1, encodedSeqLen);

    return true;
}

void NemotronOmniAudioRunner::textPreprocess(rt::LLMGenerationRequest const& request,
    std::vector<std::vector<int32_t>>& batchInputIds, std::vector<int64_t> const& audioTokenLengths,
    tokenizer::Tokenizer const* tokenizer)
{
    // Repeat each ``<so_embedding>`` placeholder ``audioTokenLengths[i]``
    // times so the runtime multimodal kernel can inject one audio frame per
    // copy. Reuse ``batchInputIds`` if the ViT runner already tokenized
    // (combined image+audio); otherwise tokenize from scratch.
    bool const alreadyTokenized = batchInputIds.size() == request.requests.size();
    int audioIndex = 0;

    for (size_t i = 0; i < request.requests.size(); ++i)
    {
        std::vector<int32_t> ids = alreadyTokenized
            ? std::move(batchInputIds[i])
            : tokenizer->encode(request.formattedRequests[i].formattedCompleteRequest);
        check::check(!ids.empty(), "Failed to encode text");

        std::vector<int32_t> newIds;
        newIds.reserve(ids.size());
        for (auto const& id : ids)
        {
            if (id == mConfig.soundContextTokenId)
            {
                int64_t const numAudioTokens = audioTokenLengths.at(audioIndex);
                for (int64_t k = 0; k < numAudioTokens; ++k)
                {
                    newIds.push_back(mConfig.soundContextTokenId);
                }
                ++audioIndex;
            }
            else
            {
                newIds.push_back(id);
            }
        }
        if (alreadyTokenized)
        {
            batchInputIds[i] = std::move(newIds);
        }
        else
        {
            batchInputIds.emplace_back(std::move(newIds));
        }
    }
}

bool NemotronOmniAudioRunner::preprocess(rt::LLMGenerationRequest const& request,
    std::vector<std::vector<int32_t>>& batchedInputIds, tokenizer::Tokenizer const* tokenizer,
    [[maybe_unused]] rt::Tensor& ropeRotaryCosSinDevice, cudaStream_t stream, [[maybe_unused]] bool imageOnly)
{
    std::vector<int64_t> audioTokenLengths;

    try
    {
        if (!request.requests.empty() && !request.requests[0].audioBuffers.empty())
        {
            if (!preprocessAudio(request.requests[0].audioBuffers, audioTokenLengths, stream))
            {
                return false;
            }
        }
        textPreprocess(request, batchedInputIds, audioTokenLengths, tokenizer);
    }
    catch (std::exception const& e)
    {
        LOG_ERROR("NemotronOmniAudioRunner::preprocess() failed: %s", e.what());
        return false;
    }

    return true;
}

bool NemotronOmniAudioRunner::infer(cudaStream_t stream)
{
    // Skip if no audio input
    if (mInputFeatures.getShape()[1] == 0)
    {
        return true;
    }

    {
        TIME_STAGE(metrics::StageNames::kMULTIMODAL_PROCESSING, stream);

        bool status{true};
        status &= mAudioContext->setInputShape("input_features", mInputFeatures.getShape().getTRTDims());
        status &= mAudioContext->setInputShape("attention_mask", mAttentionMask.getShape().getTRTDims());

        if (!status)
        {
            LOG_ERROR("NemotronOmniAudioRunner::infer(): Failed to set input shapes");
            return false;
        }

        bool const enqueueStatus = mAudioContext->enqueueV3(stream);
        if (!enqueueStatus)
        {
            LOG_ERROR("NemotronOmniAudioRunner::infer(): Failed to enqueue engine");
            return false;
        }
    }

    auto const shape = mAudioEmbedding.getShape();
    check::check(shape.getNumDims() == 3 && shape[0] == 1,
        "NemotronOmniAudioRunner: mAudioEmbedding must be [1, seq, hidden] before reshape");
    check::check(mAudioEmbedding.reshape({shape[1], shape[2]}), "Tensor reshape failed");

    return true;
}

rt::Tensor& NemotronOmniAudioRunner::getOutputEmbedding()
{
    return mAudioEmbedding;
}

} // namespace rt
} // namespace trt_edgellm
