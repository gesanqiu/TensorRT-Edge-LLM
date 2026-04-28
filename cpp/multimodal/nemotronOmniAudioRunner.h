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

#pragma once

#include "multimodalRunner.h"
#include "runtime/audioUtils.h"
#include <cuda_fp16.h>
#include <memory>
#include <string>
#include <vector>

namespace trt_edgellm
{
namespace rt
{

//! \brief Configuration for Nemotron-Omni Parakeet audio encoder
struct NemotronOmniAudioConfig
{
    int32_t melBins{0};           //!< Number of mel-frequency bins
    int32_t audioFeatureDim{0};   //!< Audio feature dimension (LLM hidden size)
    int32_t subsamplingFactor{0}; //!< Parakeet subsampling factor
    int32_t samplingRate{0};      //!< Audio sampling rate

    int32_t soundContextTokenId{0}; //!< <so_embedding> token ID
    int32_t vocabSize{0};           //!< Vocabulary size (audio token ID offset)
};

//! \brief Runner for Nemotron-Omni Parakeet audio encoder
//!
//! Handles audio preprocessing and encoder inference for Nemotron-Omni's
//! Parakeet-based audio encoder. The encoder takes mel-spectrogram features
//! and an attention mask, producing audio embeddings projected to LLM hidden size.
class NemotronOmniAudioRunner : public MultimodalRunner
{
public:
    //! \brief Constructor for NemotronOmniAudioRunner
    //! \param[in] engineDir Directory containing the audio encoder engine
    //! \param[in] stream CUDA stream for execution
    //! \throws std::runtime_error if engine loading fails or configuration is invalid
    NemotronOmniAudioRunner(std::string const& engineDir, cudaStream_t stream);

    ~NemotronOmniAudioRunner() noexcept = default;

    //! \brief Preprocess multimodal input including audio and text
    //! \param[in] request LLM generation request containing audio and text
    //! \param[in,out] batchedInputIds Batched input token IDs after preprocessing
    //! \param[in] tokenizer Tokenizer for text processing
    //! \param[in,out] ropeRotaryCosSinDevice RoPE rotary position encoding cache (unused by this model)
    //! \param[in] stream CUDA stream for execution
    //! \return True if preprocessing succeeded, false otherwise
    bool preprocess(rt::LLMGenerationRequest const& request, std::vector<std::vector<int32_t>>& batchedInputIds,
        tokenizer::Tokenizer const* tokenizer, rt::Tensor& ropeRotaryCosSinDevice, cudaStream_t stream,
        bool imageOnly = false) override;

    //! \brief Run inference on the audio encoder
    //! \param[in] stream CUDA stream for execution
    //! \return True if inference succeeded, false otherwise
    bool infer(cudaStream_t stream) override;

    //! \brief Validate and load configuration from JSON file
    //! \param[in] engineDir Path to engine directory
    //! \return True if configuration is valid and loaded successfully, false otherwise
    bool validateAndFillConfig(std::string const& engineDir) override;

    //! \brief Allocate buffers for inference
    //! \param[in] stream CUDA stream for execution
    //! \return True if allocation succeeded, false otherwise
    bool allocateBuffer(cudaStream_t stream) override;

    //! \brief Get audio embeddings from encoder output
    //! \return Reference to audio embedding tensor
    rt::Tensor& getOutputEmbedding() override;

private:
    //! \brief Load pre-computed mel-spectrogram from file
    //! \param[in] filePath Path to .npy or .raw file
    //! \param[in] format File format: "npy" or "raw"
    //! \param[out] melSpectrogram Output tensor [1, seq_len, mel_bins]
    //! \param[in] stream CUDA stream for execution
    //! \return True on success, false otherwise
    bool loadMelSpectrogramFromFile(
        std::string const& filePath, std::string const& format, rt::Tensor& melSpectrogram, cudaStream_t stream);

    //! \brief Preprocess audio buffers and prepare engine inputs
    //! \param[in] audioBuffers Input audio data with mel-spectrogram paths or waveforms
    //! \param[out] audioTokenLengths Output token lengths for each audio clip
    //! \param[in] stream CUDA stream for execution
    //! \return True if preprocessing succeeded, false otherwise
    bool preprocessAudio(std::vector<rt::audioUtils::AudioData> const& audioBuffers,
        std::vector<int64_t>& audioTokenLengths, cudaStream_t stream);

    //! \brief Tokenize text and insert audio placeholder tokens
    //! \param[in] request LLM generation request
    //! \param[out] batchInputIds Batched input IDs after tokenization and audio token insertion
    //! \param[in] audioTokenLengths Token lengths for each audio clip
    //! \param[in] tokenizer Tokenizer for text encoding
    void textPreprocess(rt::LLMGenerationRequest const& request, std::vector<std::vector<int32_t>>& batchInputIds,
        std::vector<int64_t> const& audioTokenLengths, tokenizer::Tokenizer const* tokenizer);

    NemotronOmniAudioConfig mConfig{}; //!< Nemotron-Omni Parakeet audio configuration
    rt::Tensor mInputFeatures{};       //!< [batch, seq_len, mel_bins] Mel-spectrogram encoder input
    rt::Tensor mAttentionMask{};       //!< [batch, seq_len] Attention mask for variable-length sequences
    rt::Tensor mAudioEmbedding{};      //!< [batch, encoded_seq_len, hidden_dim] Audio encoder output
    int64_t mMaxSeqLen{0};             //!< Maximum mel-spectrogram sequence length
};

} // namespace rt
} // namespace trt_edgellm
