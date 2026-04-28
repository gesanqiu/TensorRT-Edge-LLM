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

#include "common/tensor.h"
#include <cuda_runtime.h>
#include <functional>
#include <optional>

namespace trt_edgellm
{
namespace kernel
{

//! \brief Unified embedding lookup kernel (supports FP16 and FP8 tables)
//!
//! Automatically dispatches to FP16 or FP8 implementation based on the embedding table's datatype.
//! For FP8 tables, scales must be provided for per-group dequantization.
//!
//! \param[in] inputIds Input token IDs with shape [batchSize, seqLen]
//! \param[in] embeddingTable Embedding table with shape [vocabSize, hiddenSize] (FP16 or FP8)
//! \param[in] scales FP32 per-group scales with shape [vocabSize, hiddenSize / blockSize]
//!                   Required when embeddingTable is FP8, std::nullopt for FP16
//! \param[out] output Hidden states with shape [batchSize, seqLen, hiddenSize]
//! \param[in] stream CUDA stream for execution
//! \throws std::runtime_error if tensor shapes or data types are invalid, or FP8 is not supported
void embeddingLookup(rt::Tensor const& inputIds, rt::Tensor const& embeddingTable, rt::OptionalInputTensor scales,
    rt::Tensor& output, cudaStream_t stream);

//! \brief Unified embedding lookup with image embedding insertion (supports FP16 and FP8 tables)
//!
//! For legacy multimodal models (Qwen2-VL, InternVL) where tokenId > vocabSize indicates image tokens.
//! Automatically dispatches to FP16 or FP8 implementation based on the embedding table's datatype.
//! Text tokens use the embedding table, image tokens use FP16 imageEmbeds.
//!
//! \param[in] inputIds Input token IDs with shape [batchSize, seqLen]
//! \param[in] embeddingTable Embedding table with shape [vocabSize, hiddenSize] (FP16 or FP8)
//! \param[in] scales FP32 per-group scales with shape [vocabSize, hiddenSize / blockSize]
//!                   Required when embeddingTable is FP8, std::nullopt for FP16
//! \param[in] imageEmbeds Image embeddings with shape [imageTokenLen, hiddenSize], dtype FP16
//! \param[out] output Hidden states with shape [batchSize, seqLen, hiddenSize]
//! \param[in] stream CUDA stream for execution
//! \throws std::runtime_error if tensor shapes, data types are invalid, or FP8 is not supported
void embeddingLookupWithImageInsertion(rt::Tensor const& inputIds, rt::Tensor const& embeddingTable,
    rt::OptionalInputTensor scales, rt::Tensor const& imageEmbeds, rt::Tensor& output, cudaStream_t stream);

//! \brief Assemble deepstack embeddings by extracting image token embeddings from deepstack features
//!
//! This function processes input token IDs and selectively extracts embeddings for image tokens from
//! the provided deepstack features. Image tokens are identified in two ways:
//! - Legacy: token IDs >= vocabSize (Qwen2.5-VL where image tokens start at vocabSize)
//! - Explicit: token ID == imageTokenId (Qwen3-Omni where image tokens are within vocab)
//!
//! When multimodalIndices is provided, it is used to index into deepstackFeatures (required for
//! Qwen3-Omni where all image tokens share the same ID). Otherwise falls back to tokenId - vocabSize.
//!
//! \param[in] inputIds Input token IDs with shape [batchSize, seqLen]
//! \param[in] deepstackFeatures Deepstack image features with shape [numImageTokens, hiddenSize]
//! \param[in] vocabSize Vocabulary size (legacy threshold for image token detection)
//! \param[in] imageTokenId Explicit image token ID (0 = not set, use legacy >= vocabSize detection)
//! \param[in] multimodalIndices Pre-computed indices for image embeddings [batchSize, seqLen],
//!                              or std::nullopt to use legacy tokenId - vocabSize indexing
//! \param[out] deepstackEmbeds Output embeddings with shape [batchSize, seqLen, hiddenSize]
//! \param[in] stream CUDA stream for execution
//! \throws std::runtime_error if tensor shapes or data types are invalid
void assembleDeepstackEmbedding(rt::Tensor const& inputIds, rt::Tensor const& deepstackFeatures, int32_t vocabSize,
    rt::Tensor& deepstackEmbeds, cudaStream_t stream, int32_t imageTokenId = 0,
    rt::OptionalInputTensor multimodalIndices = std::nullopt);

//! \brief Unified embedding lookup with optional image and audio embeddings (supports FP16 and FP8 tables)
//!
//! Automatically dispatches to FP16 or FP8 implementation based on the embedding table's datatype.
//! This kernel handles up to three types of tokens:
//! - Normal text tokens (0 <= tokenId < vocabSize): lookup from embeddingTable
//! - Image tokens (tokenId == imageTokenId): lookup from imageEmbeds using multimodalIndices (optional)
//! - Audio tokens (tokenId == audioTokenId): lookup from audioEmbeds using multimodalIndices (optional)
//!
//! The multimodalIndices provides pre-computed indices into audioEmbeds/imageEmbeds for each
//! position. For text tokens, the multimodalIndices value is not used.
//! To indicate the presence of a modality, both token ID and the corresponding embedding tensor must be provided.
//!
//! \param[in] inputIds Input token IDs with shape [batchSize, seqLen]
//! \param[in] embeddingTable Text embedding table with shape [vocabSize, hiddenSize] (FP16 or FP8)
//! \param[in] scales FP32 per-group scales with shape [vocabSize, hiddenSize / blockSize]
//!                   Required when embeddingTable is FP8, std::nullopt for FP16
//! \param[in] multimodalIndices Pre-computed indices for audio/image embeddings [batchSize, seqLen],
//!                              can be std::nullopt if no image/audio inputs are provided
//! \param[in] imageTokenId Special token ID for image (e.g., 151655 in Qwen3), or std::nullopt if no image
//! \param[in] imageEmbeds Image embeddings with shape [totalImageTokens, hiddenSize], or std::nullopt if no image
//! \param[in] audioTokenId Special token ID for audio (e.g., 151675 in Qwen3), or std::nullopt if no audio
//! \param[in] audioEmbeds Audio embeddings with shape [totalAudioTokens, hiddenSize], or std::nullopt if no audio
//! \param[out] output Hidden states with shape [batchSize, seqLen, hiddenSize]
//! \param[in] stream CUDA stream for execution
//!
//! \note audioTokenId and imageTokenId are allowed to be smaller than vocabSize, as in the case of Qwen3.
//! \note Embeddings should contain data in the order specified by multimodalIndices
//! \note When a modality is not needed, pass std::nullopt for both its tokenId and embeds
//! \note multimodalIndices can be std::nullopt only when both imageEmbeds and audioEmbeds are std::nullopt
//! \throws std::runtime_error if tensor shapes, data types are invalid, or FP8 is not supported
void embeddingLookupMultimodal(rt::Tensor const& inputIds, rt::Tensor const& embeddingTable,
    rt::OptionalInputTensor scales, rt::OptionalInputTensor multimodalIndices, std::optional<int32_t> imageTokenId,
    rt::OptionalInputTensor imageEmbeds, std::optional<int32_t> audioTokenId, rt::OptionalInputTensor audioEmbeds,
    rt::Tensor& output, cudaStream_t stream);

} // namespace kernel
} // namespace trt_edgellm
