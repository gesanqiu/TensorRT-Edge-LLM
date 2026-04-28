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

#include "runtime/streaming.h"

#include "common/logger.h"
#include "common/utf8.h"
#include "runtime/llmInferenceSpecDecodeRuntime.h"
#include "runtime/llmRuntimeUtils.h"
#include "tokenizer/tokenizer.h"

#include <algorithm>
#include <stdexcept>

namespace trt_edgellm
{
namespace rt
{

//=============================================================================
// StreamChannel — consumer API + friend-reachable producer API
//=============================================================================

std::shared_ptr<StreamChannel> StreamChannel::create()
{
    // std::make_shared cannot access the private constructor; use the new/shared_ptr form.
    return std::shared_ptr<StreamChannel>(new StreamChannel());
}

void StreamChannel::push(StreamChunk chunk)
{
    {
        std::lock_guard<std::mutex> lk(mMutex);
        mPending.push_back(std::move(chunk));
    }
    mCv.notify_one();
}

void StreamChannel::finish(FinishReason r)
{
    {
        std::lock_guard<std::mutex> lk(mMutex);
        if (mFinished.load(std::memory_order_relaxed))
        {
            return; // Idempotent — first caller wins.
        }
        mReason.store(r, std::memory_order_release);
        mFinished.store(true, std::memory_order_release);
    }
    mCv.notify_all();
}

void StreamChannel::setOriginalBatchIdx(int32_t idx) noexcept
{
    mOriginalBatchIdx.store(idx, std::memory_order_release);
}

bool StreamChannel::isFinished() const noexcept
{
    return mFinished.load(std::memory_order_acquire);
}

FinishReason StreamChannel::getReason() const noexcept
{
    if (!mFinished.load(std::memory_order_acquire))
    {
        return FinishReason::kNotFinished;
    }
    return mReason.load(std::memory_order_relaxed);
}

int32_t StreamChannel::getOriginalBatchIdx() const noexcept
{
    return mOriginalBatchIdx.load(std::memory_order_acquire);
}

bool StreamChannel::isCancelled() const noexcept
{
    return mCancelled.load(std::memory_order_acquire);
}

void StreamChannel::cancel() noexcept
{
    mCancelled.store(true, std::memory_order_release);
    {
        std::lock_guard<std::mutex> lk(mMutex);
    }
    mCv.notify_all();
}

void StreamChannel::setStreamInterval(int32_t n) noexcept
{
    mStreamInterval.store(std::max(1, n), std::memory_order_release);
}

int32_t StreamChannel::getStreamInterval() const noexcept
{
    return mStreamInterval.load(std::memory_order_acquire);
}

void StreamChannel::setSkipSpecialTokens(bool skip) noexcept
{
    mSkipSpecialTokens.store(skip, std::memory_order_release);
}

bool StreamChannel::getSkipSpecialTokens() const noexcept
{
    return mSkipSpecialTokens.load(std::memory_order_acquire);
}

std::optional<StreamChunk> StreamChannel::tryPop()
{
    std::lock_guard<std::mutex> lk(mMutex);
    if (mPending.empty())
    {
        return std::nullopt;
    }
    StreamChunk out = std::move(mPending.front());
    mPending.pop_front();
    return out;
}

std::optional<StreamChunk> StreamChannel::waitPop(std::chrono::milliseconds timeout)
{
    std::unique_lock<std::mutex> lk(mMutex);
    if (!mCv.wait_for(lk, timeout, [&] {
            return !mPending.empty() || mFinished.load(std::memory_order_acquire)
                || mCancelled.load(std::memory_order_acquire);
        }))
    {
        return std::nullopt;
    }
    if (mPending.empty())
    {
        return std::nullopt;
    }
    StreamChunk out = std::move(mPending.front());
    mPending.pop_front();
    return out;
}

//=============================================================================
// Streaming helpers — free functions (friends of StreamChannel)
//=============================================================================

void attachStreamChannel(std::shared_ptr<StreamChannel> const& channel, int32_t originalIdx)
{
    bool const wasAttached = channel->mAttachedToRequest.exchange(true, std::memory_order_acq_rel);
    if (wasAttached)
    {
        // validateStreamingSubmission should have caught this; belt-and-suspenders.
        throw std::runtime_error("StreamChannel already attached to a live request");
    }
    channel->setOriginalBatchIdx(originalIdx);
}

bool validateStreamingSubmission(LLMGenerationRequest const& request)
{
    if (request.streamChannels.empty())
    {
        return true;
    }
    if (request.streamChannels.size() != request.requests.size())
    {
        LOG_ERROR("streamChannels size (%zu) must equal requests size (%zu)", request.streamChannels.size(),
            request.requests.size());
        return false;
    }
    for (auto const& ch : request.streamChannels)
    {
        if (!ch)
        {
            continue; // Null entry opts out of streaming for this slot.
        }
        if (ch->isFinished())
        {
            LOG_ERROR("StreamChannel already finished — create a new channel per request");
            return false;
        }
        if (ch->mAttachedToRequest.load(std::memory_order_acquire))
        {
            LOG_ERROR("StreamChannel already attached to a live request — concurrent reuse forbidden");
            return false;
        }
    }
    return true;
}

void applyCancellationToFinishStates(SpecDecodeInferenceContext& context)
{
    for (int32_t i = 0; i < context.activeBatchSize; ++i)
    {
        auto& s = context.slotStreams[i];
        if (!s.channel)
        {
            continue;
        }
        if (context.finishedStates[i])
        {
            continue; // First-writer-wins: don't overwrite a natural-finish reason.
        }
        if (s.channel->mCancelled.load(std::memory_order_acquire))
        {
            context.finishedStates[i] = 1;
            s.terminalReason = FinishReason::kCancelled;
            LOG_DEBUG("Batch %d finished, reason: cancel", i);
        }
    }
}

void emitChunks(SpecDecodeInferenceContext& context, tokenizer::Tokenizer const& tok)
{
    for (int32_t i = 0; i < context.activeBatchSize; ++i)
    {
        auto& s = context.slotStreams[i];
        if (!s.channel)
        {
            continue;
        }

        bool const isFinal = context.finishedStates[i] != 0;
        int32_t const streamInterval = s.channel->getStreamInterval();
        size_t const newlyAvailable = context.tokenIds[i].size() - s.sentTokenCount;
        bool const intervalMet = static_cast<int32_t>(newlyAvailable) >= streamInterval;

        if (!isFinal && !intervalMet)
        {
            continue;
        }

        size_t const snapLastEmitted = s.lastEmittedTokenCount;
        bool const skipSpecial = s.channel->getSkipSpecialTokens();
        std::string delta = tokenizer::emitDelta(s, tok, context.tokenIds[i], skipSpecial);
        if (isFinal && !s.pendingBytes.empty())
        {
            delta.append(tokenizer::emitDeltaFlush(s));
        }

        if (!delta.empty() || isFinal)
        {
            StreamChunk chunk;
            // Delta tokens span [lastEmittedTokenCount, sentTokenCount) — correctly
            // accumulates across prior holds where delta was empty.
            chunk.tokenIds.assign(context.tokenIds[i].begin() + static_cast<std::ptrdiff_t>(snapLastEmitted),
                context.tokenIds[i].begin() + static_cast<std::ptrdiff_t>(s.sentTokenCount));
            chunk.text = std::move(delta);
            chunk.finished = isFinal;
            if (isFinal)
            {
                chunk.reason = s.terminalReason;
            }

            s.channel->push(std::move(chunk));
            s.lastEmittedTokenCount = s.sentTokenCount;

            if (isFinal)
            {
                s.channel->finish(s.terminalReason);
            }
        }
    }
}

//=============================================================================
// StreamChannelFinalizer — RAII terminal-chunk guarantee
//=============================================================================

StreamChannelFinalizer::StreamChannelFinalizer(
    SpecDecodeInferenceContext& ctx, tokenizer::Tokenizer const& tok) noexcept
    : mCtx(ctx)
    , mTok(tok)
{
}

StreamChannelFinalizer::~StreamChannelFinalizer() noexcept
{
    // Destructor runs during unwind on every exit path from handleRequest.
    // Allocations in chunk assembly may throw on OOM — swallow them because the
    // essential invariant (consumer must unblock) is carried by finish() alone.
    for (size_t i = 0; i < mCtx.slotStreams.size(); ++i)
    {
        auto& s = mCtx.slotStreams[i];
        if (!s.channel)
        {
            continue;
        }
        if (s.channel->isFinished())
        {
            continue; // Already terminated via the normal emit path.
        }

        try
        {
            StreamChunk chunk;

            // Any tokens generated but not yet pushed (held by streamInterval
            // or appended just before the aborting exit path).
            if (i < mCtx.tokenIds.size() && s.lastEmittedTokenCount < mCtx.tokenIds[i].size())
            {
                chunk.tokenIds.assign(mCtx.tokenIds[i].begin() + static_cast<std::ptrdiff_t>(s.lastEmittedTokenCount),
                    mCtx.tokenIds[i].end());
            }

            // Decode any un-decoded tokens via idToPiece, then sanitize+flush.
            bool const skipSpecial = s.channel->getSkipSpecialTokens();
            std::string raw;
            if (i < mCtx.tokenIds.size())
            {
                for (size_t k = s.sentTokenCount; k < mCtx.tokenIds[i].size(); ++k)
                {
                    raw.append(mTok.idToPiece(mCtx.tokenIds[i][k], skipSpecial));
                }
            }
            chunk.text = utf8::sanitizeUtf8Streaming(raw, s.pendingBytes);
            chunk.text.append(utf8::sanitizeUtf8Flush(s.pendingBytes));

            chunk.finished = true;
            chunk.reason = FinishReason::kError;
            s.channel->push(std::move(chunk));
        }
        catch (...)
        {
            // Best-effort push; finish() below still unblocks the consumer.
        }

        s.channel->finish(FinishReason::kError);
    }
}

// Explicit instantiation lives next to the template definition (in
// llmRuntimeUtils.cpp). It's forward-declared there with SlotStreamState as
// a complete type via this header's include.

} // namespace rt
} // namespace trt_edgellm
