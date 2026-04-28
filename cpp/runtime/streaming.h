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

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace trt_edgellm
{
namespace tokenizer
{
class Tokenizer;
} // namespace tokenizer
namespace rt
{

// Forward declarations. SpecDecodeInferenceContext is defined in the spec-decode
// runtime header (pulled in by the .cpp that implements the helpers); the request
// struct lives in llmRuntimeUtils.h and holds the streamChannels vector.
struct SpecDecodeInferenceContext;
struct LLMGenerationRequest;

/*!
 * @brief Reason a streaming slot terminated. Only meaningful on
 *        `StreamChunk { finished = true }` or via `StreamChannel::getReason()`.
 */
enum class FinishReason : uint8_t
{
    //! The slot is still generating — carried by every non-terminal chunk.
    kNotFinished = 0,
    //! Model sampled the tokenizer's end-of-sequence token. Normal completion.
    kEndId = 1,
    //! Hit `request.maxGenerateLength` (or the KV-clamped cap). Output may be mid-sentence.
    kLength = 2,
    //! Consumer called `channel->cancel()` and the runtime observed it at an iteration
    //! boundary. KV cache released on the next eviction.
    kCancelled = 3,
    //! Runtime aborted — OOM in the finalizer, engine error, exception during
    //! `handleRequest`. Rare; the terminal chunk may omit text under memory pressure.
    kError = 4,
    //! Reserved for future stop-string support. Not emitted today.
    kStopWords = 5,
};

/*!
 * @brief Single delta chunk on a streaming channel.
 *
 * `text` is always well-formed UTF-8 when pushed by the runtime — invalid byte
 * sequences are replaced with U+FFFD before emission. Empty `text` chunks are
 * legal (e.g., terminal chunks after all bytes were emitted).
 */
struct StreamChunk
{
    std::vector<int32_t> tokenIds;                   //!< Delta tokens since last chunk (may be >1 under spec-decode).
    std::string text;                                //!< Delta text; always well-formed UTF-8.
    bool finished{false};                            //!< True for the final chunk on this channel.
    FinishReason reason{FinishReason::kNotFinished}; //!< Terminal reason (only meaningful when `finished==true`).
};

class StreamChannel; // For the friend-function signatures below.

// Free-function streaming helpers. Declared as friends of StreamChannel below
// so they can reach the private producer API (push, finish, setOriginalBatchIdx)
// while consumers remain restricted to consume/tryPop/waitPop/cancel.
void attachStreamChannel(std::shared_ptr<StreamChannel> const& channel, int32_t originalIdx);
bool validateStreamingSubmission(LLMGenerationRequest const& request);
void applyCancellationToFinishStates(SpecDecodeInferenceContext& context);
void emitChunks(SpecDecodeInferenceContext& context, tokenizer::Tokenizer const& tokenizer);

/*!
 * @brief Per-slot streaming channel.
 *
 * Encapsulated MPSC pipe between the runtime (producer) and a single consumer.
 * State is private; consumer interacts via consume()/waitPop()/tryPop()/cancel().
 * Runtime-side operations (push, finish, setOriginalBatchIdx) are accessed via
 * friendship so consumers cannot hold the lock or skip chunks.
 */
class StreamChannel
{
public:
    //! Factory — only way to construct. Enforces shared_ptr ownership.
    static std::shared_ptr<StreamChannel> create();

    StreamChannel(StreamChannel const&) = delete;
    StreamChannel& operator=(StreamChannel const&) = delete;
    StreamChannel(StreamChannel&&) = delete;
    StreamChannel& operator=(StreamChannel&&) = delete;

    ~StreamChannel() noexcept = default;

    // ── Consumer API ─────────────────────────────────────────────────────────

    /*!
     * @brief Block until `finished` or `cancelled`, delivering every chunk.
     *
     * Exits when `pending` is empty AND either `finished` or `cancelled` is set.
     * Lock is held only for pop; `handler` runs with the mutex released.
     *
     * @tparam Handler Callable with signature `void(StreamChunk&&)`.
     * @param handler  Invoked once per chunk delivered.
     * @param poll     Max interval between wakeups (defaults to 100ms).
     */
    template <typename Handler>
    void consume(Handler&& handler, std::chrono::milliseconds poll = std::chrono::milliseconds{100});

    //! Non-blocking single-chunk pop.
    std::optional<StreamChunk> tryPop();

    /*!
     * @brief Blocking single-chunk pop with timeout.
     *
     * Returns std::nullopt on timeout, or when woken by finish()/cancel() with
     * empty deque. Callers that need the terminal signal should combine
     * waitPop() with isFinished()/isCancelled().
     */
    std::optional<StreamChunk> waitPop(std::chrono::milliseconds timeout);

    bool isFinished() const noexcept;
    FinishReason getReason() const noexcept;
    int32_t getOriginalBatchIdx() const noexcept;
    bool isCancelled() const noexcept;

    //! Fire-and-forget cancellation. Safe from any thread. Wakes blocked consume/waitPop.
    void cancel() noexcept;

    //! Per-request emit throttle. Values < 1 are clamped to 1. Default 1.
    void setStreamInterval(int32_t n) noexcept;
    int32_t getStreamInterval() const noexcept;

    //! Filter special tokens (EOS, `<|im_end|>`, `<think>`, vision placeholders,
    //! etc.) out of `chunk.text`. Default `true`. `chunk.tokenIds` is unaffected
    //! either way. Set before the channel is attached.
    void setSkipSpecialTokens(bool skip) noexcept;
    bool getSkipSpecialTokens() const noexcept;

private:
    StreamChannel() = default;
    friend class StreamChannelFinalizer;
    friend void attachStreamChannel(std::shared_ptr<StreamChannel> const&, int32_t);
    friend bool validateStreamingSubmission(LLMGenerationRequest const&);
    friend void applyCancellationToFinishStates(SpecDecodeInferenceContext&);
    friend void emitChunks(SpecDecodeInferenceContext&, tokenizer::Tokenizer const&);

    // ── Producer API (runtime-only) ──────────────────────────────────────────
    void push(StreamChunk chunk);
    void finish(FinishReason reason); //!< Idempotent: first call wins.
    void setOriginalBatchIdx(int32_t idx) noexcept;

    mutable std::mutex mMutex;
    std::condition_variable mCv;
    std::deque<StreamChunk> mPending;

    std::atomic<bool> mFinished{false};
    std::atomic<FinishReason> mReason{FinishReason::kNotFinished};
    std::atomic<bool> mCancelled{false};
    std::atomic<int32_t> mOriginalBatchIdx{-1};
    std::atomic<int32_t> mStreamInterval{1};
    std::atomic<bool> mSkipSpecialTokens{true};

    std::atomic<bool> mAttachedToRequest{false};
};

// Template consume() definition: must be visible in the header.
template <typename Handler>
void StreamChannel::consume(Handler&& handler, std::chrono::milliseconds poll)
{
    std::unique_lock<std::mutex> lk(mMutex);
    while (true)
    {
        mCv.wait_for(lk, poll, [&] {
            return !mPending.empty() || mFinished.load(std::memory_order_acquire)
                || mCancelled.load(std::memory_order_acquire);
        });
        while (!mPending.empty())
        {
            StreamChunk chunk = std::move(mPending.front());
            mPending.pop_front();
            lk.unlock();
            std::forward<Handler>(handler)(std::move(chunk));
            lk.lock();
        }
        bool const done = mFinished.load(std::memory_order_acquire) || mCancelled.load(std::memory_order_acquire);
        if (done && mPending.empty())
        {
            break;
        }
    }
}

/*!
 * @brief Per-slot detokenization and streaming state.
 *
 * Lives in SpecDecodeInferenceContext, compacted in lockstep with tokenIds.
 */
struct SlotStreamState
{
    //! Channel (null ⇒ streaming disabled for this slot).
    std::shared_ptr<StreamChannel> channel;

    //! Count of tokens whose piece bytes have been fed through emitDelta.
    size_t sentTokenCount{0};

    //! sentTokenCount at last push (for streamInterval gating).
    size_t lastEmittedTokenCount{0};

    //! Trailing incomplete UTF-8 bytes carried across iterations.
    std::string pendingBytes;

    //! Terminal reason latched at the moment finishedStates[i] flips to 1.
    //! One writer (the code that flips the state), one reader (the emit hook).
    FinishReason terminalReason{FinishReason::kNotFinished};
};

/*!
 * @brief RAII guard that guarantees every attached StreamChannel terminates.
 *
 * On destruction, for every slot whose channel has not yet been finalized the
 * guard:
 *   1. Tries to build a terminal chunk containing any un-emitted tokens + text
 *      (sanitized + flushed to U+FFFD) with reason=kError and push()es it.
 *   2. Calls finish(kError) — idempotent, no-throw — to guarantee the consumer
 *      unblocks even if (1) failed under OOM.
 */
class StreamChannelFinalizer
{
public:
    StreamChannelFinalizer(SpecDecodeInferenceContext& ctx, tokenizer::Tokenizer const& tok) noexcept;
    ~StreamChannelFinalizer() noexcept;

    StreamChannelFinalizer(StreamChannelFinalizer const&) = delete;
    StreamChannelFinalizer& operator=(StreamChannelFinalizer const&) = delete;
    StreamChannelFinalizer(StreamChannelFinalizer&&) = delete;
    StreamChannelFinalizer& operator=(StreamChannelFinalizer&&) = delete;

private:
    SpecDecodeInferenceContext& mCtx;
    tokenizer::Tokenizer const& mTok;
};

} // namespace rt
} // namespace trt_edgellm
