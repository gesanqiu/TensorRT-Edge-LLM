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

#include <cstdint>
#include <string>

namespace trt_edgellm
{
namespace utf8
{

/*!
 * @brief Length (1–4) of a UTF-8 codepoint starting with this leader byte.
 *
 * Returns 0 if `c` is not a valid leader — either a continuation byte
 * (10xxxxxx), a 5+ byte leader (11111xxx), or other malformed leading bit
 * pattern. Used by both the streaming sanitizer and the tokenizer codepoint
 * decoder.
 */
int leaderByteLen(unsigned char c) noexcept;

/*!
 * @brief Decode a UTF-8 codepoint from `bytes[0..need)`.
 *
 * Preconditions (caller must verify):
 *   (1) `need == leaderByteLen(bytes[0])` and `need > 0`;
 *   (2) every continuation byte matches `(b & 0xC0) == 0x80`.
 *
 * Does NOT validate overlongs, UTF-16 surrogates, or codepoints > U+10FFFF —
 * use `isValidCodepointForLen` for that.
 */
uint32_t decodeCodepoint(unsigned char const* bytes, int need) noexcept;

/*!
 * @brief True iff `cp` is validly encoded as a `need`-byte UTF-8 codepoint.
 *
 * Rejects overlongs for the given length, UTF-16 surrogates (U+D800..U+DFFF),
 * and codepoints > U+10FFFF. `need` must be in [1, 4].
 */
bool isValidCodepointForLen(uint32_t cp, int need) noexcept;

/*!
 * @brief Consume a raw byte buffer and produce a valid UTF-8 string.
 *
 * Scans `buffer` and produces a well-formed UTF-8 output string. Invalid byte
 * sequences (isolated continuation bytes, overlong encodings, surrogates,
 * codepoints > U+10FFFF, bogus leaders) are replaced with the Unicode
 * replacement character U+FFFD ("\xEF\xBF\xBD").
 *
 * If the buffer ends mid-codepoint (valid leader but insufficient continuation
 * bytes), the trailing incomplete bytes are moved into `pending` for reuse on
 * the next call and NOT emitted. This is the only case in which bytes are held.
 *
 * `pending` is an in-out buffer: existing content is prepended to `buffer` at
 * the start of each call, and replaced with the new trailing incomplete bytes
 * (if any) on return.
 *
 * Output always equals input in terms of Unicode codepoints modulo:
 *   - trailing incomplete bytes (moved to `pending`)
 *   - invalid byte sequences (replaced with U+FFFD)
 *
 * @param buffer  Input bytes to sanitize.
 * @param pending In-out buffer: leftover incomplete bytes from a previous call
 *                are prepended to `buffer`; new trailing incomplete bytes (if
 *                any) are written back to `pending` on return.
 * @return Well-formed UTF-8 string with invalid byte sequences replaced.
 */
std::string sanitizeUtf8Streaming(std::string const& buffer, std::string& pending) noexcept;

/*!
 * @brief Final-flush variant.
 *
 * Emits all of `pending` as U+FFFD replacement characters (one per held byte)
 * and clears `pending`. Used when the slot terminates with bytes still
 * in-flight (e.g., model emitted EOS mid-codepoint) or for single-shot decode
 * paths that have no further input to arrive.
 *
 * @param pending In-out buffer: cleared on return.
 * @return String of U+FFFD codepoints, one per byte previously held in pending.
 */
std::string sanitizeUtf8Flush(std::string& pending) noexcept;

} // namespace utf8
} // namespace trt_edgellm
