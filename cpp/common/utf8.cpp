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

#include "common/utf8.h"

#include <cstdint>

namespace trt_edgellm
{
namespace utf8
{

namespace
{
constexpr char kFFFD[] = "\xEF\xBF\xBD"; //!< UTF-8 encoding of U+FFFD REPLACEMENT CHARACTER
constexpr int kFFFDLen = 3;
} // namespace

int leaderByteLen(unsigned char c) noexcept
{
    if ((c & 0x80) == 0x00)
    {
        return 1; // ASCII
    }
    if ((c & 0xE0) == 0xC0)
    {
        return 2;
    }
    if ((c & 0xF0) == 0xE0)
    {
        return 3;
    }
    if ((c & 0xF8) == 0xF0)
    {
        return 4;
    }
    return 0;
}

uint32_t decodeCodepoint(unsigned char const* bytes, int need) noexcept
{
    switch (need)
    {
    case 1: return static_cast<uint32_t>(bytes[0]);
    case 2: return (static_cast<uint32_t>(bytes[0] & 0x1F) << 6) | static_cast<uint32_t>(bytes[1] & 0x3F);
    case 3:
        return (static_cast<uint32_t>(bytes[0] & 0x0F) << 12) | (static_cast<uint32_t>(bytes[1] & 0x3F) << 6)
            | static_cast<uint32_t>(bytes[2] & 0x3F);
    case 4:
        return (static_cast<uint32_t>(bytes[0] & 0x07) << 18) | (static_cast<uint32_t>(bytes[1] & 0x3F) << 12)
            | (static_cast<uint32_t>(bytes[2] & 0x3F) << 6) | static_cast<uint32_t>(bytes[3] & 0x3F);
    default: return 0; // Precondition violated; caller should not reach here.
    }
}

bool isValidCodepointForLen(uint32_t cp, int need) noexcept
{
    bool const overlong = (need == 2 && cp < 0x80) || (need == 3 && cp < 0x800) || (need == 4 && cp < 0x10000);
    bool const surrogate = (cp >= 0xD800 && cp <= 0xDFFF);
    bool const tooBig = (cp > 0x10FFFF);
    return !overlong && !surrogate && !tooBig;
}

std::string sanitizeUtf8Streaming(std::string const& buffer, std::string& pending) noexcept
{
    // Prepend previously-held incomplete bytes (if any) and clear `pending`.
    std::string input = std::move(pending);
    input.append(buffer);
    pending.clear();

    std::string out;
    out.reserve(input.size());

    auto const* bytes = reinterpret_cast<unsigned char const*>(input.data());
    size_t i = 0;
    while (i < input.size())
    {
        unsigned char const c = bytes[i];
        int const need = leaderByteLen(c);
        if (need == 0)
        {
            // Invalid leader or isolated continuation byte.
            out.append(kFFFD, kFFFDLen);
            i += 1;
            continue;
        }
        if (i + static_cast<size_t>(need) > input.size())
        {
            // Trailing incomplete codepoint — hold in pending for next call.
            pending.assign(input, i, input.size() - i);
            break;
        }
        // Validate continuation bytes.
        bool valid = true;
        for (int k = 1; k < need; ++k)
        {
            if ((bytes[i + static_cast<size_t>(k)] & 0xC0) != 0x80)
            {
                valid = false;
                break;
            }
        }
        if (!valid)
        {
            out.append(kFFFD, kFFFDLen);
            i += 1;
            continue;
        }

        uint32_t const cp = decodeCodepoint(bytes + i, need);
        if (!isValidCodepointForLen(cp, need))
        {
            out.append(kFFFD, kFFFDLen);
            i += 1;
            continue;
        }
        out.append(input, i, static_cast<size_t>(need));
        i += static_cast<size_t>(need);
    }
    return out;
}

std::string sanitizeUtf8Flush(std::string& pending) noexcept
{
    std::string out;
    out.reserve(pending.size() * kFFFDLen);
    for (size_t k = 0; k < pending.size(); ++k)
    {
        out.append(kFFFD, kFFFDLen);
    }
    pending.clear();
    return out;
}

} // namespace utf8
} // namespace trt_edgellm
