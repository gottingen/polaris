// Copyright 2023 The Elastic-AI Authors.
// part of Elastic AI Search
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
//
#pragma once

#include <cstdint>

namespace polaris {
    
    class bfloat16 {
    public:
        bfloat16(float fp32) {
            value = fp32_to_bf16(fp32);
        }
        explicit operator float() const { return bf16_to_fp32(value); }
        explicit operator double() const { return static_cast<double>(bf16_to_fp32(value)); }
        static uint16_t fp32_to_bf16(const float fp32) {
            union {
                float fp;
                uint32_t ui;
            } b32;
            b32.fp = fp32;
            if ((b32.ui & 0xffff) > 0x7fff) {
                b32.ui = (b32.ui >> 16) | 0x1;
            } else {
                b32.ui = b32.ui >> 16;
            }
            return static_cast<uint16_t>(b32.ui & 0xffff);
        }

        static float bf16_to_fp32(const uint16_t bf16) {
            union {
                float fp;
                uint32_t ui;
            } b32;
            b32.ui = bf16;
            b32.ui <<= 16;
            b32.ui |= 0x7fff;
            return b32.fp;
        }

        uint16_t value;
    };
}  // namespace polaris
