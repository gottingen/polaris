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
#include <polaris/core/common.h>
#include <immintrin.h>
#include <polaris/utility/polaris_exception.h>

namespace polaris {

    class MemoryCache {
    public:
        inline static void prefetch(unsigned char *ptr, const size_t byteSizeOfObject) {
            switch ((byteSizeOfObject - 1) >> 6) {
                default:
                case 28:
                    _mm_prefetch(ptr, _MM_HINT_T0);
                    ptr += 64;
                case 27:
                    _mm_prefetch(ptr, _MM_HINT_T0);
                    ptr += 64;
                case 26:
                    _mm_prefetch(ptr, _MM_HINT_T0);
                    ptr += 64;
                case 25:
                    _mm_prefetch(ptr, _MM_HINT_T0);
                    ptr += 64;
                case 24:
                    _mm_prefetch(ptr, _MM_HINT_T0);
                    ptr += 64;
                case 23:
                    _mm_prefetch(ptr, _MM_HINT_T0);
                    ptr += 64;
                case 22:
                    _mm_prefetch(ptr, _MM_HINT_T0);
                    ptr += 64;
                case 21:
                    _mm_prefetch(ptr, _MM_HINT_T0);
                    ptr += 64;
                case 20:
                    _mm_prefetch(ptr, _MM_HINT_T0);
                    ptr += 64;
                case 19:
                    _mm_prefetch(ptr, _MM_HINT_T0);
                    ptr += 64;
                case 18:
                    _mm_prefetch(ptr, _MM_HINT_T0);
                    ptr += 64;
                case 17:
                    _mm_prefetch(ptr, _MM_HINT_T0);
                    ptr += 64;
                case 16:
                    _mm_prefetch(ptr, _MM_HINT_T0);
                    ptr += 64;
                case 15:
                    _mm_prefetch(ptr, _MM_HINT_T0);
                    ptr += 64;
                case 14:
                    _mm_prefetch(ptr, _MM_HINT_T0);
                    ptr += 64;
                case 13:
                    _mm_prefetch(ptr, _MM_HINT_T0);
                    ptr += 64;
                case 12:
                    _mm_prefetch(ptr, _MM_HINT_T0);
                    ptr += 64;
                case 11:
                    _mm_prefetch(ptr, _MM_HINT_T0);
                    ptr += 64;
                case 10:
                    _mm_prefetch(ptr, _MM_HINT_T0);
                    ptr += 64;
                case 9:
                    _mm_prefetch(ptr, _MM_HINT_T0);
                    ptr += 64;
                case 8:
                    _mm_prefetch(ptr, _MM_HINT_T0);
                    ptr += 64;
                case 7:
                    _mm_prefetch(ptr, _MM_HINT_T0);
                    ptr += 64;
                case 6:
                    _mm_prefetch(ptr, _MM_HINT_T0);
                    ptr += 64;
                case 5:
                    _mm_prefetch(ptr, _MM_HINT_T0);
                    ptr += 64;
                case 4:
                    _mm_prefetch(ptr, _MM_HINT_T0);
                    ptr += 64;
                case 3:
                    _mm_prefetch(ptr, _MM_HINT_T0);
                    ptr += 64;
                case 2:
                    _mm_prefetch(ptr, _MM_HINT_T0);
                    ptr += 64;
                case 1:
                    _mm_prefetch(ptr, _MM_HINT_T0);
                    ptr += 64;
                case 0:
                    _mm_prefetch(ptr, _MM_HINT_T0);
                    ptr += 64;
                    break;
            }
        }

        inline static void *alignedAlloc(const size_t allocSize) {
#if defined(NGT_AVX512)
            size_t alignment = 64;
            uint64_t mask = 0xFFFFFFFFFFFFFFC0;
#elif defined(NGT_AVX2)
            size_t alignment = 32;
            uint64_t mask = 0xFFFFFFFFFFFFFFE0;
#else
            size_t alignment = 16;
            uint64_t mask = 0xFFFFFFFFFFFFFFF0;
#endif
            uint8_t *p = new uint8_t[allocSize + alignment];
            uint8_t *ptr = p + alignment;
            ptr = reinterpret_cast<uint8_t *>((reinterpret_cast<uint64_t>(ptr) & mask));
            *p++ = 0xAB;
            while (p != ptr) *p++ = 0xCD;
            return ptr;
        }

        inline static void alignedFree(void *ptr) {
            uint8_t *p = static_cast<uint8_t *>(ptr);
            p--;
            while (*p == 0xCD) p--;
            if (*p != 0xAB) {
                POLARIS_THROW_EX("MemoryCache::alignedFree: Fatal Error! Cannot find allocated address.");
            }
            delete[] p;
        }
    };

    inline void print_error_and_terminate(std::stringstream &error_stream) {
        std::cerr << error_stream.str() << std::endl;
        throw polaris::PolarisException(error_stream.str(), -1, __PRETTY_FUNCTION__, __FILE__, __LINE__);
    }


    inline void report_memory_allocation_failure() {
        std::stringstream stream;
        stream << "Memory Allocation Failed.";
        print_error_and_terminate(stream);
    }

    inline void report_misalignment_of_requested_size(size_t align) {
        std::stringstream stream;
        stream << "Requested memory size is not a multiple of " << align << ". Can not be allocated.";
        print_error_and_terminate(stream);
    }

    inline void alloc_aligned(void **ptr, size_t size, size_t align) {
        *ptr = nullptr;
        if (IS_ALIGNED(size, align) == 0)
            report_misalignment_of_requested_size(align);
#ifndef _WINDOWS
        *ptr = ::aligned_alloc(align, size);
#else
        *ptr = ::_aligned_malloc(size, align); // note the swapped arguments!
#endif
        if (*ptr == nullptr)
            report_memory_allocation_failure();
    }

    inline void aligned_free(void *ptr) {
        // Gopal. Must have a check here if the pointer was actually allocated by
        // _alloc_aligned
        if (ptr == nullptr) {
            return;
        }
#ifndef _WINDOWS
        free(ptr);
#else
        ::_aligned_free(ptr);
#endif
    }

    inline void realloc_aligned(void **ptr, size_t size, size_t align) {
        if (IS_ALIGNED(size, align) == 0)
            report_misalignment_of_requested_size(align);
#ifdef _WINDOWS
        *ptr = ::_aligned_realloc(*ptr, size, align);
#else
        std::cerr << "No aligned realloc on GCC. Must malloc and mem_align, "
                         "left it out for now."
                      << std::endl;
#endif
        if (*ptr == nullptr)
            report_memory_allocation_failure();
    }
    // NOTE :: good efficiency when total_vec_size is integral multiple of 64
    inline void prefetch_vector(const char *vec, size_t vecsize) {
        size_t max_prefetch_size = (vecsize / 64) * 64;
        for (size_t d = 0; d < max_prefetch_size; d += 64)
            _mm_prefetch((const char *) vec + d, _MM_HINT_T0);
    }

    // NOTE :: good efficiency when total_vec_size is integral multiple of 64
    inline void prefetch_vector_l2(const char *vec, size_t vecsize) {
        size_t max_prefetch_size = (vecsize / 64) * 64;
        for (size_t d = 0; d < max_prefetch_size; d += 64)
            _mm_prefetch((const char *) vec + d, _MM_HINT_T1);
    }
}  // namespace polaris

