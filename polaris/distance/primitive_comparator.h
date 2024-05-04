//
// Copyright (C) 2015 Yahoo Japan Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//

#pragma once

#include <polaris/core/defines.h>
#include <polaris/distance/primitive.h>
#include <immintrin.h>

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

    class PrimitiveComparator {
    public:

        static double absolute(double v) { return fabs(v); }

        static int absolute(int v) { return abs(v); }

        class L1Uint8 {
        public:
            inline static double compare(const void *a, const void *b, size_t size) {
                return polaris::primitive::compare_l1((const uint8_t *) a, (const uint8_t *) b, size);
            }
        };

        class L2Uint8 {
        public:
            inline static double compare(const void *a, const void *b, size_t size) {
                return polaris::primitive::compare_l2((const uint8_t *) a, (const uint8_t *) b, size);
            }
        };

        class HammingUint8 {
        public:
            inline static double compare(const void *a, const void *b, size_t size) {
                return polaris::primitive::compare_hamming_distance((const uint8_t *) a, (const uint8_t *) b, size);
            }
        };

        class JaccardUint8 {
        public:
            inline static double compare(const void *a, const void *b, size_t size) {
                return polaris::primitive::compare_jaccard_distance((const uint8_t *) a, (const uint8_t *) b, size);
            }
        };

        class SparseJaccardFloat {
        public:
            inline static double compare(const void *a, const void *b, size_t size) {
                return polaris::primitive::compare_sparse_jaccard_distance((const float *) a, (const float *) b, size);
            }
        };

        class L2Float {
        public:
            inline static double compare(const void *a, const void *b, size_t size) {
                return polaris::primitive::compare_l2((const float *) a, (const float *) b, size);
            }
        };

        class NormalizedL2Float {
        public:
            inline static double compare(const void *a, const void *b, size_t size) {
                return polaris::primitive::compare_normalized_l2((const float *) a, (const float *) b, size);
            }
        };

        class L1Float {
        public:
            inline static double compare(const void *a, const void *b, size_t size) {
                return polaris::primitive::compare_l1((const float *) a, (const float *) b, size);
            }
        };

        class CosineSimilarityFloat {
        public:
            inline static double compare(const void *a, const void *b, size_t size) {
                return polaris::primitive::compare_cosine_similarity((const float *) a, (const float *) b, size);
            }
        };

        class NormalizedCosineSimilarityFloat {
        public:
            inline static double compare(const void *a, const void *b, size_t size) {
                return polaris::primitive::compare_normalized_cosine_similarity((const float *) a, (const float *) b,
                                                                              size);
            }
        };

        class AngleFloat {
        public:
            inline static double compare(const void *a, const void *b, size_t size) {
                return polaris::primitive::compare_angle_distance((const float *) a, (const float *) b, size);
            }
        };

        class NormalizedAngleFloat {
        public:
            inline static double compare(const void *a, const void *b, size_t size) {
                return polaris::primitive::compare_normalized_angle_distance((const float *) a, (const float *) b, size);
            }
        };

        class PoincareFloat {
        public:
            inline static double compare(const void *a, const void *b, size_t size) {
                return polaris::primitive::compare_poincare_distance((const float *) a, (const float *) b, size);
            }
        };

        class LorentzFloat {
        public:
            inline static double compare(const void *a, const void *b, size_t size) {
                return polaris::primitive::compare_lorentz_distance((const float *) a, (const float *) b, size);
            }
        };

        class SparseJaccardFloat16 {
        public:
            inline static double compare(const void *a, const void *b, size_t size) {
                return polaris::primitive::compare_sparse_jaccard_distance((const float16 *) a, (const float16 *) b,
                                                                         size);
            }
        };

        class L2Float16 {
        public:
            inline static double compare(const void *a, const void *b, size_t size) {
                return polaris::primitive::compare_l2((const float16 *) a, (const float16 *) b, size);
            }
        };

        class NormalizedL2Float16 {
        public:
            inline static double compare(const void *a, const void *b, size_t size) {
                return polaris::primitive::compare_normalized_l2((const float16 *) a, (const float16 *) b, size);
            }
        };

        class L1Float16 {
        public:
            inline static double compare(const void *a, const void *b, size_t size) {
                return polaris::primitive::compare_l1((const float16 *) a, (const float16 *) b, size);
            }
        };

        class CosineSimilarityFloat16 {
        public:
            inline static double compare(const void *a, const void *b, size_t size) {
                return polaris::primitive::compare_cosine_similarity((const float16 *) a, (const float16 *) b, size);
            }
        };

        class NormalizedCosineSimilarityFloat16 {
        public:
            inline static double compare(const void *a, const void *b, size_t size) {
                return polaris::primitive::compare_normalized_cosine_similarity((const float16 *) a, (const float16 *) b,
                                                                              size);
            }
        };

        class AngleFloat16 {
        public:
            inline static double compare(const void *a, const void *b, size_t size) {
                return polaris::primitive::compare_angle_distance((const float16 *) a, (const float16 *) b, size);
            }
        };

        class NormalizedAngleFloat16 {
        public:
            inline static double compare(const void *a, const void *b, size_t size) {
                return polaris::primitive::compare_normalized_angle_distance((const float16 *) a, (const float16 *) b,
                                                                           size);
            }
        };

        class PoincareFloat16 {
        public:
            inline static double compare(const void *a, const void *b, size_t size) {
                return polaris::primitive::compare_poincare_distance((const float16 *) a, (const float16 *) b, size);
            }
        };

        class LorentzFloat16 {
        public:
            inline static double compare(const void *a, const void *b, size_t size) {
                return polaris::primitive::compare_lorentz_distance((const float16 *) a, (const float16 *) b, size);
            }
        };

        class SparseJaccardBfloat16 {
        public:
            inline static double compare(const void *a, const void *b, size_t size) {
                POLARIS_THROW_EX("Not supported.");
            }
        };
/*
        class L2Bfloat16 {
        public:
            inline static double compare(const void *a, const void *b, size_t size) {
                return polaris::primitive::compare_l2((const bfloat16 *) a, (const bfloat16 *) b, size);
            }
        };

        class NormalizedL2Bfloat16 {
        public:
            inline static double compare(const void *a, const void *b, size_t size) {
                return PrimitiveComparator::compare_normalized_l2((const bfloat16 *) a, (const bfloat16 *) b, size);
            }
        };
*/
        class L1Bfloat16 {
        public:
            inline static double compare(const void *a, const void *b, size_t size) {
                POLARIS_THROW_EX("Not supported.");
            }
        };

        class CosineSimilarityBfloat16 {
        public:
            inline static double compare(const void *a, const void *b, size_t size) {
                POLARIS_THROW_EX("Not supported.");
            }
        };

        class NormalizedCosineSimilarityBfloat16 {
        public:
            inline static double compare(const void *a, const void *b, size_t size) {
                POLARIS_THROW_EX("Not supported.");
            }
        };

        class AngleBfloat16 {
        public:
            inline static double compare(const void *a, const void *b, size_t size) {
                POLARIS_THROW_EX("Not supported.");
            }
        };

        class NormalizedAngleBfloat16 {
        public:
            inline static double compare(const void *a, const void *b, size_t size) {
                POLARIS_THROW_EX("Not supported.");
            }
        };

        // added by Nyapicom
        class PoincareBfloat16 {
        public:
            inline static double compare(const void *a, const void *b, size_t size) {
                POLARIS_THROW_EX("Not supported.");
            }
        };

        // added by Nyapicom
        class LorentzBfloat16 {
        public:
            inline static double compare(const void *a, const void *b, size_t size) {
                POLARIS_THROW_EX("Not supported.");
            }
        };


    };


} // namespace polaris

