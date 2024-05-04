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
#include <polaris/utility/polaris_exception.h>
#include <immintrin.h>

namespace polaris {

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

