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

#include <collie/simd/simd.h>
#include <type_traits>

namespace polaris {

    template<typename T, typename Arch = collie::simd::default_arch>
    struct NormL2 {
        static float norm_l2_sqr(const T *x, size_t d) {
            float sum = 0.0f;
            for (uint32_t i = 0; i < d; i++) {
                sum += x[i] * x[i];
            }
            return sum;
        }

        static float norm_l2(const T *x, size_t d) {
            return std::sqrt(norm_l2_sqr(x, d));
        }
    };

    template<typename Arch>
    struct NormL2<float, Arch> {
        static float norm_l2_sqr(const float *x, size_t d) {
            using b_type = collie::simd::batch<float, Arch>;
            using aligend_tag = typename std::conditional<collie::simd::is_aligned<Arch>(x),
                    collie::simd::aligned_mode, collie::simd::unaligned_mode>::type;
            std::size_t inc = b_type::size;
            std::size_t vec_size = d - d % inc;
            float sum = 0.0;
            b_type sum_vec = collie::simd::broadcast(0.0f);
            for (std::size_t i = 0; i < vec_size; i += inc) {
                b_type xvec = b_type::load(x + i, aligend_tag());
                sum_vec += xvec * xvec;
            }
            sum = collie::simd::reduce_add(sum_vec);
            for (std::size_t i = vec_size; i < d; ++i) {
                sum += x[i] * x[i];
            }
            return sum;
        }

        static float norm_l2(const float *x, size_t d) {
            return norm_l2_sqr(x, d);
        }
    };

    template<typename T, typename Arch = collie::simd::default_arch>
    struct NormalizerL2 {
        static void normalize(T *x, size_t d) {
            float norm = NormL2<T, Arch>::norm_l2(x, d);
            for (uint32_t i = 0; i < d; i++) {
                x[i] /= norm;
            }
        }
    };

    template<typename Arch>
    struct NormalizerL2<float, Arch> {
        static void normalize(float *x, size_t d) {
            float norm = NormL2<float, Arch>::norm_l2(x, d);
            for (uint32_t i = 0; i < d; i++) {
                x[i] /= norm;
            }
        }

        static void normalize(float *x, size_t d, float norm) {
            for (uint32_t i = 0; i < d; i++) {
                x[i] /= norm;
            }
        }
    };

}  // namespace polars
