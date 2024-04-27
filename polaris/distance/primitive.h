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

#pragma once

#include <collie/simd/simd.h>
#include <type_traits>

namespace polars {

    /// norm L2 squared of a float vector
    /// formula: sum(x[i] * x[i]) for i in [0, d)
    template <typename Arch = collie::simd::default_arch>
    float fv_norm_l2_sqr(const float* x, size_t d) {
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

    /// formula: sqrt(sum(x[i] * x[i]) for i in [0, d))
    template <typename Arch = collie::simd::default_arch>
    float fv_norm_l2(const float* x, size_t d) {
        return std::sqrt(fv_norm_l2_sqr<Arch>(x, d));
    }

    /// L2 squared distance between two float vectors
    /// formula: sum((x[i] - y[i]) * (x[i] - y[i])) for i in [0, d)
    template <typename Arch = collie::simd::default_arch>
    float fv_l2_sqr(const float* x, const float* y, size_t d) {
        using b_type = collie::simd::batch<float, Arch>;
        using aligend_tag = typename std::conditional<collie::simd::is_aligned<Arch>(x),
                collie::simd::aligned_mode, collie::simd::unaligned_mode>::type;
        std::size_t inc = b_type::size;
        std::size_t vec_size = d - d % inc;
        float sum = 0.0;
        b_type sum_vec = collie::simd::broadcast(0.0f);
        for (std::size_t i = 0; i < vec_size; i += inc) {
            b_type xvec = b_type::load(x + i, aligend_tag());
            b_type yvec = b_type::load(y + i, aligend_tag());
             auto  delta_vec = xvec - yvec;
            sum_vec +=  delta_vec * delta_vec;
        }
        sum = collie::simd::reduce_add(sum_vec);
        for (std::size_t i = vec_size; i < d; ++i) {
            auto delta = x[i] - y[i];
            sum += delta * delta;
        }
        return sum;
    }
    /// inner product between two float vectors
    /// formula: sum(x[i] * y[i]) for i in [0, d)
    template <typename Arch = collie::simd::default_arch>
    float fv_inner_products(const float* x, const float* y, size_t d) {
        using b_type = collie::simd::batch<float, Arch>;
        using aligend_tag = typename std::conditional<collie::simd::is_aligned<Arch>(x),
                collie::simd::aligned_mode, collie::simd::unaligned_mode>::type;
        std::size_t inc = b_type::size;
        std::size_t vec_size = d - d % inc;
        float sum = 0.0;
        b_type sum_vec = collie::simd::broadcast(0.0f);
        for (std::size_t i = 0; i < vec_size; i += inc) {
            b_type xvec = b_type::load(x + i, aligend_tag());
            b_type yvec = b_type::load(y + i, aligend_tag());
            sum_vec +=  xvec * yvec;
        }
        sum = collie::simd::reduce_add(sum_vec);
        for (std::size_t i = vec_size; i < d; ++i) {
            sum +=  x[i] * y[i];
        }
        return sum;
    }

    /// L1 distance between two float vectors
    /// formula: sum(abs(x[i] - y[i])) for i in [0, d)
    template <typename Arch = collie::simd::default_arch>
    float fv_l1(const float* x, const float* y, size_t d) {
        using b_type = collie::simd::batch<float, Arch>;
        using aligend_tag = typename std::conditional<collie::simd::is_aligned<Arch>(x),
                collie::simd::aligned_mode, collie::simd::unaligned_mode>::type;
        std::size_t inc = b_type::size;
        std::size_t vec_size = d - d % inc;
        float sum = 0.0;
        b_type sum_vec = collie::simd::broadcast(0.0f);
        for (std::size_t i = 0; i < vec_size; i += inc) {
            b_type xvec = b_type::load(x + i, aligend_tag());
            b_type yvec = b_type::load(y + i, aligend_tag());
            sum_vec +=  collie::simd::abs(xvec - yvec);
        }
        sum = collie::simd::reduce_add(sum_vec);
        for (std::size_t i = vec_size; i < d; ++i) {
            sum +=  std::abs(x[i] - y[i]);
        }
        return sum;
    }
}  // namespace polars

