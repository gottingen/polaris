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

    template<typename T>
    struct simd_type_promotion {
        using type = T;
    };

    template<>
    struct simd_type_promotion<int8_t> {
        using type = int32_t;
    };

    template<>
    struct simd_type_promotion<uint8_t> {
        using type = uint32_t;
    };

    template<>
    struct simd_type_promotion<int16_t> {
        using type = int32_t;
    };

    template<>
    struct simd_type_promotion<uint16_t> {
        using type = uint32_t;
    };

    template<typename T>
    using simd_type_promotion_t = typename simd_type_promotion<T>::type;

    template<typename T, typename Arch, typename Tag, bool>
    struct NormL2Impl;

    template<typename T, typename Arch, typename Tag>
    struct NormL2Impl<T, Arch, Tag, true> {
        static float norm_l2_sqr(const T *x, size_t d) {
            using b_type = collie::simd::batch<T, Arch>;
            std::size_t inc = b_type::size;
            std::size_t vec_size = d - d % inc;
            float sum = 0.0;
            b_type sum_vec = collie::simd::broadcast(0.0f);
            for (std::size_t i = 0; i < vec_size; i += inc) {
                b_type xvec = b_type::load(x + i, Tag());
                sum_vec += xvec * xvec;
            }
            sum = collie::simd::reduce_add(sum_vec);
            for (std::size_t i = vec_size; i < d; ++i) {
                sum += x[i] * x[i];
            }
            return sum;
        }

        static float norm_l2(const T *x, size_t d) {
            return norm_l2_sqr(x, d);
        }
    };

    template<typename T, typename Arch, typename Tag>
    struct NormL2Impl<T, Arch, Tag, false> {
        static float norm_l2_sqr(const T *x, size_t d) {
            using U = simd_type_promotion_t<T>;
            using b_type = collie::simd::batch<U, Arch>;
            using index_type = typename collie::simd::as_integer_t<b_type>;
            const index_type index = collie::simd::detail::make_sequence_as_batch<index_type>();
            std::size_t inc = b_type::size;
            std::size_t vec_size = d - d % inc;
            float sum = 0.0;
            b_type sum_vec = collie::simd::broadcast(U(0));
            for (std::size_t i = 0; i < vec_size; i += inc) {
                b_type xvec =b_type::gather(x + i, index);
                sum_vec += xvec * xvec;
            }
            sum = collie::simd::reduce_add(sum_vec);
            for (std::size_t i = vec_size; i < d; ++i) {
                sum += x[i] * x[i];
            }
            return sum;
        }

        static float norm_l2(const T *x, size_t d) {
            return norm_l2_sqr(x, d);
        }
    };



    template<typename T, typename Arch, typename Tag>
    struct NormL2 {
        static float norm_l2_sqr(const T *x, size_t d) {
            return NormL2Impl<T, Arch, Tag, std::is_same_v<T, simd_type_promotion_t<T>>>::norm_l2_sqr(x, d);
        }

        static float norm_l2(const T *x, size_t d) {
            return norm_l2_sqr(x, d);
        }
    };

}  // namespace polars
