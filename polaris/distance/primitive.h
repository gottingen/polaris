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
#include <polaris/internal/polaris_assert.h>
#include <type_traits>

namespace polaris {

    template<typename T, typename Arch, typename Tag>
    float PrimL1(const T* x, const T* y, size_t d) {
        using b_type = collie::simd::batch<T,Arch>;
        std::size_t inc = b_type::size;
        std::size_t vec_size = d - d % inc;
        float sum = 0.0;
        b_type sum_vec = collie::simd::broadcast(T(0));
        for (std::size_t i = 0; i < vec_size; i += inc) {
            b_type xvec = b_type::load(x + i, Tag());
            b_type yvec = b_type::load(y + i, Tag());
            sum_vec +=  collie::simd::abs(xvec - yvec);
        }
        sum = collie::simd::reduce_add(sum_vec);
        for (std::size_t i = vec_size; i < d; ++i) {
            sum +=  std::abs(x[i] - y[i]);
        }
        return sum;
    }

    template<typename T, typename Arch, typename Tag>
    float PrimIP(const T* x, const T* y, size_t d) {
        using b_type = collie::simd::batch<T,Arch>;
        std::size_t inc = b_type::size;
        std::size_t vec_size = d - d % inc;
        float sum = 0.0;
        b_type sum_vec = collie::simd::broadcast(T(0));
        for (std::size_t i = 0; i < vec_size; i += inc) {
            b_type xvec = b_type::load(x + i, Tag());
            b_type yvec = b_type::load(y + i, Tag());
            sum_vec +=  xvec * yvec;
        }
        sum = collie::simd::reduce_add(sum_vec);
        for (std::size_t i = vec_size; i < d; ++i) {
            sum +=  x[i] * y[i];
        }
        return sum;
    }

    template<typename T, typename Arch, typename Tag>
    float PrimL2Sqr(const T* x, const T* y, size_t d) {
        using b_type = collie::simd::batch<T,Arch>;
        std::size_t inc = b_type::size;
        std::size_t vec_size = d - d % inc;
        float sum = 0.0;
        b_type sum_vec = collie::simd::broadcast(T(0));
        for (std::size_t i = 0; i < vec_size; i += inc) {
            b_type xvec = b_type::load(x + i, Tag());
            b_type yvec = b_type::load(y + i, Tag());
            auto delta = xvec - yvec;
            sum_vec +=  delta * delta;
        }
        sum = collie::simd::reduce_add(sum_vec);
        for (std::size_t i = vec_size; i < d; ++i) {
            auto delta = x[i] - y[i];
            sum +=  delta * delta;
        }
        return sum;
    }


    template<typename T, typename Arch, typename Tag>
    float PrimLinf(const T* x, const T* y, size_t d) {
        using b_type = collie::simd::batch<T,Arch>;
        std::size_t inc = b_type::size;
        std::size_t vec_size = d - d % inc;
        float sum = 0.0;
        b_type sum_vec = collie::simd::broadcast(T(0));
        for (std::size_t i = 0; i < vec_size; i += inc) {
            b_type xvec = b_type::load(x + i, Tag());
            b_type yvec = b_type::load(y + i, Tag());
            sum_vec =  collie::simd::max(sum_vec,collie::simd::abs(xvec - yvec));
        }
        sum = collie::simd::reduce_max(sum_vec);
        for (std::size_t i = vec_size; i < d; ++i) {
            sum =  std::max(sum, std::abs(x[i] - y[i]));
        }
        return sum;
    }

    template<typename T, typename Arch, typename Tag>
    void PrimAdd(size_t d, const T* x, const T* y, T* z) {
        using b_type = collie::simd::batch<T,Arch>;
        std::size_t inc = b_type::size;
        std::size_t vec_size = d - d % inc;
        for (std::size_t i = 0; i < vec_size; i += inc) {
            b_type xvec = b_type::load(x + i, Tag());
            b_type yvec = b_type::load(y + i, Tag());
            collie::simd::store(z + i, xvec + yvec, Tag());
        }
        for (std::size_t i = vec_size; i < d; ++i) {
            z[i] = x[i] + y[i];
        }
    }

    template<typename T, typename Arch, typename Tag>
    void PrimAdd(size_t d, const T* x, const T y, T* z) {
        using b_type = collie::simd::batch<T,Arch>;
        std::size_t inc = b_type::size;
        std::size_t vec_size = d - d % inc;
        b_type yvec = collie::simd::broadcast(y);
        for (std::size_t i = 0; i < vec_size; i += inc) {
            b_type xvec = b_type::load(x + i, Tag());
            collie::simd::store(z + i, xvec + yvec, Tag());
        }
        for (std::size_t i = vec_size; i < d; ++i) {
            z[i] = x[i] + y;
        }
    }

    template<typename T, typename Arch, typename Tag>
    void PrimMAdd(size_t d, const T* x, const T by, const T*y,  T* z) {
        using b_type = collie::simd::batch<T,Arch>;
        std::size_t inc = b_type::size;
        std::size_t vec_size = d - d % inc;
        b_type byvec = collie::simd::broadcast(by);
        for (std::size_t i = 0; i < vec_size; i += inc) {
            b_type xvec = b_type::load(x + i, Tag());
            b_type yvec = b_type::load(y + i, Tag());
            collie::simd::store(z + i, xvec + yvec * byvec, Tag());
        }
        for (std::size_t i = vec_size; i < d; ++i) {
            z[i] = x[i] + y[i] * by;
        }
    }

    template<typename T, typename Arch, typename Tag>
    void PrimSub(size_t d, const T* x, const T* y, T* z) {
        using b_type = collie::simd::batch<T,Arch>;
        std::size_t inc = b_type::size;
        std::size_t vec_size = d - d % inc;
        for (std::size_t i = 0; i < vec_size; i += inc) {
            b_type xvec = b_type::load(x + i, Tag());
            b_type yvec = b_type::load(y + i, Tag());
            collie::simd::store(z + i, xvec - yvec, Tag());
        }
        for (std::size_t i = vec_size; i < d; ++i) {
            z[i] = x[i] - y[i];
        }
    }

    template<typename T, typename Arch, typename Tag>
    void PrimSub(size_t d, const T* x, const T y, T* z) {
        using b_type = collie::simd::batch<T,Arch>;
        std::size_t inc = b_type::size;
        std::size_t vec_size = d - d % inc;
        b_type yvec = collie::simd::broadcast(y);
        for (std::size_t i = 0; i < vec_size; i += inc) {
            b_type xvec = b_type::load(x + i, Tag());
            collie::simd::store(z + i, xvec - yvec, Tag());
        }
        for (std::size_t i = vec_size; i < d; ++i) {
            z[i] = x[i] - y;
        }
    }
}  // namespace polaris

