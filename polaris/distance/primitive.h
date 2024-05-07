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
#include <polaris/utility/polaris_assert.h>
#include <polaris/core/common.h>
#include <type_traits>
#include <cmath>

namespace polaris {

    template<typename T, typename Arch, typename Tag>
    float PrimLinf(const T *x, const T *y, size_t d) {
        using b_type = collie::simd::batch<T, Arch>;
        std::size_t inc = b_type::size;
        std::size_t vec_size = d - d % inc;
        float sum = 0.0;
        b_type sum_vec = collie::simd::broadcast(T(0));
        for (std::size_t i = 0; i < vec_size; i += inc) {
            b_type xvec = b_type::load(x + i, Tag());
            b_type yvec = b_type::load(y + i, Tag());
            sum_vec = collie::simd::max(sum_vec, collie::simd::abs(xvec - yvec));
        }
        sum = collie::simd::reduce_max(sum_vec);
        for (std::size_t i = vec_size; i < d; ++i) {
            sum = std::max(sum, std::abs(x[i] - y[i]));
        }
        return sum;
    }

    template<typename T, typename Arch, typename Tag>
    void PrimAdd(size_t d, const T *x, const T *y, T *z) {
        using b_type = collie::simd::batch<T, Arch>;
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
    void PrimAdd(size_t d, const T *x, const T y, T *z) {
        using b_type = collie::simd::batch<T, Arch>;
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
    void PrimMAdd(size_t d, const T *x, const T by, const T *y, T *z) {
        using b_type = collie::simd::batch<T, Arch>;
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
    void PrimSub(size_t d, const T *x, const T *y, T *z) {
        using b_type = collie::simd::batch<T, Arch>;
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
    void PrimSub(size_t d, const T *x, const T y, T *z) {
        using b_type = collie::simd::batch<T, Arch>;
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

namespace polaris::primitive {

    /// unsigned type has a bug for overflow
    /// eg. uint32_t a = 0, b = 2; abs(a - b) = 4294967294
    /// but we expect abs(a-b) =2
    template<typename T>
    struct unsigned_overflow : public std::false_type {};
    template<>
    struct unsigned_overflow<uint8_t> : public std::true_type {};
    template<>
    struct unsigned_overflow<uint16_t> : public std::true_type {};
    template<>
    struct unsigned_overflow<uint32_t> : public std::true_type {};
    template<>
    struct unsigned_overflow<uint64_t> : public std::true_type {};

    /// uint8_t, uint16_t, int8_t, int16_t need to be promoted to float
    /// these type take 1 or 2 bytes, which is not enough to store the result of the multiplication
    template<typename T>
    struct need_promotion : public std::false_type {};
    template<>
    struct need_promotion<int8_t> : public std::true_type {};
    template<>
    struct need_promotion<uint8_t> : public std::true_type {};
    template<>
    struct need_promotion<int16_t> : public std::true_type {};
    template<>
    struct need_promotion<uint16_t> : public std::true_type {};

    /// l2
    template <typename T>
    float compare_simple_l2_sqrd(const T *__restrict x, const T *__restrict y, size_t d) {
        float sum = 0.0;
        for (size_t i = 0; i < d; ++i) {
            auto delta = x[i] - y[i];
            sum += delta * delta;
        }
        return sum;
    }

    template <typename T>
    float compare_simple_l2(const T *__restrict x, const T *__restrict y, size_t d) {
        return std::sqrt(compare_simple_l2_sqrd(x, y, d));
    }

    template<typename T, typename Arch, typename Tag, typename std::enable_if<unsigned_overflow<T>::value && !need_promotion<T>::value, int>::type =0>
    float compare_template_l2_sqr(const T *__restrict x, const T *__restrict y, size_t d) {
        using b_type = collie::simd::batch<T, Arch>;
        std::size_t inc = b_type::size;
        std::size_t vec_size = d - d % inc;
        float sum = 0.0;
        b_type sum_vec = collie::simd::broadcast(T(0));
        for (std::size_t i = 0; i < vec_size; i += inc) {
            b_type xvec = b_type::load(x + i, Tag());
            b_type yvec = b_type::load(y + i, Tag());
            auto max = collie::simd::max(xvec, yvec);
            auto min = collie::simd::min(xvec, yvec);
            auto delta = collie::simd::sub(max, min);
            sum_vec += collie::simd::mul(delta, delta);
        }
        sum = collie::simd::reduce_add(sum_vec);
        for (std::size_t i = vec_size; i < d; ++i) {
            auto delta = x[i] >= y[i] ? x[i] - y[i] : y[i] - x[i];
            sum += delta * delta;
        }
        return sum;
    }

    template<typename T, typename Arch, typename Tag, typename std::enable_if<!unsigned_overflow<T>::value && !need_promotion<T>::value, int>::type =0>
    float compare_template_l2_sqr(const T *__restrict x, const T *__restrict y, size_t d) {
        using b_type = collie::simd::batch<T, Arch>;
        std::size_t inc = b_type::size;
        std::size_t vec_size = d - d % inc;
        float sum = 0.0;
        b_type sum_vec = collie::simd::broadcast(T(0));
        for (std::size_t i = 0; i < vec_size; i += inc) {
            b_type xvec = b_type::load(x + i, Tag());
            b_type yvec = b_type::load(y + i, Tag());
            auto delta = collie::simd::sub(xvec, yvec);
            sum_vec += collie::simd::mul(delta, delta);
        }
        sum = collie::simd::reduce_add(sum_vec);
        for (std::size_t i = vec_size; i < d; ++i) {
            auto delta = x[i] - y[i];
            sum += delta * delta;
        }
        return sum;
    }

    template<typename T, typename U, typename Arch, typename Tag>
    float compare_template_l2_sqr(const T *__restrict x, const T *__restrict y, size_t d) {
        using b_type = collie::simd::batch<U, Arch>;
        using index_type = typename collie::simd::as_integer_t<b_type>;
        const index_type index = collie::simd::detail::make_sequence_as_batch<index_type>();
        std::size_t inc = b_type::size;
        std::size_t vec_size = d - d % inc;
        float sum = 0.0;
        b_type sum_vec = collie::simd::broadcast(U(0));
        for (std::size_t i = 0; i < vec_size; i += inc) {
            b_type xvec =b_type::gather(x + i, index);
            b_type yvec =b_type::gather(y + i, index);
            auto delta = collie::simd::sub(xvec, yvec);
            sum_vec += collie::simd::mul(delta, delta);
        }
        sum = collie::simd::reduce_add(sum_vec);
        for (std::size_t i = vec_size; i < d; ++i) {
            auto delta = x[i] - y[i];
            sum += delta * delta;
        }
        return sum;
    }

    inline distance_t compare_l2_sqr(const int8_t *__restrict a, const int8_t *__restrict b, size_t d) {
        return compare_template_l2_sqr<int8_t, float, collie::simd::best_arch, collie::simd::aligned_mode>(a, b, d);
    }

    inline distance_t compare_l2_sqr(const uint8_t *__restrict a, const uint8_t *__restrict b, size_t d) {
        return compare_template_l2_sqr<uint8_t, float, collie::simd::best_arch, collie::simd::aligned_mode>(a, b, d);
    }

    inline distance_t compare_l2_sqr(const int16_t *__restrict a, const int16_t *__restrict b, size_t d) {
        return compare_template_l2_sqr<int16_t, float, collie::simd::best_arch, collie::simd::aligned_mode>(a, b, d);
    }

    inline distance_t compare_l2_sqr(const uint16_t *__restrict a, const uint16_t *__restrict b, size_t d) {
        return compare_template_l2_sqr<uint16_t, float, collie::simd::best_arch, collie::simd::aligned_mode>(a, b, d);
    }

    inline distance_t compare_l2_sqr(const int32_t *__restrict a, const int32_t *__restrict b, size_t d) {
        return compare_template_l2_sqr<int32_t, collie::simd::best_arch, collie::simd::aligned_mode>(a, b, d);
    }

    inline distance_t compare_l2_sqr(const uint32_t *__restrict a, const uint32_t *__restrict b, size_t d) {
        return compare_template_l2_sqr<uint32_t, collie::simd::best_arch, collie::simd::aligned_mode>(a, b, d);
    }

    inline distance_t compare_l2_sqr(const int64_t *__restrict a, const int64_t *__restrict b, size_t d) {
        return compare_template_l2_sqr<int64_t, collie::simd::best_arch, collie::simd::aligned_mode>(a, b, d);
    }

    inline distance_t compare_l2_sqr(const uint64_t *__restrict a, const uint64_t *__restrict b, size_t d) {
        return compare_template_l2_sqr<uint64_t, collie::simd::best_arch, collie::simd::aligned_mode>(a, b, d);
    }

    inline distance_t compare_l2_sqr(const float *__restrict a, const float *__restrict b, size_t d) {
        return compare_template_l2_sqr<float, collie::simd::best_arch, collie::simd::aligned_mode>(a, b, d);
    }

    inline distance_t compare_l2_sqr(const float16 *__restrict a, const float16 *__restrict b, size_t size) {
        const float16 *last = a + size;
#if COLLIE_SIMD_AVX512F
        __m512 sum512 = _mm512_setzero_ps();
      while (a < last) {
    __m512 v = _mm512_sub_ps(_mm512_cvtph_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(a))),
                 _mm512_cvtph_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(b))));
    sum512 = _mm512_add_ps(sum512, _mm512_mul_ps(v, v));
    a += 16;
    b += 16;
      }

      __m256 sum256 = _mm256_add_ps(_mm512_extractf32x8_ps(sum512, 0), _mm512_extractf32x8_ps(sum512, 1));
      __m128 sum128 = _mm_add_ps(_mm256_extractf128_ps(sum256, 0), _mm256_extractf128_ps(sum256, 1));
#else
        __m256 sum256 = _mm256_setzero_ps();
        __m256 v;
        while (a < last) {
            v = _mm256_sub_ps(_mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i *>(a))),
                              _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i *>(b))));
            sum256 = _mm256_add_ps(sum256, _mm256_mul_ps(v, v));
            a += 8;
            b += 8;
            v = _mm256_sub_ps(_mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i *>(a))),
                              _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i *>(b))));
            sum256 = _mm256_add_ps(sum256, _mm256_mul_ps(v, v));
            a += 8;
            b += 8;
        }
        __m128 sum128 = _mm_add_ps(_mm256_extractf128_ps(sum256, 0), _mm256_extractf128_ps(sum256, 1));
#endif
        __m128 tmp = _mm_hadd_ps(sum128, _mm_set1_ps(0));
        double s = _mm_cvtss_f32(_mm_shuffle_ps(tmp, tmp, _MM_SHUFFLE(0, 0, 0, 0))) +
                   _mm_cvtss_f32(_mm_shuffle_ps(tmp, tmp, _MM_SHUFFLE(0, 0, 0, 1)));
        return s;
    }


    inline distance_t compare_l2(const int8_t *__restrict a, const int8_t *__restrict b, size_t d) {
        return std::sqrt(compare_template_l2_sqr<int8_t, float, collie::simd::best_arch, collie::simd::aligned_mode>(a, b, d));
    }

    inline distance_t compare_l2(const uint8_t *__restrict a, const uint8_t *__restrict b, size_t d) {
        return std::sqrt(compare_template_l2_sqr<uint8_t, float, collie::simd::best_arch, collie::simd::aligned_mode>(a, b, d));
    }

    inline distance_t compare_l2(const int16_t *__restrict a, const int16_t *__restrict b, size_t d) {
        return std::sqrt(compare_template_l2_sqr<int16_t, float, collie::simd::best_arch, collie::simd::aligned_mode>(a, b, d));
    }

    inline distance_t compare_l2(const uint16_t *__restrict a, const uint16_t *__restrict b, size_t d) {
        return std::sqrt(compare_template_l2_sqr<uint16_t, float, collie::simd::best_arch, collie::simd::aligned_mode>(a, b, d));
    }

    inline distance_t compare_l2(const int32_t *__restrict a, const int32_t *__restrict b, size_t d) {
        return std::sqrt(compare_template_l2_sqr<int32_t, collie::simd::best_arch, collie::simd::aligned_mode>(a, b, d));
    }

    inline distance_t compare_l2(const uint32_t *__restrict a, const uint32_t *__restrict b, size_t d) {
        return std::sqrt(compare_template_l2_sqr<uint32_t, collie::simd::best_arch, collie::simd::aligned_mode>(a, b, d));
    }

    inline distance_t compare_l2(const int64_t *__restrict a, const int64_t *__restrict b, size_t d) {
        return std::sqrt(compare_template_l2_sqr<int64_t, collie::simd::best_arch, collie::simd::aligned_mode>(a, b, d));
    }

    inline distance_t compare_l2(const uint64_t *__restrict a, const uint64_t *__restrict b, size_t d) {
        return std::sqrt(compare_template_l2_sqr<uint64_t, collie::simd::best_arch, collie::simd::aligned_mode>(a, b, d));
    }

    inline distance_t compare_l2(const float *__restrict a, const float *__restrict b, size_t d) {
        return std::sqrt(compare_template_l2_sqr<float, collie::simd::best_arch, collie::simd::aligned_mode>(a, b, d));
    }

    inline distance_t compare_l2(const float16 *__restrict a, const float16 *__restrict b, size_t size) {
        return std::sqrt(compare_l2_sqr(a, b, size));
    }

    inline distance_t compare_l2(const double *__restrict a, const double *__restrict b, size_t d) {
        return std::sqrt(compare_template_l2_sqr<double, collie::simd::best_arch, collie::simd::aligned_mode>(a, b, d));
    }

    /// l1
    template <typename T>
    inline distance_t compare_simple_l1(const T * __restrict a, const T *__restrict b, size_t size) {
        distance_t sum = 0;
        for (size_t i = 0; i < size; i++) {
            sum += std::abs((distance_t)a[i] - (distance_t)b[i]);
        }
        return sum;
    }

    template<typename T, typename Arch, typename Tag, typename std::enable_if<unsigned_overflow<T>::value && !need_promotion<T>::value, int>::type =0>
    distance_t compare_template_l1(const T *__restrict x, const T *__restrict y, size_t d) {
        using b_type = collie::simd::batch<T, Arch>;
        std::size_t inc = b_type::size;
        std::size_t vec_size = d - d % inc;
        float sum = 0.0;
        b_type sum_vec = collie::simd::broadcast(T(0));
        for (std::size_t i = 0; i < vec_size; i += inc) {
            b_type xvec = b_type::load(x + i, Tag());
            b_type yvec = b_type::load(y + i, Tag());
            auto max = collie::simd::max(xvec, yvec);
            auto min = collie::simd::min(xvec, yvec);
            sum_vec += collie::simd::sub(max, min);
        }
        sum = collie::simd::reduce_add(sum_vec);
        for (std::size_t i = vec_size; i < d; ++i) {
            sum += x[i] >= y[i] ? x[i] - y[i] : y[i] - x[i];
        }
        return sum;
    }

    template<typename T, typename Arch, typename Tag, typename std::enable_if<!unsigned_overflow<T>::value && !need_promotion<T>::value, int>::type =0>
    float compare_template_l1(const T *__restrict x, const T *__restrict y, size_t d) {
        using b_type = collie::simd::batch<T, Arch>;
        std::size_t inc = b_type::size;
        std::size_t vec_size = d - d % inc;
        float sum = 0.0;
        b_type sum_vec = collie::simd::broadcast(T(0));
        for (std::size_t i = 0; i < vec_size; i += inc) {
            b_type xvec = b_type::load(x + i, Tag());
            b_type yvec = b_type::load(y + i, Tag());
            sum_vec += collie::simd::abs(xvec - yvec);
        }
        sum = collie::simd::reduce_add(sum_vec);
        for (std::size_t i = vec_size; i < d; ++i) {
            sum += std::abs(x[i] - y[i]);
        }
        return sum;
    }

    template<typename T, typename U, typename Arch, typename Tag>
    float compare_template_l1(const T *__restrict x, const T *__restrict y, size_t d) {
        using b_type = collie::simd::batch<U, Arch>;
        using index_type = typename collie::simd::as_integer_t<b_type>;
        const index_type index = collie::simd::detail::make_sequence_as_batch<index_type>();
        std::size_t inc = b_type::size;
        std::size_t vec_size = d - d % inc;
        float sum = 0.0;
        b_type sum_vec = collie::simd::broadcast(U(0));
        for (std::size_t i = 0; i < vec_size; i += inc) {
            b_type xvec =b_type::gather(x + i, index);
            b_type yvec =b_type::gather(y + i, index);
            sum_vec += collie::simd::abs(xvec - yvec);
        }
        sum = collie::simd::reduce_add(sum_vec);
        for (std::size_t i = vec_size; i < d; ++i) {
            sum += std::abs((U)x[i] - (U)y[i]);
        }
        return sum;
    }

    inline distance_t compare_l1(const float * __restrict a, const float *__restrict b, size_t size) {
        return compare_template_l1<float, collie::simd::best_arch, collie::simd::aligned_mode>(a, b, size);
    }

    inline distance_t compare_l1(const uint16_t * __restrict a, const uint16_t *__restrict b, size_t size) {
        return compare_template_l1<uint16_t, float, collie::simd::best_arch, collie::simd::aligned_mode>(a, b, size);
    }

    inline distance_t compare_l1(const int16_t * __restrict a, const int16_t *__restrict b, size_t size) {
        return compare_template_l1<int16_t, float, collie::simd::best_arch, collie::simd::aligned_mode>(a, b, size);
    }

    inline distance_t compare_l1(const uint32_t * __restrict a, const uint32_t *__restrict b, size_t size) {
        return compare_template_l1<uint32_t, collie::simd::best_arch, collie::simd::aligned_mode>(a, b, size);
    }

    inline distance_t compare_l1(const int32_t * __restrict a, const int32_t *__restrict b, size_t size) {
        return compare_template_l1<int32_t, collie::simd::best_arch, collie::simd::aligned_mode>(a, b, size);
    }

    inline distance_t compare_l1(const uint64_t * __restrict a, const uint64_t *__restrict b, size_t size) {
        return compare_template_l1<uint64_t, collie::simd::best_arch, collie::simd::aligned_mode>(a, b, size);
    }

    inline distance_t compare_l1(const int64_t * __restrict a, const int64_t *__restrict b, size_t size) {
        return compare_template_l1<int64_t, collie::simd::best_arch, collie::simd::aligned_mode>(a, b, size);
    }

    inline distance_t compare_l1(const double * __restrict a, const double *__restrict b, size_t size) {
        return compare_template_l1<double, collie::simd::best_arch, collie::simd::aligned_mode>(a, b, size);
    }

    inline distance_t compare_l1(const float16 * __restrict a, const float16 * __restrict b, size_t size) {
        __m256 sum = _mm256_setzero_ps();
        const float16 *last = a + size;
        const float16 *lastgroup = last - 7;
        while (a < lastgroup) {
            __m256 x1 = _mm256_sub_ps(_mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i *>(a))),
                                      _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i *>(b))));
            const __m256 mask = _mm256_set1_ps(-0.0f);
            __m256 v = _mm256_andnot_ps(mask, x1);
            sum = _mm256_add_ps(sum, v);
            a += 8;
            b += 8;
        }
        __attribute__((aligned(32))) float f[8];
        _mm256_store_ps(f, sum);
        double s = f[0] + f[1] + f[2] + f[3] + f[4] + f[5] + f[6] + f[7];
        while (a < last) {
            double d = fabs(*a++ - *b++);
            s += d;
        }
        return s;
    }

    inline distance_t compare_l1(const bfloat16 *__restrict a, const bfloat16 *__restrict b, size_t size) {
        abort();
    }
    inline distance_t compare_l1(const uint8_t * __restrict a, const uint8_t * __restrict b, size_t size) {
        return compare_template_l1<unsigned char, float, collie::simd::best_arch, collie::simd::aligned_mode>(a, b, size);
    }
    inline distance_t compare_l1(const int8_t * __restrict a, const int8_t  * __restrict b, size_t size) {
        return compare_template_l1<int8_t, float, collie::simd::best_arch, collie::simd::aligned_mode>(a, b, size);
    }
    /// hamming
#if !defined(__POPCNT__)
    inline distance_t popCount(uint32_t x) {
          x = (x & 0x55555555) + (x >> 1 & 0x55555555);
          x = (x & 0x33333333) + (x >> 2 & 0x33333333);
          x = (x & 0x0F0F0F0F) + (x >> 4 & 0x0F0F0F0F);
          x = (x & 0x00FF00FF) + (x >> 8 & 0x00FF00FF);
          x = (x & 0x0000FFFF) + (x >> 16 & 0x0000FFFF);
          return x;
        }

        template <typename OBJECT_TYPE>
        inline distance_t compare_hamming_distance(const OBJECT_TYPE * __restrict a, const OBJECT_TYPE *__restrict b, size_t size) {
          const uint32_t *last = reinterpret_cast<const uint32_t*>(a + size);

          const uint32_t *uinta = reinterpret_cast<const uint32_t*>(a);
          const uint32_t *uintb = reinterpret_cast<const uint32_t*>(b);
          size_t count = 0;
          while( uinta < last ){
        count += popCount(*uinta++ ^ *uintb++);
          }

          return static_cast<distance_t>(count);
        }
#else

    template<typename OBJECT_TYPE>
    inline distance_t compare_hamming_distance(const OBJECT_TYPE *__restrict a, const OBJECT_TYPE *__restrict b, size_t size) {
        const uint64_t *last = reinterpret_cast<const uint64_t *>(a + size);

        const uint64_t *uinta = reinterpret_cast<const uint64_t *>(a);
        const uint64_t *uintb = reinterpret_cast<const uint64_t *>(b);
        size_t count = 0;
        while (uinta < last) {
            count += _mm_popcnt_u64(*uinta++ ^ *uintb++);
            count += _mm_popcnt_u64(*uinta++ ^ *uintb++);
        }

        return static_cast<distance_t>(count);
    }
#endif

    /// jaccard
#if !defined(__POPCNT__)
    template <typename OBJECT_TYPE>
          inline distance_t compare_jaccard_distance(const OBJECT_TYPE *__restrict a, const OBJECT_TYPE *__restrict b, size_t size) {
          const uint32_t *last = reinterpret_cast<const uint32_t*>(a + size);

          const uint32_t *uinta = reinterpret_cast<const uint32_t*>(a);
          const uint32_t *uintb = reinterpret_cast<const uint32_t*>(b);
          size_t count = 0;
          size_t countDe = 0;
          while( uinta < last ){
        count   += popCount(*uinta   & *uintb);
        countDe += popCount(*uinta++ | *uintb++);
        count   += popCount(*uinta   & *uintb);
        countDe += popCount(*uinta++ | *uintb++);
          }

          return 1.0 - static_cast<distance_t>(count) / static_cast<distance_t>(countDe);
        }
#else

    template<typename OBJECT_TYPE>
    inline distance_t compare_jaccard_distance(const OBJECT_TYPE *__restrict a, const OBJECT_TYPE *__restrict b, size_t size) {
        const uint64_t *last = reinterpret_cast<const uint64_t *>(a + size);

        const uint64_t *uinta = reinterpret_cast<const uint64_t *>(a);
        const uint64_t *uintb = reinterpret_cast<const uint64_t *>(b);
        size_t count = 0;
        size_t countDe = 0;
        while (uinta < last) {
            count += _mm_popcnt_u64(*uinta & *uintb);
            countDe += _mm_popcnt_u64(*uinta++ | *uintb++);
            count += _mm_popcnt_u64(*uinta & *uintb);
            countDe += _mm_popcnt_u64(*uinta++ | *uintb++);
        }

        return 1.0 - static_cast<distance_t>(count) / static_cast<distance_t>(countDe);
    }

#endif
    /// sparse jaccard
    inline distance_t compare_sparse_jaccard_distance(const unsigned char *__restrict a, const unsigned char *__restrict b, size_t size) {
        std::cerr << "compare_sparse_jaccard_distance: Not implemented." << std::endl;
        abort();
    }

    inline distance_t compare_sparse_jaccard_distance(const float16 *__restrict a, const float16 *__restrict b, size_t size) {
        std::cerr << "compare_sparse_jaccard_distance: Not implemented." << std::endl;
        abort();
    }

    inline distance_t compare_sparse_jaccard_distance(const bfloat16 *__restrict a, const bfloat16 *__restrict b, size_t size) {
        std::cerr << "compare_sparse_jaccard_distance: Not implemented." << std::endl;
        abort();
    }

    inline distance_t compare_sparse_jaccard_distance(const float *__restrict a, const float *__restrict b, size_t size) {
        size_t loca = 0;
        size_t locb = 0;
        const uint32_t *ai = reinterpret_cast<const uint32_t *>(a);
        const uint32_t *bi = reinterpret_cast<const uint32_t *>(b);
        size_t count = 0;
        while (locb < size && ai[loca] != 0 && bi[loca] != 0) {
            int64_t sub = static_cast<int64_t>(ai[loca]) - static_cast<int64_t>(bi[locb]);
            count += sub == 0;
            loca += sub <= 0;
            locb += sub >= 0;
        }
        while (ai[loca] != 0) {
            loca++;
        }
        while (locb < size && bi[locb] != 0) {
            locb++;
        }
        return 1.0f - static_cast<distance_t>(count) / static_cast<distance_t>(loca + locb - count);
    }

    /// dot product

    template<typename T>
    distance_t compare_simple_dot_product(const T * __restrict a, const T *__restrict b, size_t size) {
        float sum = 0.0;
        for (size_t i = 0; i < size; i++) {
            sum += (distance_t)a[i] * (distance_t)b[i];
        }
        return sum;
    }

    template<typename T, typename Arch, typename Tag, std::enable_if_t<!need_promotion<T>::value, int> = 0>
    distance_t compare_template_dot_product(const T *__restrict x, const T *__restrict y, size_t d) {
        using b_type = collie::simd::batch<T, Arch>;
        std::size_t inc = b_type::size;
        std::size_t vec_size = d - d % inc;
        float sum = 0.0;
        b_type sum_vec = collie::simd::broadcast(T(0));
        for (std::size_t i = 0; i < vec_size; i += inc) {
            b_type xvec = b_type::load(x + i, Tag());
            b_type yvec = b_type::load(y + i, Tag());
            sum_vec += xvec * yvec;
        }
        sum = collie::simd::reduce_add(sum_vec);
        for (std::size_t i = vec_size; i < d; ++i) {
            sum += x[i] * y[i];
        }
        return sum;
    }

    template<typename T, typename U, typename Arch, typename Tag, std::enable_if_t<need_promotion<T>::value, int> = 0>
    distance_t compare_template_dot_product(const T *__restrict x, const T *__restrict y, size_t d) {
        using b_type = collie::simd::batch<U, Arch>;
        using index_type = typename collie::simd::as_integer_t<b_type>;
        const index_type index = collie::simd::detail::make_sequence_as_batch<index_type>();
        std::size_t inc = b_type::size;
        std::size_t vec_size = d - d % inc;
        float sum = 0.0;
        b_type sum_vec = collie::simd::broadcast(U(0));
        for (std::size_t i = 0; i < vec_size; i += inc) {
            b_type xvec =b_type::gather(x + i, index);
            b_type yvec =b_type::gather(y + i, index);
            sum_vec += xvec * yvec;
        }
        sum = collie::simd::reduce_add(sum_vec);
        for (std::size_t i = vec_size; i < d; ++i) {
            sum += x[i] * y[i];
        }
        return sum;
    }

    inline distance_t compare_template_dot_product(const float16 *__restrict a, const float16 *__restrict b, size_t size) {
        const float16 *last = a + size;
#if defined(NGT_AVX512)
        __m512 sum512 = _mm512_setzero_ps();
            while (a < last) {
          sum512 = _mm512_add_ps(sum512, _mm512_mul_ps(_mm512_cvtph_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(a))),
                                   _mm512_cvtph_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(b)))));

          a += 16;
          b += 16;
            }
            __m256 sum256 = _mm256_add_ps(_mm512_extractf32x8_ps(sum512, 0), _mm512_extractf32x8_ps(sum512, 1));
            __m128 sum128 = _mm_add_ps(_mm256_extractf128_ps(sum256, 0), _mm256_extractf128_ps(sum256, 1));
#elif defined(NGT_AVX2)
        __m256 sum256 = _mm256_setzero_ps();
        while (a < last) {
            sum256 = _mm256_add_ps(sum256, _mm256_mul_ps(
                    _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i *>(a))),
                    _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i *>(b)))));
            a += 8;
            b += 8;
        }
        __m128 sum128 = _mm_add_ps(_mm256_extractf128_ps(sum256, 0), _mm256_extractf128_ps(sum256, 1));
#else
        __m128 sum128 = _mm_setzero_ps();
            while (a < last) {
          __m128i va = _mm_load_si128(reinterpret_cast<const __m128i*>(a));
          __m128i vb = _mm_load_si128(reinterpret_cast<const __m128i*>(b));
          sum128 = _mm_add_ps(sum128, _mm_mul_ps(_mm_cvtph_ps(va), _mm_cvtph_ps(vb)));
          va = _mm_srli_si128(va, 8);
          vb = _mm_srli_si128(vb, 8);
          sum128 = _mm_add_ps(sum128, _mm_mul_ps(_mm_cvtph_ps(va), _mm_cvtph_ps(vb)));
          a += 8;
          b += 8;
            }
#endif
        __attribute__((aligned(32))) float f[4];
        _mm_store_ps(f, sum128);
        double s = static_cast<double>(f[0]) + static_cast<double>(f[1]) + static_cast<double>(f[2]) +
                   static_cast<double>(f[3]);
        return s;
    }

    inline distance_t compare_dot_product(const int8_t *__restrict a, const int8_t *__restrict b, size_t size) {
        return compare_template_dot_product<int8_t, float, collie::simd::best_arch, collie::simd::aligned_mode>(a, b, size);
    }

    inline distance_t compare_dot_product(const uint8_t *__restrict a, const uint8_t *__restrict b, size_t size) {
        return compare_template_dot_product<uint8_t, float, collie::simd::best_arch, collie::simd::aligned_mode>(a, b, size);
    }

    inline distance_t compare_dot_product(const int16_t *__restrict a, const int16_t *__restrict b, size_t size) {
        return compare_template_dot_product<int16_t, float, collie::simd::best_arch, collie::simd::aligned_mode>(a, b, size);
    }

    inline distance_t compare_dot_product(const uint16_t *__restrict a, const uint16_t *__restrict b, size_t size) {
        return compare_template_dot_product<uint16_t, float, collie::simd::best_arch, collie::simd::aligned_mode>(a, b, size);
    }

    inline distance_t compare_dot_product(const int32_t *__restrict a, const int32_t *__restrict b, size_t size) {
        return compare_template_dot_product<int32_t, collie::simd::best_arch, collie::simd::aligned_mode>(a, b, size);
    }

    inline distance_t compare_dot_product(const uint32_t *__restrict a, const uint32_t *__restrict b, size_t size) {
        return compare_template_dot_product<uint32_t, collie::simd::best_arch, collie::simd::aligned_mode>(a, b, size);
    }

    inline distance_t compare_dot_product(const int64_t *__restrict a, const int64_t *__restrict b, size_t size) {
        return compare_template_dot_product<int64_t, collie::simd::best_arch, collie::simd::aligned_mode>(a, b, size);
    }

    inline distance_t compare_dot_product(const uint64_t *__restrict a, const uint64_t *__restrict b, size_t size) {
        return compare_template_dot_product<uint64_t, collie::simd::best_arch, collie::simd::aligned_mode>(a, b, size);
    }

    inline distance_t compare_dot_product(const float *__restrict a, const float *__restrict b, size_t size) {
        return compare_template_dot_product<float, collie::simd::best_arch, collie::simd::aligned_mode>(a, b, size);
    }

    inline distance_t compare_dot_product(const float16 *__restrict a, const float16 *__restrict b, size_t size) {
        return compare_template_dot_product(a, b, size);
    }

    inline distance_t compare_dot_product(const bfloat16 *__restrict a, const bfloat16 *__restrict b, size_t size) {
        abort();
    }

    inline distance_t compare_dot_product(const double *__restrict a, const double *__restrict b, size_t size) {
        return compare_template_dot_product<double, collie::simd::best_arch, collie::simd::aligned_mode>(a, b, size);
    }


    /// distance cosine

    template<typename T>
    distance_t compare_simple_cosine(const T * __restrict a, const T *__restrict b, size_t size) {
        float sum = 0.0;
        float normA = 0.0;
        float normB = 0.0;
        for (size_t i = 0; i < size; i++) {
            normA += (distance_t)a[i] * (distance_t)a[i];
            normB += (distance_t)b[i] * (distance_t)b[i];
            sum += (distance_t)a[i] * (distance_t)b[i];
        }
        return sum / std::sqrt(normA * normB);
    }

    template<typename T, typename Arch, typename Tag, std::enable_if_t<!need_promotion<T>::value, int> = 0>
    distance_t compare_template_cosine(const T *__restrict x, const T *__restrict y, size_t d) {
        using b_type = collie::simd::batch<T, Arch>;
        std::size_t inc = b_type::size;
        std::size_t vec_size = d - d % inc;
        float sum = 0.0;
        b_type sum_vec = collie::simd::broadcast(T(0));
        b_type normA = collie::simd::broadcast(T(0));
        b_type normB = collie::simd::broadcast(T(0));
        for (std::size_t i = 0; i < vec_size; i += inc) {
            b_type xvec = b_type::load(x + i, Tag());
            b_type yvec = b_type::load(y + i, Tag());
            sum_vec += xvec * yvec;
            normA += xvec * xvec;
            normB += yvec * yvec;
        }
        sum = collie::simd::reduce_add(sum_vec);
        float na = collie::simd::reduce_add(normA);
        float nb = collie::simd::reduce_add(normB);
        for (std::size_t i = vec_size; i < d; ++i) {
            sum += x[i] * y[i];
            na += x[i] * x[i];
            nb += y[i] * y[i];
        }
        return sum / std::sqrt(na * nb);
    }

    template<typename T, typename U, typename Arch, typename Tag, std::enable_if_t<need_promotion<T>::value, int> = 0>
    distance_t compare_template_cosine(const T *__restrict x, const T *__restrict y, size_t d) {
        using b_type = collie::simd::batch<U, Arch>;
        using index_type = typename collie::simd::as_integer_t<b_type>;
        const index_type index = collie::simd::detail::make_sequence_as_batch<index_type>();
        std::size_t inc = b_type::size;
        std::size_t vec_size = d - d % inc;
        float sum = 0.0;
        b_type sum_vec = collie::simd::broadcast(U(0));
        b_type normA = collie::simd::broadcast(U(0));
        b_type normB = collie::simd::broadcast(U(0));
        for (std::size_t i = 0; i < vec_size; i += inc) {
            b_type xvec =b_type::gather(x + i, index);
            b_type yvec =b_type::gather(y + i, index);
            sum_vec += xvec * yvec;
            normA += xvec * xvec;
            normB += yvec * yvec;
        }
        sum = collie::simd::reduce_add(sum_vec);
        float na = collie::simd::reduce_add(normA);
        float nb = collie::simd::reduce_add(normB);
        for (std::size_t i = vec_size; i < d; ++i) {
            sum += x[i] * y[i];
            na += x[i] * x[i];
            nb += y[i] * y[i];
        }
        return sum / std::sqrt(na * nb);
    }
    inline distance_t compare_template_cosine(const float16 *__restrict a, const float16 *__restrict b, size_t size) {

        const float16 *last = a + size;
#if defined(NGT_AVX512)
        __m512 normA = _mm512_setzero_ps();
            __m512 normB = _mm512_setzero_ps();
            __m512 sum = _mm512_setzero_ps();
            while (a < last) {
          __m512 am = _mm512_cvtph_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(a)));
          __m512 bm = _mm512_cvtph_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(b)));
          normA = _mm512_add_ps(normA, _mm512_mul_ps(am, am));
          normB = _mm512_add_ps(normB, _mm512_mul_ps(bm, bm));
          sum = _mm512_add_ps(sum, _mm512_mul_ps(am, bm));
          a += 16;
          b += 16;
            }
            __m256 am256 = _mm256_add_ps(_mm512_extractf32x8_ps(normA, 0), _mm512_extractf32x8_ps(normA, 1));
            __m256 bm256 = _mm256_add_ps(_mm512_extractf32x8_ps(normB, 0), _mm512_extractf32x8_ps(normB, 1));
            __m256 s256 = _mm256_add_ps(_mm512_extractf32x8_ps(sum, 0), _mm512_extractf32x8_ps(sum, 1));
            __m128 am128 = _mm_add_ps(_mm256_extractf128_ps(am256, 0), _mm256_extractf128_ps(am256, 1));
            __m128 bm128 = _mm_add_ps(_mm256_extractf128_ps(bm256, 0), _mm256_extractf128_ps(bm256, 1));
            __m128 s128 = _mm_add_ps(_mm256_extractf128_ps(s256, 0), _mm256_extractf128_ps(s256, 1));
#elif defined(NGT_AVX2)
        __m256 normA = _mm256_setzero_ps();
        __m256 normB = _mm256_setzero_ps();
        __m256 sum = _mm256_setzero_ps();
        __m256 am, bm;
        while (a < last) {
            am = _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i *>(a)));
            bm = _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i *>(b)));
            normA = _mm256_add_ps(normA, _mm256_mul_ps(am, am));
            normB = _mm256_add_ps(normB, _mm256_mul_ps(bm, bm));
            sum = _mm256_add_ps(sum, _mm256_mul_ps(am, bm));
            a += 8;
            b += 8;
        }
        __m128 am128 = _mm_add_ps(_mm256_extractf128_ps(normA, 0), _mm256_extractf128_ps(normA, 1));
        __m128 bm128 = _mm_add_ps(_mm256_extractf128_ps(normB, 0), _mm256_extractf128_ps(normB, 1));
        __m128 s128 = _mm_add_ps(_mm256_extractf128_ps(sum, 0), _mm256_extractf128_ps(sum, 1));
#else
        __m128 am128 = _mm_setzero_ps();
            __m128 bm128 = _mm_setzero_ps();
            __m128 s128 = _mm_setzero_ps();
            __m128 am, bm;
            while (a < last) {
          __m128i va = _mm_load_si128(reinterpret_cast<const __m128i*>(a));
          __m128i vb = _mm_load_si128(reinterpret_cast<const __m128i*>(b));
          am = _mm_cvtph_ps(va);
          bm = _mm_cvtph_ps(vb);
          am128 = _mm_add_ps(am128, _mm_mul_ps(am, am));
          bm128 = _mm_add_ps(bm128, _mm_mul_ps(bm, bm));
          s128 = _mm_add_ps(s128, _mm_mul_ps(am, bm));
          va = _mm_srli_si128(va, 8);
          vb = _mm_srli_si128(vb, 8);
          am = _mm_cvtph_ps(va);
          bm = _mm_cvtph_ps(vb);
          am128 = _mm_add_ps(am128, _mm_mul_ps(am, am));
          bm128 = _mm_add_ps(bm128, _mm_mul_ps(bm, bm));
          s128 = _mm_add_ps(s128, _mm_mul_ps(am, bm));
          a += 8;
          b += 8;
            }

#endif

        __attribute__((aligned(32))) float f[4];
        _mm_store_ps(f, am128);
        distance_t na = f[0] + f[1] + f[2] + f[3];
        _mm_store_ps(f, bm128);
        double nb = f[0] + f[1] + f[2] + f[3];
        _mm_store_ps(f, s128);
        double s = f[0] + f[1] + f[2] + f[3];

        double cosine = s / sqrt(na * nb);
        return cosine;
    }

    inline distance_t compare_cosine(const int8_t *__restrict a, const int8_t *__restrict b, size_t size) {
        return compare_template_cosine<int8_t, float, collie::simd::best_arch, collie::simd::aligned_mode>(a, b, size);
    }

    inline distance_t compare_cosine(const uint8_t *__restrict a, const uint8_t *__restrict b, size_t size) {
        return compare_template_cosine<uint8_t, float, collie::simd::best_arch, collie::simd::aligned_mode>(a, b, size);
    }

    inline distance_t compare_cosine(const int16_t *__restrict a, const int16_t *__restrict b, size_t size) {
        return compare_template_cosine<int16_t, float, collie::simd::best_arch, collie::simd::aligned_mode>(a, b, size);
    }

    inline distance_t compare_cosine(const uint16_t *__restrict a, const uint16_t *__restrict b, size_t size) {
        return compare_template_cosine<uint16_t, float, collie::simd::best_arch, collie::simd::aligned_mode>(a, b, size);
    }

    inline distance_t compare_cosine(const int32_t *__restrict a, const int32_t *__restrict b, size_t size) {
        return compare_template_cosine<int32_t, collie::simd::best_arch, collie::simd::aligned_mode>(a, b, size);
    }

    inline distance_t compare_cosine(const uint32_t *__restrict a, const uint32_t *__restrict b, size_t size) {
        return compare_template_cosine<uint32_t, collie::simd::best_arch, collie::simd::aligned_mode>(a, b, size);
    }

    inline distance_t compare_cosine(const int64_t *__restrict a, const int64_t *__restrict b, size_t size) {
        return compare_template_cosine<int64_t, collie::simd::best_arch, collie::simd::aligned_mode>(a, b, size);
    }

    inline distance_t compare_cosine(const uint64_t *__restrict a, const uint64_t *__restrict b, size_t size) {
        return compare_template_cosine<uint64_t, collie::simd::best_arch, collie::simd::aligned_mode>(a, b, size);
    }

    inline distance_t compare_cosine(const float *__restrict a, const float *__restrict b, size_t size) {
        return compare_template_cosine<float, collie::simd::best_arch, collie::simd::aligned_mode>(a, b, size);
    }

    inline distance_t compare_cosine(const double *__restrict a, const double *__restrict b, size_t size) {
        return compare_template_cosine<double, collie::simd::best_arch, collie::simd::aligned_mode>(a, b, size);
    }

    inline distance_t compare_cosine(const float16 *__restrict a, const float16 *__restrict b, size_t size) {
        return compare_template_cosine(a, b, size);
    }

    inline distance_t compare_cosine(const bfloat16 *__restrict a, const bfloat16 *__restrict b, size_t size) {
        abort();
    }

    template<typename T>
    distance_t compare_simple_cosine_similarity(const T * __restrict a, const T *__restrict b, size_t size) {
        auto v = 1.0 - compare_simple_cosine(a, b, size);
        return v < 0.0 ? -v : v;
    }

    inline distance_t compare_cosine_similarity(const int8_t *__restrict a, const int8_t *__restrict b, size_t size) {
        auto v = 1.0 - compare_cosine(a, b, size);
        return v < 0.0 ? -v : v;
    }

    inline distance_t compare_cosine_similarity(const uint8_t *__restrict a, const uint8_t *__restrict b, size_t size) {
        auto v = 1.0 - compare_cosine(a, b, size);
        return v < 0.0 ? -v : v;
    }

    inline distance_t compare_cosine_similarity(const int16_t *__restrict a, const int16_t *__restrict b, size_t size) {
        auto v = 1.0 - compare_cosine(a, b, size);
        return v < 0.0 ? -v : v;
    }

    inline distance_t compare_cosine_similarity(const uint16_t *__restrict a, const uint16_t *__restrict b, size_t size) {
        auto v = 1.0 - compare_cosine(a, b, size);
        return v < 0.0 ? -v : v;
    }

    inline distance_t compare_cosine_similarity(const int32_t *__restrict a, const int32_t *__restrict b, size_t size) {
        auto v = 1.0 - compare_cosine(a, b, size);
        return v < 0.0 ? -v : v;
    }

    inline distance_t compare_cosine_similarity(const uint32_t *__restrict a, const uint32_t *__restrict b, size_t size) {
        auto v = 1.0 - compare_cosine(a, b, size);
        return v < 0.0 ? -v : v;
    }

    inline distance_t compare_cosine_similarity(const int64_t *__restrict a, const int64_t *__restrict b, size_t size) {
        auto v = 1.0 - compare_cosine(a, b, size);
        return v < 0.0 ? -v : v;
    }

    inline distance_t compare_cosine_similarity(const uint64_t *__restrict a, const uint64_t *__restrict b, size_t size) {
        auto v = 1.0 - compare_cosine(a, b, size);
        return v < 0.0 ? -v : v;
    }

    inline distance_t compare_cosine_similarity(const float *__restrict a, const float *__restrict b, size_t size) {
        auto v = 1.0 - compare_cosine(a, b, size);
        return v < 0.0 ? -v : v;
    }

    inline distance_t compare_cosine_similarity(const double *__restrict a, const double *__restrict b, size_t size) {
        auto v = 1.0 - compare_cosine(a, b, size);
        return v < 0.0 ? -v : v;
    }

    inline distance_t compare_cosine_similarity(const float16 *__restrict a, const float16 *__restrict b, size_t size) {
        auto v = 1.0 - compare_cosine(a, b, size);
        return v < 0.0 ? -v : v;
    }

    inline distance_t compare_cosine_similarity(const bfloat16 *__restrict a, const bfloat16 *__restrict b, size_t size) {
        abort();
    }

    /// distance angle
    template<typename T>
    distance_t compare_simple_angle(const T * __restrict a, const T *__restrict b, size_t size) {
        float sum = 0.0;
        float normA = 0.0;
        float normB = 0.0;
        for (size_t i = 0; i < size; i++) {
            normA += (distance_t)a[i] * (distance_t)a[i];
            normB += (distance_t)b[i] * (distance_t)b[i];
            sum += (distance_t)a[i] * (distance_t)b[i];
        }
        auto cosine = sum / std::sqrt(normA * normB);
        if (cosine >= 1.0) {
            return 0.0;
        } else if (cosine <= -1.0) {
            return std::acos(-1.0);
        } else {
            return std::acos(cosine);
        }
    }

    inline distance_t compare_angle_distance(const int8_t *__restrict a, const int8_t *__restrict b, size_t size) {
        distance_t cosine = compare_cosine(a, b, size);
        if (cosine >= 1.0) {
            return 0.0;
        } else if (cosine <= -1.0) {
            return std::acos(-1.0);
        } else {
            return std::acos(cosine);
        }
    }

    inline distance_t compare_angle_distance(const uint8_t *__restrict a, const uint8_t *__restrict b, size_t size) {
        distance_t cosine = compare_cosine(a, b, size);
        if (cosine >= 1.0) {
            return 0.0;
        } else if (cosine <= -1.0) {
            return std::acos(-1.0);
        } else {
            return std::acos(cosine);
        }
    }

    inline distance_t compare_angle_distance(const int16_t *__restrict a, const int16_t *__restrict b, size_t size) {
        distance_t cosine = compare_cosine(a, b, size);
        if (cosine >= 1.0) {
            return 0.0;
        } else if (cosine <= -1.0) {
            return std::acos(-1.0);
        } else {
            return std::acos(cosine);
        }
    }

    inline distance_t compare_angle_distance(const uint16_t *__restrict a, const uint16_t *__restrict b, size_t size) {
        distance_t cosine = compare_cosine(a, b, size);
        if (cosine >= 1.0) {
            return 0.0;
        } else if (cosine <= -1.0) {
            return std::acos(-1.0);
        } else {
            return std::acos(cosine);
        }
    }

    inline distance_t compare_angle_distance(const int32_t *__restrict a, const int32_t *__restrict b, size_t size) {
        distance_t cosine = compare_cosine(a, b, size);
        if (cosine >= 1.0) {
            return 0.0;
        } else if (cosine <= -1.0) {
            return std::acos(-1.0);
        } else {
            return std::acos(cosine);
        }
    }

    inline distance_t compare_angle_distance(const uint32_t *__restrict a, const uint32_t *__restrict b, size_t size) {
        distance_t cosine = compare_cosine(a, b, size);
        if (cosine >= 1.0) {
            return 0.0;
        } else if (cosine <= -1.0) {
            return std::acos(-1.0);
        } else {
            return std::acos(cosine);
        }
    }

    inline distance_t compare_angle_distance(const int64_t *__restrict a, const int64_t *__restrict b, size_t size) {
        distance_t cosine = compare_cosine(a, b, size);
        if (cosine >= 1.0) {
            return 0.0;
        } else if (cosine <= -1.0) {
            return std::acos(-1.0);
        } else {
            return std::acos(cosine);
        }
    }

    inline distance_t compare_angle_distance(const uint64_t *__restrict a, const uint64_t *__restrict b, size_t size) {
        distance_t cosine = compare_cosine(a, b, size);
        if (cosine >= 1.0) {
            return 0.0;
        } else if (cosine <= -1.0) {
            return std::acos(-1.0);
        } else {
            return std::acos(cosine);
        }
    }

    inline distance_t compare_angle_distance(const float *__restrict a, const float *__restrict b, size_t size) {
        distance_t cosine = compare_cosine(a, b, size);
        if (cosine >= 1.0) {
            return 0.0;
        } else if (cosine <= -1.0) {
            return std::acos(-1.0);
        } else {
            return std::acos(cosine);
        }
    }

    inline distance_t compare_angle_distance(const double *__restrict a, const double *__restrict b, size_t size) {
        distance_t cosine = compare_cosine(a, b, size);
        if (cosine >= 1.0) {
            return 0.0;
        } else if (cosine <= -1.0) {
            return std::acos(-1.0);
        } else {
            return std::acos(cosine);
        }
    }

    inline distance_t compare_angle_distance(const float16 *__restrict a, const float16 *__restrict b, size_t size) {
        distance_t cosine = compare_cosine(a, b, size);
        if (cosine >= 1.0) {
            return 0.0;
        } else if (cosine <= -1.0) {
            return std::acos(-1.0);
        } else {
            return std::acos(cosine);
        }
    }

    inline distance_t compare_angle_distance(const bfloat16 *__restrict a, const bfloat16 *__restrict b, size_t size) {
        distance_t cosine = compare_cosine(a, b, size);
        if (cosine >= 1.0) {
            return 0.0;
        } else if (cosine <= -1.0) {
            return std::acos(-1.0);
        } else {
            return std::acos(cosine);
        }
    }

    inline distance_t compare_normalized_angle_distance(const int8_t *__restrict a, const int8_t *__restrict b, size_t size) {
        distance_t cosine = polaris::primitive::compare_dot_product(a, b, size);
        if (cosine >= 1.0) {
            return 0.0;
        } else if (cosine <= -1.0) {
            return std::acos(-1.0);
        } else {
            return std::acos(cosine);
        }
    }

    inline distance_t compare_normalized_angle_distance(const uint8_t *__restrict a, const uint8_t *__restrict b, size_t size) {
        distance_t cosine = polaris::primitive::compare_dot_product(a, b, size);
        if (cosine >= 1.0) {
            return 0.0;
        } else if (cosine <= -1.0) {
            return std::acos(-1.0);
        } else {
            return std::acos(cosine);
        }
    }

    inline distance_t compare_normalized_angle_distance(const int16_t *__restrict a, const int16_t *__restrict b, size_t size) {
        distance_t cosine = polaris::primitive::compare_dot_product(a, b, size);
        if (cosine >= 1.0) {
            return 0.0;
        } else if (cosine <= -1.0) {
            return std::acos(-1.0);
        } else {
            return std::acos(cosine);
        }
    }

    inline distance_t compare_normalized_angle_distance(const uint16_t *__restrict a, const uint16_t *__restrict b, size_t size) {
        distance_t cosine = polaris::primitive::compare_dot_product(a, b, size);
        if (cosine >= 1.0) {
            return 0.0;
        } else if (cosine <= -1.0) {
            return std::acos(-1.0);
        } else {
            return std::acos(cosine);
        }
    }

    inline distance_t compare_normalized_angle_distance(const int32_t *__restrict a, const int32_t *__restrict b, size_t size) {
        distance_t cosine = polaris::primitive::compare_dot_product(a, b, size);
        if (cosine >= 1.0) {
            return 0.0;
        } else if (cosine <= -1.0) {
            return std::acos(-1.0);
        } else {
            return std::acos(cosine);
        }
    }

    inline distance_t compare_normalized_angle_distance(const uint32_t *__restrict a, const uint32_t *__restrict b, size_t size) {
        distance_t cosine = polaris::primitive::compare_dot_product(a, b, size);
        if (cosine >= 1.0) {
            return 0.0;
        } else if (cosine <= -1.0) {
            return std::acos(-1.0);
        } else {
            return std::acos(cosine);
        }
    }

    inline distance_t compare_normalized_angle_distance(const int64_t *__restrict a, const int64_t *__restrict b, size_t size) {
        distance_t cosine = polaris::primitive::compare_dot_product(a, b, size);
        if (cosine >= 1.0) {
            return 0.0;
        } else if (cosine <= -1.0) {
            return std::acos(-1.0);
        } else {
            return std::acos(cosine);
        }
    }

    inline distance_t compare_normalized_angle_distance(const uint64_t *__restrict a, const uint64_t *__restrict b, size_t size) {
        distance_t cosine = polaris::primitive::compare_dot_product(a, b, size);
        if (cosine >= 1.0) {
            return 0.0;
        } else if (cosine <= -1.0) {
            return std::acos(-1.0);
        } else {
            return std::acos(cosine);
        }
    }

    inline distance_t compare_normalized_angle_distance(const float *__restrict a, const float *__restrict b, size_t size) {
        distance_t cosine = polaris::primitive::compare_dot_product(a, b, size);
        if (cosine >= 1.0) {
            return 0.0;
        } else if (cosine <= -1.0) {
            return std::acos(-1.0);
        } else {
            return std::acos(cosine);
        }
    }

    inline distance_t compare_normalized_angle_distance(const double *__restrict a, const double *__restrict b, size_t size) {
        distance_t cosine = polaris::primitive::compare_dot_product(a, b, size);
        if (cosine >= 1.0) {
            return 0.0;
        } else if (cosine <= -1.0) {
            return std::acos(-1.0);
        } else {
            return std::acos(cosine);
        }
    }

    inline distance_t compare_normalized_angle_distance(const float16 *__restrict a, const float16 *__restrict b, size_t size) {
        distance_t cosine = polaris::primitive::compare_dot_product(a, b, size);
        if (cosine >= 1.0) {
            return 0.0;
        } else if (cosine <= -1.0) {
            return std::acos(-1.0);
        } else {
            return std::acos(cosine);
        }
    }

    inline distance_t compare_normalized_angle_distance(const bfloat16 *__restrict a, const bfloat16 *__restrict b, size_t size) {
        abort();
    }

    /// distance l2

    inline distance_t compare_normalized_l2(const int8_t * __restrict a, const int8_t * __restrict b, size_t size) {
        double v = 2.0 - 2.0 * polaris::primitive::compare_dot_product(a, b, size);
        if (v < 0.0) {
            return 0.0;
        } else {
            return sqrt(v);
        }
    }

    inline distance_t compare_normalized_l2(const uint8_t *__restrict a, const uint8_t *__restrict b, size_t size) {
        double v = 2.0 - 2.0 * polaris::primitive::compare_dot_product(a, b, size);
        if (v < 0.0) {
            return 0.0;
        } else {
            return sqrt(v);
        }
    }

    inline distance_t compare_normalized_l2(const int16_t *__restrict a, const int16_t *__restrict b, size_t size) {
        double v = 2.0 - 2.0 * polaris::primitive::compare_dot_product(a, b, size);
        if (v < 0.0) {
            return 0.0;
        } else {
            return sqrt(v);
        }
    }

    inline distance_t compare_normalized_l2(const uint16_t *__restrict a, const uint16_t *__restrict b, size_t size) {
        double v = 2.0 - 2.0 * polaris::primitive::compare_dot_product(a, b, size);
        if (v < 0.0) {
            return 0.0;
        } else {
            return sqrt(v);
        }
    }

    inline distance_t compare_normalized_l2(const int32_t *__restrict a, const int32_t *__restrict b, size_t size) {
        double v = 2.0 - 2.0 * polaris::primitive::compare_dot_product(a, b, size);
        if (v < 0.0) {
            return 0.0;
        } else {
            return sqrt(v);
        }
    }

    inline distance_t compare_normalized_l2(const uint32_t *__restrict a, const uint32_t *__restrict b, size_t size) {
        double v = 2.0 - 2.0 * polaris::primitive::compare_dot_product(a, b, size);
        if (v < 0.0) {
            return 0.0;
        } else {
            return sqrt(v);
        }
    }

    inline distance_t compare_normalized_l2(const int64_t *__restrict a, const int64_t *__restrict b, size_t size) {
        double v = 2.0 - 2.0 * polaris::primitive::compare_dot_product(a, b, size);
        if (v < 0.0) {
            return 0.0;
        } else {
            return sqrt(v);
        }
    }

    inline distance_t compare_normalized_l2(const uint64_t *__restrict a, const uint64_t *__restrict b, size_t size) {
        double v = 2.0 - 2.0 * polaris::primitive::compare_dot_product(a, b, size);
        if (v < 0.0) {
            return 0.0;
        } else {
            return sqrt(v);
        }
    }

    inline distance_t compare_normalized_l2(const float *__restrict a, const float *__restrict b, size_t size) {
        double v = 2.0 - 2.0 * polaris::primitive::compare_dot_product(a, b, size);
        if (v < 0.0) {
            return 0.0;
        } else {
            return sqrt(v);
        }
    }

    inline distance_t compare_normalized_l2(const double *__restrict a, const double *__restrict b, size_t size) {
        double v = 2.0 - 2.0 * polaris::primitive::compare_dot_product(a, b, size);
        if (v < 0.0) {
            return 0.0;
        } else {
            return sqrt(v);
        }
    }

    inline distance_t compare_normalized_l2(const float16 *__restrict a, const float16 *__restrict b, size_t size) {
        double v = 2.0 - 2.0 * polaris::primitive::compare_dot_product(a, b, size);
        if (v < 0.0) {
            return 0.0;
        } else {
            return sqrt(v);
        }
    }

    inline distance_t compare_normalized_l2(const bfloat16 *__restrict a, const bfloat16 *__restrict b, size_t size) {
        abort();
    }

    /// distance poincare
    template<typename OBJECT_TYPE>
    inline distance_t compare_poincare_distance(const OBJECT_TYPE *__restrict a, const OBJECT_TYPE *__restrict b, size_t size) {
        // Unlike the other distance functions, this is not optimized...
        double a2 = 0.0;
        double b2 = 0.0;
        double c2 = polaris::primitive::compare_l2(a, b, size);
        for (size_t i = 0; i < size; i++) {
            a2 += static_cast<double>(a[i]) * static_cast<double>(a[i]);
            b2 += static_cast<double>(b[i]) * static_cast<double>(b[i]);
        }
        return std::acosh(1 + 2.0 * c2 * c2 / (1.0 - a2) / (1.0 - b2));
    }

    /// distance lorentz
    template<typename OBJECT_TYPE>
    inline distance_t compare_lorentz_distance(const OBJECT_TYPE *__restrict a, const OBJECT_TYPE *__restrict b, size_t size) {
        // Unlike the other distance functions, this is not optimized...
        double sum = static_cast<double>(a[0]) * static_cast<double>(b[0]);
        for (size_t i = 1; i < size; i++) {
            sum -= static_cast<double>(a[i]) * static_cast<double>(b[i]);
        }
        return std::acosh(sum);
    }

    /// normalized cosine similarity

    inline distance_t compare_normalized_cosine_similarity(const int8_t * __restrict a, const int8_t *__restrict b, size_t size) {
        auto v = 1.0 - polaris::primitive::compare_dot_product(a, b, size);
        return v < 0.0 ? -v : v;
    }

    inline distance_t compare_normalized_cosine_similarity(const uint8_t * __restrict a, const uint8_t *__restrict b, size_t size) {
        auto v = 1.0 - polaris::primitive::compare_dot_product(a, b, size);
        return v < 0.0 ? -v : v;
    }

    inline distance_t compare_normalized_cosine_similarity(const int16_t * __restrict a, const int16_t *__restrict b, size_t size) {
        auto v = 1.0 - polaris::primitive::compare_dot_product(a, b, size);
        return v < 0.0 ? -v : v;
    }

    inline distance_t compare_normalized_cosine_similarity(const uint16_t * __restrict a, const uint16_t *__restrict b, size_t size) {
        auto v = 1.0 - polaris::primitive::compare_dot_product(a, b, size);
        return v < 0.0 ? -v : v;
    }

    inline distance_t compare_normalized_cosine_similarity(const int32_t * __restrict a, const int32_t *__restrict b, size_t size) {
        auto v = 1.0 - polaris::primitive::compare_dot_product(a, b, size);
        return v < 0.0 ? -v : v;
    }

    inline distance_t compare_normalized_cosine_similarity(const uint32_t * __restrict a, const uint32_t *__restrict b, size_t size) {
        auto v = 1.0 - polaris::primitive::compare_dot_product(a, b, size);
        return v < 0.0 ? -v : v;
    }

    inline distance_t compare_normalized_cosine_similarity(const int64_t * __restrict a, const int64_t *__restrict b, size_t size) {
        auto v = 1.0 - polaris::primitive::compare_dot_product(a, b, size);
        return v < 0.0 ? -v : v;
    }

    inline distance_t compare_normalized_cosine_similarity(const uint64_t * __restrict a, const uint64_t *__restrict b, size_t size) {
        auto v = 1.0 - polaris::primitive::compare_dot_product(a, b, size);
        return v < 0.0 ? -v : v;
    }

    inline distance_t compare_normalized_cosine_similarity(const float * __restrict a, const float *__restrict b, size_t size) {
        auto v = 1.0 - polaris::primitive::compare_dot_product(a, b, size);
        return v < 0.0 ? -v : v;
    }

    inline distance_t compare_normalized_cosine_similarity(const double * __restrict a, const double *__restrict b, size_t size) {
        auto v = 1.0 - polaris::primitive::compare_dot_product(a, b, size);
        return v < 0.0 ? -v : v;
    }

    inline distance_t compare_normalized_cosine_similarity(const float16 * __restrict a, const float16 *__restrict b, size_t size) {
        auto v = 1.0 - polaris::primitive::compare_dot_product(a, b, size);
        return v < 0.0 ? -v : v;
    }

    inline distance_t compare_normalized_cosine_similarity(const bfloat16 * __restrict a, const bfloat16 *__restrict b, size_t size) {
        abort();
    }


}  // namespace polaris::primitive

