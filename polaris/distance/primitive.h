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
    float PrimL1(const T *x, const T *y, size_t d) {
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

    template<typename T, typename Arch, typename Tag>
    float PrimIP(const T *x, const T *y, size_t d) {
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

    template<typename T, typename Arch, typename Tag>
    float PrimL2Sqr(const T *x, const T *y, size_t d) {
        using b_type = collie::simd::batch<T, Arch>;
        std::size_t inc = b_type::size;
        std::size_t vec_size = d - d % inc;
        float sum = 0.0;
        b_type sum_vec = collie::simd::broadcast(T(0));
        for (std::size_t i = 0; i < vec_size; i += inc) {
            b_type xvec = b_type::load(x + i, Tag());
            b_type yvec = b_type::load(y + i, Tag());
            auto delta = xvec - yvec;
            sum_vec += delta * delta;
        }
        sum = collie::simd::reduce_add(sum_vec);
        for (std::size_t i = vec_size; i < d; ++i) {
            auto delta = x[i] - y[i];
            sum += delta * delta;
        }
        return sum;
    }


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

    /// l2
    inline distance_t compare_l2(const float *__restrict a, const float *__restrict b, size_t d) {
        using b_type = collie::simd::batch<float, collie::simd::best_arch>;
        std::size_t inc = b_type::size;
        std::size_t vec_size = d - d % inc;
        float sum = 0.0;
        b_type sum_vec = collie::simd::broadcast(0.0f);
        for (std::size_t i = 0; i < vec_size; i += inc) {
            b_type xvec = b_type::load(a + i, collie::simd::aligned_mode());
            b_type yvec = b_type::load(a + i, collie::simd::aligned_mode());
            auto delta = xvec - yvec;
            sum_vec += delta * delta;
        }
        sum = collie::simd::reduce_add(sum_vec);
        for (std::size_t i = vec_size; i < d; ++i) {
            auto delta = a[i] - a[i];
            sum += delta * delta;
        }
        return std::sqrt(sum);
    }

#if COLLIE_SIMD_AVX512F
    inline static float compare_l2(const bfloat16 *__restrict a, const bfloat16 *__restrict b, size_t size) {
        const bfloat16 *last = a + size;
        __m512 sum512 = _mm512_setzero_ps();
        while (a < last) {
            __m512i av = _mm512_slli_epi32(_mm512_cvtepu16_epi32(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(a))), 16);
            __m512i bv = _mm512_slli_epi32(_mm512_cvtepu16_epi32(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(b))), 16);
            __m512 sub = _mm512_sub_ps(reinterpret_cast<__m512>(av), reinterpret_cast<__m512>(bv));
            sum512 = _mm512_fmadd_ps(sub, sub, sum512);
            a += 16;
            b += 16;
        }
        __m256 sum256 = _mm256_add_ps(_mm512_extractf32x8_ps(sum512, 0), _mm512_extractf32x8_ps(sum512, 1));
        __m128 sum128 = _mm_add_ps(_mm256_extractf128_ps(sum256, 0), _mm256_extractf128_ps(sum256, 1));
        __m128 tmp = _mm_hadd_ps(sum128, _mm_set1_ps(0));
        double d = _mm_cvtss_f32(_mm_shuffle_ps(tmp, tmp, _MM_SHUFFLE(0, 0, 0, 0))) +
                   _mm_cvtss_f32(_mm_shuffle_ps(tmp, tmp, _MM_SHUFFLE(0, 0, 0, 1)));
        //return sqrt(d);
        return d;
    }
#endif

    inline distance_t compare_l2(const float16 *__restrict a, const float16 *__restrict b, size_t size) {
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
        return sqrt(s);
    }

    inline distance_t compare_l2(const unsigned char *__restrict a, const unsigned char *__restrict b, size_t size) {
        __m128 sum = _mm_setzero_ps();
        const unsigned char *last = a + size;
        const unsigned char *lastgroup = last - 7;
        const __m128i zero = _mm_setzero_si128();
        while (a < lastgroup) {
            //__m128i x1 = _mm_cvtepu8_epi16(*reinterpret_cast<__m128i const*>(a));
            __m128i x1 = _mm_cvtepu8_epi16(_mm_loadu_si128((__m128i const *) a));
            //__m128i x2 = _mm_cvtepu8_epi16(*reinterpret_cast<__m128i const*>(b));
            __m128i x2 = _mm_cvtepu8_epi16(_mm_loadu_si128((__m128i const *) b));
            x1 = _mm_subs_epi16(x1, x2);
            __m128i v = _mm_mullo_epi16(x1, x1);
            sum = _mm_add_ps(sum, _mm_cvtepi32_ps(_mm_unpacklo_epi16(v, zero)));
            sum = _mm_add_ps(sum, _mm_cvtepi32_ps(_mm_unpackhi_epi16(v, zero)));
            a += 8;
            b += 8;
        }
        __attribute__((aligned(32))) float f[4];
        _mm_store_ps(f, sum);
        double s = f[0] + f[1] + f[2] + f[3];
        while (a < last) {
            int d = (int) *a++ - (int) *b++;
            s += d * d;
        }
        return sqrt(s);
    }
    /// l1
    inline distance_t compare_l1(const float * __restrict a, const float *__restrict b, size_t size) {
        __m256 sum = _mm256_setzero_ps();
        const float *last = a + size;
        const float *lastgroup = last - 7;
        while (a < lastgroup) {
            __m256 x1 = _mm256_sub_ps(_mm256_loadu_ps(a), _mm256_loadu_ps(b));
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

    inline distance_t compare_l1(const unsigned char *__restrict a, const unsigned char *__restrict b, size_t size) {
        __m128 sum = _mm_setzero_ps();
        const unsigned char *last = a + size;
        const unsigned char *lastgroup = last - 7;
        const __m128i zero = _mm_setzero_si128();
        while (a < lastgroup) {
            __m128i x1 = _mm_cvtepu8_epi16(*reinterpret_cast<__m128i const *>(a));
            __m128i x2 = _mm_cvtepu8_epi16(*reinterpret_cast<__m128i const *>(b));
            x1 = _mm_subs_epi16(x1, x2);
            x1 = _mm_sign_epi16(x1, x1);
            sum = _mm_add_ps(sum, _mm_cvtepi32_ps(_mm_unpacklo_epi16(x1, zero)));
            sum = _mm_add_ps(sum, _mm_cvtepi32_ps(_mm_unpackhi_epi16(x1, zero)));
            a += 8;
            b += 8;
        }
        __attribute__((aligned(32))) float f[4];
        _mm_store_ps(f, sum);
        distance_t s = f[0] + f[1] + f[2] + f[3];
        while (a < last) {
            distance_t d = std::fabs(static_cast<double>(*a++) - static_cast<double>(*b++));
            s += d;
        }
        return s;
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
        inline distance_t compare_hamming_distance(const OBJECT_TYPE *a, const OBJECT_TYPE *b, size_t size) {
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
    inline distance_t compare_hamming_distance(const OBJECT_TYPE *a, const OBJECT_TYPE *b, size_t size) {
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
          inline distance_t compare_jaccard_distance(const OBJECT_TYPE *a, const OBJECT_TYPE *b, size_t size) {
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
    inline distance_t compare_jaccard_distance(const OBJECT_TYPE *a, const OBJECT_TYPE *b, size_t size) {
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
    inline distance_t compare_sparse_jaccard_distance(const unsigned char *a, const unsigned char *b, size_t size) {
        std::cerr << "compare_sparse_jaccard_distance: Not implemented." << std::endl;
        abort();
    }

    inline distance_t compare_sparse_jaccard_distance(const float16 *a, const float16 *b, size_t size) {
        std::cerr << "compare_sparse_jaccard_distance: Not implemented." << std::endl;
        abort();
    }

    inline distance_t compare_sparse_jaccard_distance(const bfloat16 *a, const bfloat16 *b, size_t size) {
        std::cerr << "compare_sparse_jaccard_distance: Not implemented." << std::endl;
        abort();
    }

    inline distance_t compare_sparse_jaccard_distance(const float *a, const float *b, size_t size) {
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
    inline distance_t compare_dot_product(const float * __restrict a, const float *__restrict b, size_t size) {
        const float *last = a + size;
#if defined(NGT_AVX512)
        __m512 sum512 = _mm512_setzero_ps();
            while (a < last) {
          sum512 = _mm512_add_ps(sum512, _mm512_mul_ps(_mm512_loadu_ps(a), _mm512_loadu_ps(b)));
          a += 16;
          b += 16;
            }
            __m256 sum256 = _mm256_add_ps(_mm512_extractf32x8_ps(sum512, 0), _mm512_extractf32x8_ps(sum512, 1));
            __m128 sum128 = _mm_add_ps(_mm256_extractf128_ps(sum256, 0), _mm256_extractf128_ps(sum256, 1));
#elif defined(NGT_AVX2)
        __m256 sum256 = _mm256_setzero_ps();
        while (a < last) {
            sum256 = _mm256_add_ps(sum256, _mm256_mul_ps(_mm256_loadu_ps(a), _mm256_loadu_ps(b)));
            a += 8;
            b += 8;
        }
        __m128 sum128 = _mm_add_ps(_mm256_extractf128_ps(sum256, 0), _mm256_extractf128_ps(sum256, 1));
#else
        __m128 sum128 = _mm_setzero_ps();
            while (a < last) {
          sum128 = _mm_add_ps(sum128, _mm_mul_ps(_mm_loadu_ps(a), _mm_loadu_ps(b)));
          a += 4;
          b += 4;
            }
#endif
        __attribute__((aligned(32))) float f[4];
        _mm_store_ps(f, sum128);
        double s = static_cast<double>(f[0]) + static_cast<double>(f[1]) + static_cast<double>(f[2]) +
                   static_cast<double>(f[3]);
        return s;
    }

    inline distance_t compare_dot_product(const float16 *__restrict a, const float16 *__restrict b, size_t size) {
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

    inline distance_t compare_dot_product(const bfloat16 *__restrict a, const bfloat16 *__restrict b, size_t size) {
        abort();
    }

    inline distance_t compare_dot_product(const unsigned char *__restrict a, const unsigned char *__restrict b, size_t size) {
        double sum = 0.0;
        for (size_t loc = 0; loc < size; loc++) {
            sum += static_cast<double>(a[loc]) * static_cast<double>(b[loc]);
        }
        return sum;
    }

    /// cosine
    inline distance_t compare_cosine(const float *a, const float *b, size_t size) {

        const float *last = a + size;
#if defined(NGT_AVX512)
        __m512 normA = _mm512_setzero_ps();
            __m512 normB = _mm512_setzero_ps();
            __m512 sum = _mm512_setzero_ps();
            while (a < last) {
          __m512 am = _mm512_loadu_ps(a);
          __m512 bm = _mm512_loadu_ps(b);
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
            am = _mm256_loadu_ps(a);
            bm = _mm256_loadu_ps(b);
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
          am = _mm_loadu_ps(a);
          bm = _mm_loadu_ps(b);
          am128 = _mm_add_ps(am128, _mm_mul_ps(am, am));
          bm128 = _mm_add_ps(bm128, _mm_mul_ps(bm, bm));
          s128 = _mm_add_ps(s128, _mm_mul_ps(am, bm));
          a += 4;
          b += 4;
            }

#endif

        __attribute__((aligned(32))) float f[4];
        _mm_store_ps(f, am128);
        distance_t na = f[0] + f[1] + f[2] + f[3];
        _mm_store_ps(f, bm128);
        distance_t nb = f[0] + f[1] + f[2] + f[3];
        _mm_store_ps(f, s128);
        distance_t s = f[0] + f[1] + f[2] + f[3];

        distance_t cosine = s / std::sqrt(na * nb);
        return cosine;
    }

    inline distance_t compare_cosine(const float16 *a, const float16 *b, size_t size) {

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

    inline distance_t compare_cosine(const bfloat16 *a, const bfloat16 *b, size_t size) {
        abort();
    }

    inline distance_t compare_cosine(const unsigned char *a, const unsigned char *b, size_t size) {
        double normA = 0.0;
        double normB = 0.0;
        distance_t sum = 0.0;
        for (size_t loc = 0; loc < size; loc++) {
            normA += static_cast<double>(a[loc]) * static_cast<double>(a[loc]);
            normB += static_cast<double>(b[loc]) * static_cast<double>(b[loc]);
            sum += static_cast<double>(a[loc]) * static_cast<double>(b[loc]);
        }

        double cosine = sum / sqrt(normA * normB);

        return cosine;
    }

    template<typename OBJECT_TYPE>
    inline distance_t compare_angle_distance(const OBJECT_TYPE *a, const OBJECT_TYPE *b, size_t size) {
        distance_t cosine = compare_cosine(a, b, size);
        if (cosine >= 1.0) {
            return 0.0;
        } else if (cosine <= -1.0) {
            return std::acos(-1.0);
        } else {
            return std::acos(cosine);
        }
    }

    template<typename OBJECT_TYPE>
    inline distance_t compare_normalized_angle_distance(const OBJECT_TYPE *a, const OBJECT_TYPE *b, size_t size) {
        distance_t cosine = polaris::primitive::compare_dot_product(a, b, size);
        if (cosine >= 1.0) {
            return 0.0;
        } else if (cosine <= -1.0) {
            return acos(-1.0);
        } else {
            return acos(cosine);
        }
    }

    template<typename OBJECT_TYPE>
    inline distance_t compare_normalized_l2(const OBJECT_TYPE *a, const OBJECT_TYPE *b, size_t size) {
        double v = 2.0 - 2.0 * polaris::primitive::compare_dot_product(a, b, size);
        if (v < 0.0) {
            return 0.0;
        } else {
            return sqrt(v);
        }
    }

    template<typename OBJECT_TYPE>
    inline distance_t compare_poincare_distance(const OBJECT_TYPE *a, const OBJECT_TYPE *b, size_t size) {
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

    template<typename OBJECT_TYPE>
    inline distance_t compare_lorentz_distance(const OBJECT_TYPE *a, const OBJECT_TYPE *b, size_t size) {
        // Unlike the other distance functions, this is not optimized...
        double sum = static_cast<double>(a[0]) * static_cast<double>(b[0]);
        for (size_t i = 1; i < size; i++) {
            sum -= static_cast<double>(a[i]) * static_cast<double>(b[i]);
        }
        return std::acosh(sum);
    }

    template<typename OBJECT_TYPE>
    inline distance_t compare_cosine_similarity(const OBJECT_TYPE *a, const OBJECT_TYPE *b, size_t size) {
        auto v = 1.0 - compare_cosine(a, b, size);
        return v < 0.0 ? -v : v;
    }

    template<typename OBJECT_TYPE>
    inline distance_t
    compare_normalized_cosine_similarity(const OBJECT_TYPE *a, const OBJECT_TYPE *b, size_t size) {
        auto v = 1.0 - polaris::primitive::compare_dot_product(a, b, size);
        return v < 0.0 ? -v : v;
    }

}  // namespace polaris::primitive

