// TODO
// CHECK COSINE ON LINUX

#ifdef _WINDOWS
#include <immintrin.h>
#include <smmintrin.h>
#include <tmmintrin.h>
#include <intrin.h>
#else

#include <immintrin.h>

#endif

#include <polaris/distance/simd_utils.h>
#include <polaris/distance/cosine_similarity.h>
#include <iostream>

#include <polaris/distance/distance_impl.h>
#include <polaris/distance/norm.h>
#include <polaris/distance/primitive.h>
#include <polaris/graph/vamana/utils.h>
#include <polaris/graph/vamana/logger.h>
#include <polaris/utility/polaris_exception.h>
#define USE_AVX2
namespace polaris {

    //
    // Base Class Implementatons
    //
    /*
    template<typename T>
    float Distance<T>::compare(const T *a, const T *b, const float normA, const float normB, uint32_t length) const {
        throw std::logic_error("This function is not implemented.");
    }
    */
    template<typename T>
    polaris::MetricType Distance<T>::get_metric() const {
        return _distance_metric;
    }

    template<typename T>
    void Distance<T>::preprocess_query(const T *query_vec, const size_t query_dim, T *scratch_query) {
        std::memcpy(scratch_query, query_vec, query_dim * sizeof(T));
    }

    template<typename T>
    size_t Distance<T>::get_required_alignment() const {
        return _alignment_factor;
    }

    //
    // Cosine distance functions.
    //

    float DistanceCosineInt8::compare(const int8_t *a, const int8_t *b, uint32_t length) const {
        int magA = 0, magB = 0, scalarProduct = 0;
        for (uint32_t i = 0; i < length; i++) {
            magA += ((int32_t) a[i]) * ((int32_t) a[i]);
            magB += ((int32_t) b[i]) * ((int32_t) b[i]);
            scalarProduct += ((int32_t) a[i]) * ((int32_t) b[i]);
        }
        // similarity == 1-cosine distance
        return 1.0f - (float) (scalarProduct / (sqrt(magA) * sqrt(magB)));
    }

    float DistanceCosineFloat::compare(const float *a, const float *b, uint32_t length) const {
        float magA = 0, magB = 0, scalarProduct = 0;
        for (uint32_t i = 0; i < length; i++) {
            magA += (a[i]) * (a[i]);
            magB += (b[i]) * (b[i]);
            scalarProduct += (a[i]) * (b[i]);
        }
        // similarity == 1-cosine distance
        return 1.0f - (scalarProduct / (sqrt(magA) * sqrt(magB)));
    }

    float SlowDistanceCosineUInt8::compare(const uint8_t *a, const uint8_t *b, uint32_t length) const {
        int magA = 0, magB = 0, scalarProduct = 0;
        for (uint32_t i = 0; i < length; i++) {
            magA += ((uint32_t) a[i]) * ((uint32_t) a[i]);
            magB += ((uint32_t) b[i]) * ((uint32_t) b[i]);
            scalarProduct += ((uint32_t) a[i]) * ((uint32_t) b[i]);
        }
        // similarity == 1-cosine distance
        return 1.0f - (float) (scalarProduct / (sqrt(magA) * sqrt(magB)));
    }

    //
    // L2 distance functions.
    //

    float DistanceL2Int8::compare(const int8_t *a, const int8_t *b, uint32_t size) const {
        int32_t result = 0;
#pragma omp simd reduction(+ : result) aligned(a, b : 8)
        for (int32_t i = 0; i < (int32_t) size; i++) {
            result += ((int32_t) ((int16_t) a[i] - (int16_t) b[i])) * ((int32_t) ((int16_t) a[i] - (int16_t) b[i]));
        }
        return (float) result;
    }

    float DistanceL2UInt8::compare(const uint8_t *a, const uint8_t *b, uint32_t size) const {
        uint32_t result = 0;
#ifndef _WINDOWS
#pragma omp simd reduction(+ : result) aligned(a, b : 8)
#endif
        for (int32_t i = 0; i < (int32_t) size; i++) {
            result += ((int32_t) ((int16_t) a[i] - (int16_t) b[i])) * ((int32_t) ((int16_t) a[i] - (int16_t) b[i]));
        }
        return (float) result;
    }

    float DistanceL2Float::compare(const float *__restrict a, const float *__restrict b, uint32_t size) const {
        return primitive::compare_template_l2_sqr<float, collie::simd::best_arch, collie::simd::aligned_mode>(
                a, b, size);
    }

    template<typename T>
    float SlowDistanceL2<T>::compare(const T *a, const T *b, uint32_t length) const {
        float result = 0.0f;
        for (uint32_t i = 0; i < length; i++) {
            result += ((float) (a[i] - b[i])) * (a[i] - b[i]);
        }
        return result;
    }

    float AVXDistanceL2Int8::compare(const int8_t *, const int8_t *, uint32_t) const {
        return 0;
    }

    float AVXDistanceL2Float::compare(const float *, const float *, uint32_t) const {
        return 0;
    }

    template<typename T>
    float DistanceInnerProduct<T>::inner_product(const T *a, const T *b, uint32_t size) const {
        if (!std::is_floating_point<T>::value) {
            polaris::cerr << "ERROR: Inner Product only defined for float currently." << std::endl;
            throw polaris::PolarisException("ERROR: Inner Product only defined for float currently.", -1,
                                            __PRETTY_FUNCTION__, __FILE__,
                                            __LINE__);
        }

        float result = 0;

#ifdef __GNUC__
#ifdef USE_AVX2
#define AVX_DOT(addr1, addr2, dest, tmp1, tmp2)                                                                        \
    tmp1 = _mm256_loadu_ps(addr1);                                                                                     \
    tmp2 = _mm256_loadu_ps(addr2);                                                                                     \
    tmp1 = _mm256_mul_ps(tmp1, tmp2);                                                                                  \
    dest = _mm256_add_ps(dest, tmp1);

        __m256 sum;
        __m256 l0, l1;
        __m256 r0, r1;
        uint32_t D = (size + 7) & ~7U;
        uint32_t DR = D % 16;
        uint32_t DD = D - DR;
        const float *l = (float *) a;
        const float *r = (float *) b;
        const float *e_l = l + DD;
        const float *e_r = r + DD;
        float unpack[8] __attribute__((aligned(32))) = {0, 0, 0, 0, 0, 0, 0, 0};

        sum = _mm256_loadu_ps(unpack);
        if (DR) {
            AVX_DOT(e_l, e_r, sum, l0, r0);
        }

        for (uint32_t i = 0; i < DD; i += 16, l += 16, r += 16) {
            AVX_DOT(l, r, sum, l0, r0);
            AVX_DOT(l + 8, r + 8, sum, l1, r1);
        }
        _mm256_storeu_ps(unpack, sum);
        result = unpack[0] + unpack[1] + unpack[2] + unpack[3] + unpack[4] + unpack[5] + unpack[6] + unpack[7];

#else
#ifdef __SSE2__
#define SSE_DOT(addr1, addr2, dest, tmp1, tmp2)                                                                        \
    tmp1 = _mm128_loadu_ps(addr1);                                                                                     \
    tmp2 = _mm128_loadu_ps(addr2);                                                                                     \
    tmp1 = _mm128_mul_ps(tmp1, tmp2);                                                                                  \
    dest = _mm128_add_ps(dest, tmp1);
        __m128 sum;
        __m128 l0, l1, l2, l3;
        __m128 r0, r1, r2, r3;
        uint32_t D = (size + 3) & ~3U;
        uint32_t DR = D % 16;
        uint32_t DD = D - DR;
        const float *l = a;
        const float *r = b;
        const float *e_l = l + DD;
        const float *e_r = r + DD;
        float unpack[4] __attribute__((aligned(16))) = {0, 0, 0, 0};

        sum = _mm_load_ps(unpack);
        switch (DR)
        {
        case 12:
            SSE_DOT(e_l + 8, e_r + 8, sum, l2, r2);
        case 8:
            SSE_DOT(e_l + 4, e_r + 4, sum, l1, r1);
        case 4:
            SSE_DOT(e_l, e_r, sum, l0, r0);
        default:
            break;
        }
        for (uint32_t i = 0; i < DD; i += 16, l += 16, r += 16)
        {
            SSE_DOT(l, r, sum, l0, r0);
            SSE_DOT(l + 4, r + 4, sum, l1, r1);
            SSE_DOT(l + 8, r + 8, sum, l2, r2);
            SSE_DOT(l + 12, r + 12, sum, l3, r3);
        }
        _mm_storeu_ps(unpack, sum);
        result += unpack[0] + unpack[1] + unpack[2] + unpack[3];
#else

        float dot0, dot1, dot2, dot3;
        const float *last = a + size;
        const float *unroll_group = last - 3;

        /* Process 4 items with each loop for efficiency. */
        while (a < unroll_group)
        {
            dot0 = a[0] * b[0];
            dot1 = a[1] * b[1];
            dot2 = a[2] * b[2];
            dot3 = a[3] * b[3];
            result += dot0 + dot1 + dot2 + dot3;
            a += 4;
            b += 4;
        }
        /* Process last 0-3 pixels.  Not needed for standard vector lengths. */
        while (a < last)
        {
            result += *a++ * *b++;
        }
#endif
#endif
#endif
        return result;
    }

    float AVXDistanceInnerProductFloat::compare(const float *a, const float *b, uint32_t size) const {
        float result = 0.0f;
#define AVX_DOT(addr1, addr2, dest, tmp1, tmp2)                                                                        \
    tmp1 = _mm256_loadu_ps(addr1);                                                                                     \
    tmp2 = _mm256_loadu_ps(addr2);                                                                                     \
    tmp1 = _mm256_mul_ps(tmp1, tmp2);                                                                                  \
    dest = _mm256_add_ps(dest, tmp1);

        __m256 sum;
        __m256 l0, l1;
        __m256 r0, r1;
        uint32_t D = (size + 7) & ~7U;
        uint32_t DR = D % 16;
        uint32_t DD = D - DR;
        const float *l = (float *) a;
        const float *r = (float *) b;
        const float *e_l = l + DD;
        const float *e_r = r + DD;
        float unpack[8] __attribute__((aligned(32))) = {0, 0, 0, 0, 0, 0, 0, 0};

        sum = _mm256_loadu_ps(unpack);
        if (DR) {
            AVX_DOT(e_l, e_r, sum, l0, r0);
        }

        for (uint32_t i = 0; i < DD; i += 16, l += 16, r += 16) {
            AVX_DOT(l, r, sum, l0, r0);
            AVX_DOT(l + 8, r + 8, sum, l1, r1);
        }
        _mm256_storeu_ps(unpack, sum);
        result = unpack[0] + unpack[1] + unpack[2] + unpack[3] + unpack[4] + unpack[5] + unpack[6] + unpack[7];

        return -result;
    }

    void AVXNormalizedCosineDistanceFloat::preprocess_query(const float *query_vec, const size_t query_dim,
                                                            float *query_scratch) {
        normalize_and_copy(query_vec, (uint32_t) query_dim, query_scratch);
    }

    void AVXNormalizedCosineDistanceFloat::normalize_and_copy(const float *query_vec, const uint32_t query_dim,
                                                              float *query_target) const {
        float norm = get_norm(query_vec, query_dim);

        for (uint32_t i = 0; i < query_dim; i++) {
            query_target[i] = query_vec[i] / norm;
        }
    }

// Get the right distance function for the given metric.
    template<>
    polaris::Distance<float> *get_distance_function(polaris::MetricType m) {
        if (m == polaris::MetricType::METRIC_L2) {
            if (Avx2SupportedCPU) {
                polaris::cout << "L2: Using AVX2 distance computation DistanceL2Float" << std::endl;
                return new polaris::DistanceL2Float();
            } else if (AvxSupportedCPU) {
                polaris::cout << "L2: AVX2 not supported. Using AVX distance computation" << std::endl;
                return new polaris::AVXDistanceL2Float();
            } else {
                polaris::cout << "L2: Older CPU. Using slow distance computation" << std::endl;
                return new polaris::SlowDistanceL2<float>();
            }
        } else if (m == polaris::MetricType::METRIC_COSINE) {
            polaris::cout << "Cosine: Using either AVX or AVX2 implementation" << std::endl;
            return new polaris::DistanceCosineFloat();
        } else if (m == polaris::MetricType::METRIC_INNER_PRODUCT) {
            polaris::cout << "Inner product: Using AVX2 implementation "
                             "AVXDistanceInnerProductFloat"
                          << std::endl;
            return new polaris::AVXDistanceInnerProductFloat();
        } else {
            std::stringstream stream;
            stream << "Only L2, cosine, and inner product supported for floating "
                      "point vectors as of now."
                   << std::endl;
            polaris::cerr << stream.str() << std::endl;
            throw polaris::PolarisException(stream.str(), -1, __PRETTY_FUNCTION__, __FILE__, __LINE__);
        }
    }

    template<>
    polaris::Distance<int8_t> *get_distance_function(polaris::MetricType m) {
        if (m == polaris::MetricType::METRIC_L2) {
            if (Avx2SupportedCPU) {
                polaris::cout << "Using AVX2 distance computation DistanceL2Int8." << std::endl;
                return new polaris::DistanceL2Int8();
            } else if (AvxSupportedCPU) {
                polaris::cout << "AVX2 not supported. Using AVX distance computation" << std::endl;
                return new polaris::AVXDistanceL2Int8();
            } else {
                polaris::cout << "Older CPU. Using slow distance computation "
                                 "SlowDistanceL2Int<int8_t>."
                              << std::endl;
                return new polaris::SlowDistanceL2<int8_t>();
            }
        } else if (m == polaris::MetricType::METRIC_COSINE) {
            polaris::cout << "Using either AVX or AVX2 for Cosine similarity "
                             "DistanceCosineInt8."
                          << std::endl;
            return new polaris::DistanceCosineInt8();
        } else {
            std::stringstream stream;
            stream << "Only L2 and cosine supported for signed byte vectors." << std::endl;
            polaris::cerr << stream.str() << std::endl;
            throw polaris::PolarisException(stream.str(), -1, __PRETTY_FUNCTION__, __FILE__, __LINE__);
        }
    }

    template<>
    polaris::Distance<uint8_t> *get_distance_function(polaris::MetricType m) {
        if (m == polaris::MetricType::METRIC_L2) {
            return new polaris::DistanceL2UInt8();
        } else if (m == polaris::MetricType::METRIC_COSINE) {
            polaris::cout << "AVX/AVX2 distance function not defined for Uint8. Using "
                             "slow version SlowDistanceCosineUint8() "
                             "Contact gopalsr@microsoft.com if you need AVX/AVX2 support."
                          << std::endl;
            return new polaris::SlowDistanceCosineUInt8();
        } else {
            std::stringstream stream;
            stream << "Only L2 and cosine supported for uint32_t byte vectors." << std::endl;
            polaris::cerr << stream.str() << std::endl;
            throw polaris::PolarisException(stream.str(), -1, __PRETTY_FUNCTION__, __FILE__, __LINE__);
        }
    }

    template POLARIS_API
    class DistanceInnerProduct<float>;

    template POLARIS_API
    class DistanceInnerProduct<int8_t>;

    template POLARIS_API
    class DistanceInnerProduct<uint8_t>;

    template POLARIS_API
    class SlowDistanceL2<float>;

    template POLARIS_API
    class SlowDistanceL2<int8_t>;

    template POLARIS_API
    class SlowDistanceL2<uint8_t>;

    template POLARIS_API Distance<float> *get_distance_function(MetricType m);

    template POLARIS_API Distance<int8_t> *get_distance_function(MetricType m);

    template POLARIS_API Distance<uint8_t> *get_distance_function(MetricType m);

} // namespace polaris
