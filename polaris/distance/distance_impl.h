#pragma once

#include <polaris/distance/distance.h>

namespace polaris {

    class DistanceCosineInt8 : public Distance<int8_t> {
    public:
        DistanceCosineInt8() : Distance<int8_t>(polaris::MetricType::METRIC_COSINE) {
        }

        POLARIS_API virtual float compare(const int8_t *a, const int8_t *b, uint32_t length) const;
    };

    class DistanceL2Int8 : public Distance<int8_t> {
    public:
        DistanceL2Int8() : Distance<int8_t>(polaris::MetricType::METRIC_L2) {
        }

        POLARIS_API virtual float compare(const int8_t *a, const int8_t *b, uint32_t size) const;
    };

    // AVX implementations. Borrowed from HNSW code.
    class AVXDistanceL2Int8 : public Distance<int8_t> {
    public:
        AVXDistanceL2Int8() : Distance<int8_t>(polaris::MetricType::METRIC_L2) {
        }

        POLARIS_API virtual float compare(const int8_t *a, const int8_t *b, uint32_t length) const;
    };

    class DistanceCosineFloat : public Distance<float> {
    public:
        DistanceCosineFloat() : Distance<float>(polaris::MetricType::METRIC_COSINE) {
        }

        POLARIS_API virtual float compare(const float *a, const float *b, uint32_t length) const;
    };

    class DistanceL2Float : public Distance<float> {
    public:
        DistanceL2Float() : Distance<float>(polaris::MetricType::METRIC_L2) {
        }

        POLARIS_API virtual float compare(const float *a, const float *b, uint32_t size) const __attribute__((hot));

    };

    class AVXDistanceL2Float : public Distance<float> {
    public:
        AVXDistanceL2Float() : Distance<float>(polaris::MetricType::METRIC_L2) {
        }

        POLARIS_API virtual float compare(const float *a, const float *b, uint32_t length) const;
    };

    template<typename T>
    class SlowDistanceL2 : public Distance<T> {
    public:
        SlowDistanceL2() : Distance<T>(polaris::MetricType::METRIC_L2) {
        }

        POLARIS_API virtual float compare(const T *a, const T *b, uint32_t length) const;
    };

    class SlowDistanceCosineUInt8 : public Distance<uint8_t> {
    public:
        SlowDistanceCosineUInt8() : Distance<uint8_t>(polaris::MetricType::METRIC_COSINE) {
        }

        POLARIS_API virtual float compare(const uint8_t *a, const uint8_t *b, uint32_t length) const;
    };

    class DistanceL2UInt8 : public Distance<uint8_t> {
    public:
        DistanceL2UInt8() : Distance<uint8_t>(polaris::MetricType::METRIC_L2) {
        }

        POLARIS_API virtual float compare(const uint8_t *a, const uint8_t *b, uint32_t size) const;
    };

    template<typename T>
    class DistanceInnerProduct : public Distance<T> {
    public:
        DistanceInnerProduct() : Distance<T>(polaris::MetricType::METRIC_INNER_PRODUCT) {
        }

        DistanceInnerProduct(polaris::MetricType metric) : Distance<T>(metric) {
        }

        inline float inner_product(const T *a, const T *b, unsigned size) const;

        inline float compare(const T *a, const T *b, unsigned size) const {
            float result = inner_product(a, b, size);
            //      if (result < 0)
            //      return std::numeric_limits<float>::max();
            //      else
            return -result;
        }
    };

    class AVXDistanceInnerProductFloat : public Distance<float> {
    public:
        AVXDistanceInnerProductFloat() : Distance<float>(polaris::MetricType::METRIC_INNER_PRODUCT) {
        }

        POLARIS_API virtual float compare(const float *a, const float *b, uint32_t length) const;
    };

    class AVXNormalizedCosineDistanceFloat : public Distance<float> {
    private:
        AVXDistanceInnerProductFloat _innerProduct;

    protected:
        void normalize_and_copy(const float *a, uint32_t length, float *a_norm) const;

    public:
        AVXNormalizedCosineDistanceFloat() : Distance<float>(polaris::MetricType::METRIC_COSINE) {
        }

        POLARIS_API virtual float compare(const float *a, const float *b, uint32_t length) const {
            // Inner product returns negative values to indicate distance.
            // This will ensure that cosine is between -1 and 1.
            return 1.0f + _innerProduct.compare(a, b, length);
        }

        POLARIS_API virtual void preprocess_query(const float *query_vec, const size_t query_dim,
                                                  float *scratch_query_vector) override;
    };

} // namespace polaris
