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

#include <polaris/core/common.h>
#include <polaris/core/array_view.h>
#include <polaris/distance/primitive_comparator.h>

namespace polaris {

    class DistanceComparator {
    public:
        POLARIS_API explicit DistanceComparator(size_t d) : dimension(d) {}

        POLARIS_API virtual float operator()(const ArrayView &a, const ArrayView &b) = 0;

        POLARIS_API [[nodiscard]] virtual MetricType get_metric() const = 0;

        POLARIS_API virtual ~DistanceComparator() = default;

        size_t dimension;
    };

    template<typename OBJECT_TYPE>
    class ComparatorL1 : public DistanceComparator {
    public:
        explicit ComparatorL1(size_t d) : DistanceComparator(d) {}

        float operator()(const ArrayView &a, const ArrayView &b)  override {
            return polaris::primitive::compare_l1((const OBJECT_TYPE *) a.data(), (const OBJECT_TYPE *) b.data(), dimension);
        }

        [[nodiscard]] MetricType get_metric() const override {
            return MetricType::METRIC_L1;
        }
    };

    template<typename OBJECT_TYPE>
    class ComparatorL2 : public DistanceComparator {
    public:
        explicit ComparatorL2(size_t d) : DistanceComparator(d) {}

        float operator()(const ArrayView &a, const ArrayView &b) override {
            return polaris::primitive::compare_l2((OBJECT_TYPE *) a.data(), (OBJECT_TYPE *) b.data(),
                                                  dimension);
        }
        [[nodiscard]] MetricType get_metric() const override {
            return MetricType::METRIC_L2;
        }
    };

    /*
    template<typename OBJECT_TYPE>
    class ComparatorFastL2 : public DistanceComparator {
    public:
        explicit ComparatorFastL2(size_t d) : DistanceComparator(d) {}

        float operator()(const ArrayView &a, const ArrayView &b) override {

            auto ip= polaris::primitive::compare_dot_product((OBJECT_TYPE *) a.data(), (OBJECT_TYPE *) b.data(),
                                                           dimension);
            auto v = a.l2_norm_sq() + b.l2_norm_sq() - 2 * ip;
            return v <= 0.0 ? 0.0 : sqrt(v);
        }
        [[nodiscard]] MetricType get_metric() const override {
            return MetricType::METRIC_FAST_L2;
        }
    };*/

    template<typename OBJECT_TYPE>
    class ComparatorNormalizedL2 : public DistanceComparator {
    public:
        explicit ComparatorNormalizedL2(size_t d) : DistanceComparator(d) {}

        float operator()(const ArrayView &a, const ArrayView &b) override {
            return polaris::primitive::compare_normalized_l2((OBJECT_TYPE *) a.data(),
                                                             (OBJECT_TYPE *) b.data(), dimension);
        }
        [[nodiscard]] MetricType get_metric() const override {
            return MetricType::METRIC_NORMALIZED_L2;
        }
    };

    template<typename OBJECT_TYPE>
    class ComparatorHammingDistance : public DistanceComparator {
    public:
        explicit ComparatorHammingDistance(size_t d) : DistanceComparator(d) {}

        float operator()(const ArrayView &a, const ArrayView &b) override {
            return polaris::primitive::compare_hamming_distance((OBJECT_TYPE *) a.data(),
                                                                (OBJECT_TYPE *) b.data(), dimension);
        }
        [[nodiscard]] MetricType get_metric() const override {
            return MetricType::METRIC_HAMMING;
        }
    };


    template<typename OBJECT_TYPE>
    class ComparatorJaccardDistance : public DistanceComparator {
    public:
        explicit ComparatorJaccardDistance(size_t d) : DistanceComparator(d) {}

        float operator()(const ArrayView &a, const ArrayView &b) override {
            return polaris::primitive::compare_jaccard_distance((OBJECT_TYPE *) a.data(),
                                                                (OBJECT_TYPE *) b.data(), dimension);
        }
        [[nodiscard]] MetricType get_metric() const override {
            return MetricType::METRIC_JACCARD;
        }
    };

    template<typename OBJECT_TYPE>
    class ComparatorSparseJaccardDistance : public DistanceComparator {
    public:
        explicit ComparatorSparseJaccardDistance(size_t d) : DistanceComparator(d) {}

        float operator()(const ArrayView &a, const ArrayView &b) override {
            return polaris::primitive::compare_sparse_jaccard_distance((OBJECT_TYPE *) a.data(),
                                                                       (OBJECT_TYPE *) b.data(), dimension);
        }

        [[nodiscard]] MetricType get_metric() const override {
            return MetricType::METRIC_SPARSE_JACCARD;
        }
    };

    template<typename OBJECT_TYPE>
    class ComparatorAngleDistance : public DistanceComparator {
    public:
        explicit ComparatorAngleDistance(size_t d) : DistanceComparator(d) {}

        float operator()(const ArrayView &a, const ArrayView &b) override {
            return polaris::primitive::compare_angle_distance((OBJECT_TYPE *) a.data(),
                                                              (OBJECT_TYPE *)b.data(), dimension);
        }
        [[nodiscard]] MetricType get_metric() const override {
            return MetricType::METRIC_ANGLE;
        }
    };

    template<typename OBJECT_TYPE>
    class ComparatorNormalizedAngleDistance : public DistanceComparator {
    public:
        explicit ComparatorNormalizedAngleDistance(size_t d) : DistanceComparator(d) {}

        float operator()(const ArrayView &a, const ArrayView &b) override {
            return polaris::primitive::compare_normalized_angle_distance((OBJECT_TYPE *) a.data(),
                                                                         (OBJECT_TYPE *) b.data(), dimension);
        }
        [[nodiscard]] MetricType get_metric() const override {
            return MetricType::METRIC_NORMALIZED_ANGLE;
        }
    };

    template<typename OBJECT_TYPE>
    class ComparatorCosineSimilarity : public DistanceComparator {
    public:
        explicit ComparatorCosineSimilarity(size_t d) : DistanceComparator(d) {}

        float operator()(const ArrayView &a, const ArrayView &b) override {
            return polaris::primitive::compare_cosine_similarity((const OBJECT_TYPE *) a.data(),
                                                                 (const OBJECT_TYPE *) b.data(), dimension);
        }
        [[nodiscard]] MetricType get_metric() const override {
            return MetricType::METRIC_COSINE;
        }
    };

    template<typename OBJECT_TYPE>
    class ComparatorNormalizedCosineSimilarity : public DistanceComparator {
    public:
        explicit ComparatorNormalizedCosineSimilarity(size_t d) : DistanceComparator(d) {}

        float operator()(const ArrayView &a, const ArrayView &b) override{
            return polaris::primitive::compare_normalized_cosine_similarity((OBJECT_TYPE *) a.data(),
                                                                            (OBJECT_TYPE *) b.data(), dimension);
        }
        [[nodiscard]] MetricType get_metric() const override {
            return MetricType::METRIC_NORMALIZED_COSINE;
        }
    };

    template<typename OBJECT_TYPE>
    class ComparatorPoincareDistance : public DistanceComparator {  // added by Nyapicom
    public:
        explicit ComparatorPoincareDistance(size_t d) : DistanceComparator(d) {}

        float operator()(const ArrayView &a, const ArrayView &b) override {
            return polaris::primitive::compare_poincare_distance((OBJECT_TYPE *)a.data(),
                                                                 (OBJECT_TYPE *) b.data(), dimension);
        }
        [[nodiscard]] MetricType get_metric() const override {
            return MetricType::METRIC_POINCARE;
        }
    };

    template<typename OBJECT_TYPE>
    class ComparatorLorentzDistance : public DistanceComparator {  // added by Nyapicom
    public:
        explicit ComparatorLorentzDistance(size_t d) : DistanceComparator(d) {}

        float operator()(const ArrayView &a, const ArrayView &b) override {
            return polaris::primitive::compare_lorentz_distance((OBJECT_TYPE *) a.data(),
                                                                (OBJECT_TYPE *) b.data(), dimension);
        }
        [[nodiscard]] MetricType get_metric() const override {
            return MetricType::METRIC_LORENTZ;
        }
    };

    template<typename OBJECT_TYPE>
    class ComparatorInnerProduct : public DistanceComparator {
    public:
        explicit ComparatorInnerProduct(size_t d) : DistanceComparator(d) {}

        float operator()(const ArrayView &a, const ArrayView &b)  override {
            return polaris::primitive::compare_dot_product((OBJECT_TYPE *) a.data(), (OBJECT_TYPE *) b.data(),
                                                           dimension);
        }
        [[nodiscard]] MetricType get_metric() const override {
            return MetricType::METRIC_INNER_PRODUCT;
        }
    };

    template<typename T>
    inline DistanceComparator *get_distance_comparator(MetricType m, size_t dimension) {
        switch (m) {
            case MetricType::METRIC_L1:
                return new ComparatorL1<T>(dimension);
            case MetricType::METRIC_L2:
                return new ComparatorL2<T>(dimension);
            case MetricType::METRIC_NORMALIZED_L2:
                return new ComparatorNormalizedL2<T>(dimension);
            case MetricType::METRIC_HAMMING:
                return new ComparatorHammingDistance<T>(dimension);
            case MetricType::METRIC_JACCARD:
                return new ComparatorJaccardDistance<T>(dimension);
            case MetricType::METRIC_SPARSE_JACCARD:
                return new ComparatorSparseJaccardDistance<T>(dimension);
            case MetricType::METRIC_ANGLE:
                return new ComparatorAngleDistance<T>(dimension);
            case MetricType::METRIC_NORMALIZED_ANGLE:
                return new ComparatorNormalizedAngleDistance<T>(dimension);
            case MetricType::METRIC_COSINE:
                return new ComparatorCosineSimilarity<T>(dimension);
            case MetricType::METRIC_NORMALIZED_COSINE:
                return new ComparatorNormalizedCosineSimilarity<T>(dimension);
            case MetricType::METRIC_POINCARE:
                return new ComparatorPoincareDistance<T>(dimension);
            case MetricType::METRIC_LORENTZ:
                return new ComparatorLorentzDistance<T>(dimension);
            case MetricType::METRIC_INNER_PRODUCT:
                return new ComparatorInnerProduct<T>(dimension);
            default:
                return nullptr;
        }
    }

    inline DistanceComparator *get_distance_comparator(MetricType m, ObjectType type, size_t dimension) {
        switch (type) {
            case ObjectType::UINT8:
                return get_distance_comparator<uint8_t>(m, dimension);
            case ObjectType::FLOAT:
                return get_distance_comparator<float>(m, dimension);
            case ObjectType::FLOAT16:
                return get_distance_comparator<float16>(m, dimension);
            case ObjectType::DOUBLE:
            case ObjectType::UINT16:
            case ObjectType::INT16:
            case ObjectType::UINT32:
            case ObjectType::INT32:
            case ObjectType::UINT64:
            case ObjectType::INT64:
            case ObjectType::BFLOAT16:
            case ObjectType::INT8:
            default:
                return nullptr;
        }
    }

} // namespace polaris
