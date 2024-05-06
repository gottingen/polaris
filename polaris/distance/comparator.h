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
        DistanceComparator(size_t d) : dimension(d) {}

        virtual float operator()(const ArrayView &a, const ArrayView &b) = 0;

        size_t dimension;

        virtual ~DistanceComparator() {}
    };

    template<typename OBJECT_TYPE>
    class ComparatorL1 : public DistanceComparator {
    public:
        ComparatorL1(size_t d) : DistanceComparator(d) {}

        float operator()(const ArrayView &a, const ArrayView &b)  override {
            return polaris::primitive::compare_l1((OBJECT_TYPE *) a.data(), (OBJECT_TYPE *) b.data(), dimension);
        }
    };

    template<typename OBJECT_TYPE>
    class ComparatorL2 : public DistanceComparator {
    public:
        ComparatorL2(size_t d) : DistanceComparator(d) {}

        float operator()(const ArrayView &a, const ArrayView &b) override {
            return polaris::primitive::compare_l2((OBJECT_TYPE *) a.data(), (OBJECT_TYPE *) b.data(),
                                                  dimension);
        }
    };

    template<typename OBJECT_TYPE>
    class ComparatorFastL2 : public DistanceComparator {
    public:
        ComparatorFastL2(size_t d) : DistanceComparator(d) {}

        float operator()(const ArrayView &a, const ArrayView &b) override {

            auto ip= polaris::primitive::compare_dot_product((OBJECT_TYPE *) a.data(), (OBJECT_TYPE *) b.data(),
                                                           dimension);
            auto v = a.l2_norm_sq() + b.l2_norm_sq() - 2 * ip;
            return v <= 0.0 ? 0.0 : sqrt(v);
        }
    };

    template<typename OBJECT_TYPE>
    class ComparatorNormalizedL2 : public DistanceComparator {
    public:
        ComparatorNormalizedL2(size_t d) : DistanceComparator(d) {}

        float operator()(const ArrayView &a, const ArrayView &b) override {
            return polaris::primitive::compare_normalized_l2((OBJECT_TYPE *) a.data(),
                                                             (OBJECT_TYPE *) b.data(), dimension);
        }
    };

    template<typename OBJECT_TYPE>
    class ComparatorHammingDistance : public DistanceComparator {
    public:
        ComparatorHammingDistance(size_t d) : DistanceComparator(d) {}

        float operator()(const ArrayView &a, const ArrayView &b) override {
            return polaris::primitive::compare_hamming_distance((OBJECT_TYPE *) a.data(),
                                                                (OBJECT_TYPE *) b.data(), dimension);
        }
    };


    template<typename OBJECT_TYPE>
    class ComparatorJaccardDistance : public DistanceComparator {
    public:
        ComparatorJaccardDistance(size_t d) : DistanceComparator(d) {}

        float operator()(const ArrayView &a, const ArrayView &b) override {
            return polaris::primitive::compare_jaccard_distance((OBJECT_TYPE *) a.data(),
                                                                (OBJECT_TYPE *) b.data(), dimension);
        }
    };

    template<typename OBJECT_TYPE>
    class ComparatorSparseJaccardDistance : public DistanceComparator {
    public:
        ComparatorSparseJaccardDistance(size_t d) : DistanceComparator(d) {}

        float operator()(const ArrayView &a, const ArrayView &b) override {
            return polaris::primitive::compare_sparse_jaccard_distance((OBJECT_TYPE *) a.data(),
                                                                       (OBJECT_TYPE *) b.data(), dimension);
        }
    };

    template<typename OBJECT_TYPE>
    class ComparatorAngleDistance : public DistanceComparator {
    public:
        ComparatorAngleDistance(size_t d) : DistanceComparator(d) {}

        float operator()(const ArrayView &a, const ArrayView &b) override {
            return polaris::primitive::compare_angle_distance((OBJECT_TYPE *) a.data(),
                                                              (OBJECT_TYPE *)b.data(), dimension);
        }
    };

    template<typename OBJECT_TYPE>
    class ComparatorNormalizedAngleDistance : public DistanceComparator {
    public:
        ComparatorNormalizedAngleDistance(size_t d) : DistanceComparator(d) {}

        float operator()(const ArrayView &a, const ArrayView &b) override {
            return polaris::primitive::compare_normalized_angle_distance((OBJECT_TYPE *) a.data(),
                                                                         (OBJECT_TYPE *) b.data(), dimension);
        }
    };

    template<typename OBJECT_TYPE>
    class ComparatorCosineSimilarity : public DistanceComparator {
    public:
        ComparatorCosineSimilarity(size_t d) : DistanceComparator(d) {}

        float operator()(const ArrayView &a, const ArrayView &b) override {
            return polaris::primitive::compare_cosine_similarity((OBJECT_TYPE *) a.data(),
                                                                 (OBJECT_TYPE *) b.data(), dimension);
        }
    };

    template<typename OBJECT_TYPE>
    class ComparatorNormalizedCosineSimilarity : public DistanceComparator {
    public:
        ComparatorNormalizedCosineSimilarity(size_t d) : DistanceComparator(d) {}

        float operator()(const ArrayView &a, const ArrayView &b) override{
            return polaris::primitive::compare_normalized_cosine_similarity((OBJECT_TYPE *) a.data(),
                                                                            (OBJECT_TYPE *) b.data(), dimension);
        }
    };

    template<typename OBJECT_TYPE>
    class ComparatorPoincareDistance : public DistanceComparator {  // added by Nyapicom
    public:
        ComparatorPoincareDistance(size_t d) : DistanceComparator(d) {}

        float operator()(const ArrayView &a, const ArrayView &b) override {
            return polaris::primitive::compare_poincare_distance((OBJECT_TYPE *)a.data(),
                                                                 (OBJECT_TYPE *) b.data(), dimension);
        }
    };

    template<typename OBJECT_TYPE>
    class ComparatorLorentzDistance : public DistanceComparator {  // added by Nyapicom
    public:
        ComparatorLorentzDistance(size_t d) : DistanceComparator(d) {}

        float operator()(const ArrayView &a, const ArrayView &b) override {
            return polaris::primitive::compare_lorentz_distance((OBJECT_TYPE *) a.data(),
                                                                (OBJECT_TYPE *) b.data(), dimension);
        }
    };

    template<typename OBJECT_TYPE>
    class ComparatorInnerProduct : public DistanceComparator {
    public:
        ComparatorInnerProduct(size_t d) : DistanceComparator(d) {}

        float operator()(const ArrayView &a, const ArrayView &b)  override {
            return polaris::primitive::compare_dot_product((OBJECT_TYPE *) a.data(), (OBJECT_TYPE *) b.data(),
                                                           dimension);
        }
    };
} // namespace polaris
