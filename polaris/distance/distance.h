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

#include <polaris/internal/platform_macros.h>
#include <cstring>
#include <polaris/core/defines.h>
#include <polaris/core/metric_type.h>

namespace polaris {

    template<typename T>
    class Distance {
    public:
        POLARIS_API Distance(polaris::MetricType dist_metric) : _distance_metric(dist_metric) {
        }

        // distance comparison function
        POLARIS_API virtual float compare(const T *a, const T *b, uint32_t length) const = 0;

        // Needed only for COSINE-BYTE and INNER_PRODUCT-BYTE
        POLARIS_API virtual float compare(const T *a, const T *b, const float normA, const float normB,
                                          uint32_t length) const;

        // For MIPS, normalization adds an extra dimension to the vectors.
        // This function lets callers know if the normalization process
        // changes the dimension.
        POLARIS_API virtual uint32_t post_normalization_dimension(uint32_t orig_dimension) const;

        POLARIS_API virtual polaris::MetricType get_metric() const;

        // This is for efficiency. If no normalization is required, the callers
        // can simply ignore the normalize_data_for_build() function.
        POLARIS_API virtual bool preprocessing_required() const;

        // Check the preprocessing_required() function before calling this.
        // Clients can call the function like this:
        //
        //  if (metric->preprocessing_required()){
        //     T* normalized_data_batch;
        //      Split data into batches of batch_size and for each, call:
        //       metric->preprocess_base_points(data_batch, batch_size);
        //
        //  TODO: This does not take into account the case for SSD inner product
        //  where the dimensions change after normalization.
        POLARIS_API virtual void preprocess_base_points(T *original_data, const size_t orig_dim,
                                                        const size_t num_points);

        // Invokes normalization for a single vector during search. The scratch space
        // has to be created by the caller keeping track of the fact that
        // normalization might change the dimension of the query vector.
        POLARIS_API virtual void preprocess_query(const T *query_vec, const size_t query_dim, T *scratch_query);

        // If an algorithm has a requirement that some data be aligned to a certain
        // boundary it can use this function to indicate that requirement. Currently,
        // we are setting it to 8 because that works well for AVX2. If we have AVX512
        // implementations of distance algos, they might have to set this to 16
        // (depending on how they are implemented)
        POLARIS_API virtual size_t get_required_alignment() const;

        // Providing a default implementation for the virtual destructor because we
        // don't expect most metric implementations to need it.
        POLARIS_API virtual ~Distance() = default;

    protected:
        polaris::MetricType _distance_metric;
        size_t _alignment_factor = 8;
    };

    template<typename T>
    Distance<T> *get_distance_function(MetricType m);
}  // namespace polaris
