// Copyright 2024 The EA Authors.
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

#include <cstdint>
#include <polaris/graph/vamana/pq_common.h>
#include <polaris/graph/vamana/utils.h>

namespace polaris {

    template<typename T>
    class PQScratch {
    public:
        float *aligned_pqtable_dist_scratch = nullptr; // MUST BE AT LEAST [256 * NCHUNKS]
        float *aligned_dist_scratch = nullptr;         // MUST BE AT LEAST diskann MAX_DEGREE
        uint8_t *aligned_pq_coord_scratch = nullptr;   // AT LEAST  [N_CHUNKS * MAX_DEGREE]
        float *rotated_query = nullptr;
        float *aligned_query_float = nullptr;

        PQScratch(size_t graph_degree, size_t aligned_dim);

        void initialize(size_t dim, const T *query, const float norm = 1.0f);

        virtual ~PQScratch();
    };

} // namespace polaris