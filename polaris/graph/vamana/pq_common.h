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

#include <string>
#include <sstream>

#define NUM_PQ_BITS 8
#define NUM_PQ_CENTROIDS (1 << NUM_PQ_BITS)
#define MAX_OPQ_ITERS 20
#define NUM_KMEANS_REPS_PQ 12
#define MAX_PQ_TRAINING_SET_SIZE 256000
#define MAX_PQ_CHUNKS 512

namespace polaris {
    inline std::string get_quantized_vectors_filename(const std::string &prefix, bool use_opq, uint32_t num_chunks) {
        return prefix + (use_opq ? "_opq" : "pq") + std::to_string(num_chunks) + "_compressed.bin";
    }

    inline std::string get_pivot_data_filename(const std::string &prefix, bool use_opq, uint32_t num_chunks) {
        return prefix + (use_opq ? "_opq" : "pq") + std::to_string(num_chunks) + "_pivots.bin";
    }

    inline std::string get_rotation_matrix_suffix(const std::string &pivot_data_filename) {
        return pivot_data_filename + "_rotation_matrix.bin";
    }

} // namespace polaris
