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
// Created by jeff on 24-4-29.
//
#include <polaris/datasets/random.h>
#include <polaris/utility/platform_macros.h>
#include <polaris/graph/vamana/math_utils.h>
#include <polaris/graph/vamana/pq.h>
#include <polaris/graph/vamana/partition.h>
#include <iostream>

#define KMEANS_ITERS_FOR_PQ 15

namespace polaris {

    template<typename T>
    collie::Status generate_pq(const std::string &data_path, const std::string &index_prefix_path, const size_t num_pq_centers,
                     const size_t num_pq_chunks, const float sampling_rate, const bool opq) {
        std::string pq_pivots_path = index_prefix_path + "_pq_pivots.bin";
        std::string pq_compressed_vectors_path = index_prefix_path + "_pq_compressed.bin";

        // generates random sample and sets it to train_data and updates train_size
        size_t train_size, train_dim;
        float *train_data;
        gen_random_slice<T>(data_path, sampling_rate, train_data, train_size, train_dim);
        std::cout << "For computing pivots, loaded sample data of size " << train_size << std::endl;

        if (opq) {
            COLLIE_RETURN_NOT_OK(polaris::generate_opq_pivots(train_data, train_size, (uint32_t) train_dim, (uint32_t) num_pq_centers,
                                         (uint32_t) num_pq_chunks, pq_pivots_path, true));
        } else {
            COLLIE_RETURN_NOT_OK(polaris::generate_pq_pivots(train_data, train_size, (uint32_t) train_dim, (uint32_t) num_pq_centers,
                                        (uint32_t) num_pq_chunks, KMEANS_ITERS_FOR_PQ, pq_pivots_path));
        }
        polaris::generate_pq_data_from_pivots<T>(data_path, (uint32_t) num_pq_centers, (uint32_t) num_pq_chunks,
                                                 pq_pivots_path, pq_compressed_vectors_path, true);

        delete[] train_data;

        return collie::Status::ok_status();
    }

    collie::Status
    generate_pq_float(const std::string &base_file, const std::string &output_prefix, uint32_t num_pq_chunks,
                      float sampling_rate, bool opq) {
        const size_t num_pq_centers = 256;
        return generate_pq<float>(base_file, output_prefix, num_pq_centers, num_pq_chunks, sampling_rate, opq);
    }

    collie::Status generate_pq_int8(const std::string &base_file, const std::string &output_prefix, uint32_t num_pq_chunks, float sampling_rate, bool opq) {
        const size_t num_pq_centers = 256;
        return generate_pq<int8_t>(base_file, output_prefix, num_pq_centers, num_pq_chunks, sampling_rate, opq);
    }
    collie::Status generate_pq_uint8(const std::string &base_file, const std::string &output_prefix, uint32_t num_pq_chunks, float sampling_rate, bool opq) {
        const size_t num_pq_centers = 256;
        return generate_pq<uint8_t>(base_file, output_prefix, num_pq_centers, num_pq_chunks, sampling_rate, opq);
    }
}  // namespace polaris