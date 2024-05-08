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

#include <cassert>
#include <sstream>
#include <stack>
#include <string>
#include <unordered_map>

#include <polaris/graph/vamana/neighbor.h>
#include <polaris/core/parameters.h>
#include <turbo/container/flat_hash_set.h>
#include <polaris/graph/vamana/utils.h>

#include <polaris/utility/platform_macros.h>

namespace polaris {
    template<typename T>
    void gen_random_slice(const std::string &base_file, const std::string &output_prefix, double sampling_rate);

    template<typename T>
    void gen_random_slice(const std::string &data_file, double p_val, float *&sampled_data, size_t &slice_size,
                          size_t &ndims);

    template<typename T>
    void gen_random_slice(const T *inputdata, size_t npts, size_t ndims, double p_val, float *&sampled_data,
                          size_t &slice_size);

    int estimate_cluster_sizes(float *test_data_float, size_t num_test, float *pivots, const size_t num_centers,
                               const size_t dim, const size_t k_base, std::vector<size_t> &cluster_sizes);

    template<typename T>
    int shard_data_into_clusters(const std::string data_file, float *pivots, const size_t num_centers, const size_t dim,
                                 const size_t k_base, std::string prefix_path);

    template<typename T>
    int shard_data_into_clusters_only_ids(const std::string data_file, float *pivots, const size_t num_centers,
                                          const size_t dim, const size_t k_base, std::string prefix_path);

    template<typename T>
    int
    retrieve_shard_data_from_ids(const std::string data_file, std::string idmap_filename, std::string data_filename);

    template<typename T>
    int partition(const std::string data_file, const float sampling_rate, size_t num_centers, size_t max_k_means_reps,
                  const std::string prefix_path, size_t k_base);

    template<typename T>
    int partition_with_ram_budget(const std::string data_file, const double sampling_rate, double ram_budget,
                                  size_t graph_degree, const std::string prefix_path, size_t k_base);

}  // namespace polaris