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

#include <algorithm>
#include <fcntl.h>
#include <cassert>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>
#include <random>
#include <set>

#ifdef __APPLE__
#else

#include <malloc.h>

#endif


#include <unistd.h>

typedef int FileHandle;

#include <polaris/io/cached_io.h>
#include <polaris/utility/common_includes.h>

#include <polaris/graph/vamana/utils.h>
#include <polaris/utility/platform_macros.h>
#include <polaris/core/index_config.h>

namespace polaris {
    const size_t MAX_SAMPLE_POINTS_FOR_WARMUP = 100000;
    const double PQ_TRAINING_SET_FRACTION = 0.1;
    const double SPACE_FOR_CACHED_NODES_IN_GB = 0.25;
    const double THRESHOLD_FOR_CACHING_IN_GB = 1.0;
    const uint32_t NUM_NODES_TO_CACHE = 250000;
    const uint32_t WARMUP_L = 20;
    const uint32_t NUM_KMEANS_REPS = 12;

    template<typename T>
    class PQFlashIndex;

    POLARIS_API double get_memory_budget(const std::string &mem_budget_str);

    POLARIS_API double get_memory_budget(double search_ram_budget_in_gb);

    POLARIS_API collie::Status add_new_file_to_single_index(std::string index_file, std::string new_file);

    POLARIS_API size_t calculate_num_pq_chunks(double final_index_ram_limit, size_t points_num, uint32_t dim);

    POLARIS_API void read_idmap(const std::string &fname, std::vector<uint32_t> &ivecs);

    template<typename T>
    POLARIS_API collie::Result<T*> load_warmup(const std::string &cache_warmup_file, uint64_t &warmup_num, uint64_t warmup_dim,
                               uint64_t warmup_aligned_dim);

    POLARIS_API int merge_shards(const std::string &vamana_prefix, const std::string &vamana_suffix,
                                 const std::string &idmaps_prefix, const std::string &idmaps_suffix,
                                 const uint64_t nshards, uint32_t max_degree, const std::string &output_vamana,
                                 const std::string &medoids_file);

    template<typename T>
    POLARIS_API std::string preprocess_base_file(const std::string &infile, const std::string &indexPrefix,
                                                 polaris::MetricType &distMetric);

    template<typename T>
    POLARIS_API collie::Status build_merged_vamana_index(std::string base_file, polaris::MetricType _compareMetric, uint32_t L,
                                              uint32_t R, double sampling_rate, double ram_budget,
                                              std::string mem_index_path, std::string medoids_file,
                                              std::string centroids_file, size_t build_pq_bytes, bool use_opq,
                                              uint32_t num_threads);
    template<typename T>
    POLARIS_API collie::Status create_disk_layout(const std::string base_file, const std::string mem_index_file,
                                        const std::string output_file,
                                        const std::string reorder_data_file = std::string(""));

} // namespace polaris
