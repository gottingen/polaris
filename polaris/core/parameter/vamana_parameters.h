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

#include <polaris/core/parameter/basic.h>

namespace polaris {

    namespace defaults {
        const float ALPHA = 1.2f;
        const uint32_t NUM_THREADS = 0;
        const uint32_t MAX_OCCLUSION_SIZE = 750;
        const uint32_t NUM_FROZEN_POINTS_STATIC = 0;
        const uint32_t NUM_FROZEN_POINTS_DYNAMIC = 1;

        // In-mem index related limits
        const float GRAPH_SLACK_FACTOR = 1.3f;

        // SSD VamanaIndex related limits
        const uint64_t MAX_GRAPH_DEGREE = 512;
        const uint64_t SECTOR_LEN = 4096;
        const uint64_t MAX_N_SECTOR_READS = 128;

        // following constants should always be specified, but are useful as a
        // sensible default at cli / python boundaries
        const uint32_t MAX_DEGREE = 64;
        const uint32_t BUILD_LIST_SIZE = 100;
        const uint32_t SATURATE_GRAPH = false;
        const uint32_t SEARCH_LIST_SIZE = 100;
    } // namespace defaults

    class IndexWriteParameters {
    public:
        uint32_t search_list_size{0}; // L
        uint32_t max_degree{0};       // R
        bool saturate_graph{false};
        uint32_t max_occlusion_size{0}; // C
        float alpha{0.0};
        uint32_t num_threads{0};

        IndexWriteParameters() = default;
        IndexWriteParameters(const uint32_t search_list_size, const uint32_t max_degree, const bool saturate_graph,
                             const uint32_t max_occlusion_size, const float alpha, const uint32_t num_threads)
                : search_list_size(search_list_size), max_degree(max_degree), saturate_graph(saturate_graph),
                  max_occlusion_size(max_occlusion_size), alpha(alpha), num_threads(num_threads) {
        }

        [[nodiscard]] turbo::Status export_property(polaris::PropertySet &p) const;

        [[nodiscard]] turbo::Status import_property(const polaris::PropertySet &p);

    };

    class IndexSearchParams {
    public:
        IndexSearchParams() = default;
        IndexSearchParams(const uint32_t initial_search_list_size, const uint32_t num_search_threads)
                : initial_search_list_size(initial_search_list_size), num_search_threads(num_search_threads) {
        }

        uint32_t initial_search_list_size{0}; // search L
        uint32_t num_search_threads{0};       // search threads

        [[nodiscard]] turbo::Status export_property(polaris::PropertySet &p) const;

        [[nodiscard]] turbo::Status import_property(const polaris::PropertySet &p);
    };

    struct VamanaIndexConfig : public BasicParameters {
        DatabaseType data_strategy;
        DatabaseType graph_strategy;
        bool dynamic_index{false};
        bool pq_dist_build{false};
        bool concurrent_consolidate{false};
        bool use_opq{false};
        size_t num_pq_chunks{0};
        size_t num_frozen_pts{defaults::NUM_FROZEN_POINTS_STATIC};

        // Params for building index
        std::shared_ptr<IndexWriteParameters> index_write_params;
        // Params for searching index
        std::shared_ptr<IndexSearchParams> index_search_params;

        [[nodiscard]] turbo::Status export_property(polaris::PropertySet &p) const;

        [[nodiscard]] turbo::Status import_property(const polaris::PropertySet &p);
    };

    struct VamanaDiskIndexConfig : public BasicParameters {
        uint32_t R{0};
        uint32_t L{0};
        float    B{0.0f};
        float    M{0.0f};
        uint32_t num_threads{0};
        uint32_t pq_dims{0};
        bool    append_reorder_data{false};
        uint32_t  build_pq_bytes{0};
        uint32_t pq_chunks{0};
        bool use_opq{false};
        uint32_t num_nodes_to_cache{0};

        [[nodiscard]] turbo::Status export_property(polaris::PropertySet &p) const;

        [[nodiscard]] turbo::Status import_property(const polaris::PropertySet &p);
    };

}  // namespace polaris

