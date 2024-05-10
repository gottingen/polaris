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

#include <sstream>
#include <typeinfo>
#include <unordered_map>

#include <omp.h>
#include <polaris/utility/polaris_exception.h>

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
        const uint32_t search_list_size; // L
        const uint32_t max_degree;       // R
        const bool saturate_graph;
        const uint32_t max_occlusion_size; // C
        const float alpha;
        const uint32_t num_threads;

        IndexWriteParameters(const uint32_t search_list_size, const uint32_t max_degree, const bool saturate_graph,
                             const uint32_t max_occlusion_size, const float alpha, const uint32_t num_threads)
                : search_list_size(search_list_size), max_degree(max_degree), saturate_graph(saturate_graph),
                  max_occlusion_size(max_occlusion_size), alpha(alpha), num_threads(num_threads) {
        }

        friend class IndexWriteParametersBuilder;
    };

    class IndexSearchParams {
    public:
        IndexSearchParams(const uint32_t initial_search_list_size, const uint32_t num_search_threads)
                : initial_search_list_size(initial_search_list_size), num_search_threads(num_search_threads) {
        }

        const uint32_t initial_search_list_size; // search L
        const uint32_t num_search_threads;       // search threads
    };

    class IndexWriteParametersBuilder {
        /**
         * Fluent builder pattern to keep track of the 7 non-default properties
         * and their order. The basic ctor was getting unwieldy.
         */
    public:
        IndexWriteParametersBuilder(const uint32_t search_list_size, // L
                                    const uint32_t max_degree        // R
        )
                : _search_list_size(search_list_size), _max_degree(max_degree) {
        }

        IndexWriteParametersBuilder &with_max_occlusion_size(const uint32_t max_occlusion_size) {
            _max_occlusion_size = max_occlusion_size;
            return *this;
        }

        IndexWriteParametersBuilder &with_saturate_graph(const bool saturate_graph) {
            _saturate_graph = saturate_graph;
            return *this;
        }

        IndexWriteParametersBuilder &with_alpha(const float alpha) {
            _alpha = alpha;
            return *this;
        }

        IndexWriteParametersBuilder &with_num_threads(const uint32_t num_threads) {
            _num_threads = num_threads == 0 ? omp_get_num_procs() : num_threads;
            return *this;
        }

        IndexWriteParameters build() const {
            return IndexWriteParameters(_search_list_size, _max_degree, _saturate_graph, _max_occlusion_size, _alpha,
                                        _num_threads);
        }

        IndexWriteParametersBuilder(const IndexWriteParameters &wp)
                : _search_list_size(wp.search_list_size), _max_degree(wp.max_degree),
                  _max_occlusion_size(wp.max_occlusion_size), _saturate_graph(wp.saturate_graph), _alpha(wp.alpha) {
        }

        IndexWriteParametersBuilder(const IndexWriteParametersBuilder &) = delete;

        IndexWriteParametersBuilder &operator=(const IndexWriteParametersBuilder &) = delete;

    private:
        uint32_t _search_list_size{};
        uint32_t _max_degree{};
        uint32_t _max_occlusion_size{defaults::MAX_OCCLUSION_SIZE};
        bool _saturate_graph{defaults::SATURATE_GRAPH};
        float _alpha{defaults::ALPHA};
        uint32_t _num_threads{defaults::NUM_THREADS};
    };

    enum class DataStoreStrategy {
        MEMORY
    };

    enum class GraphStoreStrategy {
        MEMORY
    };

    struct VamanaIndexConfig {
        DataStoreStrategy data_strategy;
        GraphStoreStrategy graph_strategy;
        bool dynamic_index{false};
        bool enable_tags{false};
        bool pq_dist_build{false};
        bool concurrent_consolidate{false};
        bool use_opq{false};
        size_t num_pq_chunks{0};
        size_t num_frozen_pts{defaults::NUM_FROZEN_POINTS_STATIC};

        // Params for building index
        std::shared_ptr<IndexWriteParameters> index_write_params;
        // Params for searching index
        std::shared_ptr<IndexSearchParams> index_search_params;
    };

} // namespace polaris
