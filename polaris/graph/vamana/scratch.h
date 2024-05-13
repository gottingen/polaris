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

#include <vector>

#include <turbo/container/flat_hash_set.h>
#include <turbo/container/flat_hash_map.h>
#include <turbo/status/status.h>
#include <polaris/core/vamana_parameters.h>
#include <polaris/io/aligned_file_reader.h>
#include <polaris/graph/vamana/abstract_scratch.h>
#include <polaris/graph/vamana/neighbor.h>
#include <polaris/utility/concurrent_queue.h>
#include <collie/container/dynamic_bitset.h>

namespace polaris {
    template<typename T>
    class PQScratch;

    //
    // AbstractScratch space for in-memory index based search
    //
    template<typename T>
    class InMemQueryScratch : public AbstractScratch<T> {
    public:
        InMemQueryScratch() = default;
        ~InMemQueryScratch() override;

        turbo::Status initialize(uint32_t search_l, uint32_t indexing_l, uint32_t r, uint32_t maxc, size_t dim,
                                 size_t aligned_dim,
                                 size_t alignment_factor, bool init_pq_scratch = false);

        void resize_for_new_L(uint32_t new_search_l);

        void clear();

        inline uint32_t get_L() {
            return _L;
        }

        inline uint32_t get_R() {
            return _R;
        }

        inline uint32_t get_maxc() {
            return _maxc;
        }

        void set_query(const T *query);

        inline T *aligned_query() {
            return this->_aligned_query_T;
        }

        inline PQScratch<T> *pq_scratch() {
            return this->_pq_scratch;
        }

        inline std::vector<Neighbor> &pool() {
            return _pool;
        }

        inline NeighborPriorityQueue &best_l_nodes() {
            return _best_l_nodes;
        }

        inline std::vector<float> &occlude_factor() {
            return _occlude_factor;
        }

        inline turbo::flat_hash_set<uint32_t> &inserted_into_pool_rs() {
            return _inserted_into_pool_rs;
        }

        inline collie::dynamic_bitset<> &inserted_into_pool_bs() {
            return *_inserted_into_pool_bs;
        }

        inline std::vector<uint32_t> &id_scratch() {
            return _id_scratch;
        }

        inline std::vector<float> &dist_scratch() {
            return _dist_scratch;
        }

        inline turbo::flat_hash_set<uint32_t> &expanded_nodes_set() {
            return _expanded_nodes_set;
        }

        inline std::vector<Neighbor> &expanded_nodes_vec() {
            return _expanded_nghrs_vec;
        }

        inline std::vector<uint32_t> &occlude_list_output() {
            return _occlude_list_output;
        }

        using AbstractScratch<T>::query_view;

    private:
        uint32_t _L{0};
        uint32_t _R{0};
        uint32_t _maxc{0};
        uint32_t _dim{0};
        uint32_t _aligned_dim{0};

        // _pool stores all neighbors explored from best_L_nodes.
        // Usually around L+R, but could be higher.
        // Initialized to 3L+R for some slack, expands as needed.
        std::vector<Neighbor> _pool;

        // _best_l_nodes is reserved for storing best L entries
        // Underlying storage is L+1 to support inserts
        NeighborPriorityQueue _best_l_nodes;

        // _occlude_factor.size() >= pool.size() in occlude_list function
        // _pool is clipped to maxc in occlude_list before affecting _occlude_factor
        // _occlude_factor is initialized to maxc size
        std::vector<float> _occlude_factor;

        // Capacity initialized to 20L
        turbo::flat_hash_set<uint32_t> _inserted_into_pool_rs;

        // Use a pointer here to allow for forward declaration of dynamic_bitset
        // in public headers to avoid making boost a dependency for clients
        // of DiskANN.
        collie::dynamic_bitset<> *_inserted_into_pool_bs;

        // _id_scratch.size() must be > R*GRAPH_SLACK_FACTOR for iterate_to_fp
        std::vector<uint32_t> _id_scratch;

        // _dist_scratch must be > R*GRAPH_SLACK_FACTOR for iterate_to_fp
        // _dist_scratch should be at least the size of id_scratch
        std::vector<float> _dist_scratch;

        //  Buffers used in process delete, capacity increases as needed
        turbo::flat_hash_set<uint32_t> _expanded_nodes_set;
        std::vector<Neighbor> _expanded_nghrs_vec;
        std::vector<uint32_t> _occlude_list_output;
    };

    //
    // AbstractScratch space for SSD index based search
    //

    template<typename T>
    class SSDQueryScratch : public AbstractScratch<T> {
    public:
        T *coord_scratch = nullptr; // MUST BE AT LEAST [sizeof(T) * data_dim]

        char *sector_scratch = nullptr; // MUST BE AT LEAST [MAX_N_SECTOR_READS * SECTOR_LEN]
        size_t sector_idx = 0;          // index of next [SECTOR_LEN] scratch to use

        turbo::flat_hash_set<size_t> visited;
        NeighborPriorityQueue retset;
        std::vector<Neighbor> full_retset;

        SSDQueryScratch(size_t aligned_dim, size_t visited_reserve);

        ~SSDQueryScratch();

        void reset();
    };

    template<typename T>
    class SSDThreadData {
    public:
        SSDQueryScratch<T> scratch;
        IOContext ctx;

        SSDThreadData(size_t aligned_dim, size_t visited_reserve);

        void clear();
    };

    //
    // Class to avoid the hassle of pushing and popping the query scratch.
    //
    template<typename T>
    class ScratchStoreManager {
    public:
        ScratchStoreManager(ConcurrentQueue<T *> &query_scratch) : _scratch_pool(query_scratch) {
            _scratch = query_scratch.pop();
            while (_scratch == nullptr) {
                query_scratch.wait_for_push_notify();
                _scratch = query_scratch.pop();
            }
        }

        T *scratch_space() {
            return _scratch;
        }

        ~ScratchStoreManager() {
            _scratch->clear();
            _scratch_pool.push(_scratch);
            _scratch_pool.push_notify_all();
        }

        void destroy() {
            while (!_scratch_pool.empty()) {
                auto scratch = _scratch_pool.pop();
                while (scratch == nullptr) {
                    _scratch_pool.wait_for_push_notify();
                    scratch = _scratch_pool.pop();
                }
                delete scratch;
            }
        }

    private:
        T *_scratch;
        ConcurrentQueue<T *> &_scratch_pool;

        ScratchStoreManager(const ScratchStoreManager<T> &);

        ScratchStoreManager &operator=(const ScratchStoreManager<T> &);
    };
} // namespace polaris
