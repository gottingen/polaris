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

#include <polaris/utility/common_includes.h>
#include <polaris/core/common.h>
#include <polaris/io/aligned_file_reader.h>
#include <polaris/utility/concurrent_queue.h>
#include <polaris/graph/vamana/neighbor.h>
#include <polaris/core/vamana_parameters.h>
#include <polaris/core/search_context.h>
#include <polaris/core/percentile_stats.h>
#include <polaris/graph/vamana/pq.h>
#include <polaris/graph/vamana/utils.h>
#include <polaris/utility/platform_macros.h>
#include <polaris/graph/vamana/scratch.h>
#include <turbo/container/flat_hash_map.h>
#include <turbo/container/flat_hash_set.h>
#include <polaris/utility/natural_number_map.h>
#include <turbo/status/status.h>
#include <polaris/core/index_config.h>

#define FULL_PRECISION_REORDER_MULTIPLIER 3

namespace polaris {

    template<typename T>
    class PQFlashIndex {
    public:
        POLARIS_API PQFlashIndex(std::shared_ptr<AlignedFileReader> &fileReader,
                                 polaris::MetricType metric = polaris::MetricType::METRIC_L2);

        POLARIS_API ~PQFlashIndex();

    public:
        POLARIS_API static turbo::Status
        build(const char *dataFilePath, const char *indexFilePath, const IndexConfig &indexConfig,
              const std::vector<vid_t> &tags,
              const std::string &codebook_prefix = "");

        POLARIS_API static turbo::Status
        build(const char *dataFilePath, const char *indexFilePath, const std::string &tags_file,const IndexConfig &indexConfig,
              const std::string &codebook_prefix = "");

        POLARIS_API static turbo::Status
        build(const char *dataFilePath, const char *indexFilePath, const IndexConfig &indexConfig,
              const std::string &codebook_prefix = "");

    public:
        // load compressed data, and obtains the handle to the disk-resident index
        POLARIS_API turbo::Status load(uint32_t num_threads, const char *index_prefix);

        POLARIS_API turbo::Status load_from_separate_paths(uint32_t num_threads, const char *index_filepath,
                                                 const char *pivots_filepath, const char *compressed_filepath);

        POLARIS_API void load_cache_list(std::vector<uint32_t> &node_list);

        POLARIS_API void generate_cache_list_from_sample_queries(std::string sample_bin, uint64_t l_search,
                                                                 uint64_t beamwidth, uint64_t num_nodes_to_cache,
                                                                 uint32_t num_threads,
                                                                 std::vector<uint32_t> &node_list);

        POLARIS_API void cache_bfs_levels(uint64_t num_nodes_to_cache, std::vector<uint32_t> &node_list,
                                          const bool shuffle = false);

        POLARIS_API uint32_t range_search(const T *query1, const double range, const uint64_t min_l_search,
                                          const uint64_t max_l_search, std::vector<uint64_t> &indices,
                                          std::vector<float> &distances, const uint64_t min_beam_width,
                                          QueryStats *stats = nullptr);

        POLARIS_API turbo::Status search(SearchContext &search_context);

        POLARIS_API uint64_t get_data_dim();

        std::shared_ptr<AlignedFileReader> &reader;

        POLARIS_API polaris::MetricType get_metric();

        //
        // node_ids: input list of node_ids to be read
        // coord_buffers: pointers to pre-allocated buffers that coords need to copied to. If null, dont copy.
        // nbr_buffers: pre-allocated buffers to copy neighbors into
        //
        // returns a vector of bool one for each node_id: true if read is success, else false
        //
        POLARIS_API std::vector<bool> read_nodes(const std::vector<uint32_t> &node_ids,
                                                 std::vector<T *> &coord_buffers,
                                                 std::vector<std::pair<uint32_t, uint32_t *>> &nbr_buffers);

        POLARIS_API std::vector<std::uint8_t> get_pq_vector(std::uint64_t vid);

        POLARIS_API uint64_t get_num_points();

        POLARIS_API uint32_t optimize_beamwidth(T *tuning_sample, uint64_t tuning_sample_num,
                                                uint64_t tuning_sample_aligned_dim, uint32_t L, uint32_t nthreads,
                                                uint32_t start_bw = 2);

    protected:
        POLARIS_API void use_medoids_data_as_centroids();

        POLARIS_API void setup_thread_data(uint64_t nthreads, uint64_t visited_reserve = 4096);

        POLARIS_API void cached_beam_search(const T *query, const uint64_t k_search, const uint64_t l_search,
                                            uint64_t *res_ids, float *res_dists, const uint64_t beam_width,
                                            const bool use_reorder_data = false, QueryStats *stats = nullptr);

        POLARIS_API void cached_beam_search(const T *query, const uint64_t k_search, const uint64_t l_search,
                                            uint64_t *res_ids, float *res_dists, const uint64_t beam_width,
                                            const uint32_t io_limit, const bool use_reorder_data = false,
                                            QueryStats *stats = nullptr);

    private:

        void reset_stream_for_reading(std::basic_istream<char> &infile);

        // sector # on disk where node_id is present with in the graph part
        POLARIS_API uint64_t get_node_sector(uint64_t node_id);

        // ptr to start of the node
        POLARIS_API char *offset_to_node(char *sector_buf, uint64_t node_id);

        // returns region of `node_buf` containing [NNBRS][NBR_ID(uint32_t)]
        POLARIS_API uint32_t *offset_to_node_nhood(char *node_buf);

        // returns region of `node_buf` containing [COORD(T)]
        POLARIS_API T *offset_to_node_coords(char *node_buf);

        // index info for multi-node sectors
        // nhood of node `i` is in sector: [i / nnodes_per_sector]
        // offset in sector: [(i % nnodes_per_sector) * max_node_len]
        //
        // index info for multi-sector nodes
        // nhood of node `i` is in sector: [i * DIV_ROUND_UP(_max_node_len, SECTOR_LEN)]
        // offset in sector: [0]
        //
        // Common info
        // coords start at ofsset
        // #nbrs of node `i`: *(unsigned*) (offset + disk_bytes_per_point)
        // nbrs of node `i` : (unsigned*) (offset + disk_bytes_per_point + 1)

        uint64_t _max_node_len = 0;
        uint64_t _nnodes_per_sector = 0; // 0 for multi-sector nodes, >0 for multi-node sectors
        uint64_t _max_degree = 0;
        // tag
        turbo::flat_hash_map<vid_t, uint32_t> _tag_to_location;
        natural_number_map<uint32_t, vid_t> _location_to_tag;
        // Data used for searching with re-order vectors
        uint64_t _ndims_reorder_vecs = 0;
        uint64_t _reorder_data_start_sector = 0;
        uint64_t _nvecs_per_sector = 0;

        polaris::MetricType metric = polaris::MetricType::METRIC_L2;

        // used only for inner product search to re-scale the result value
        // (due to the pre-processing of base during index build)
        float _max_base_norm = 0.0f;

        // data info
        uint64_t _num_points = 0;
        uint64_t _num_frozen_points = 0;
        uint64_t _frozen_location = 0;
        uint64_t _data_dim = 0;
        uint64_t _aligned_dim = 0;
        uint64_t _disk_bytes_per_point = 0; // Number of bytes

        std::string _disk_index_file;
        std::vector<std::pair<uint32_t, uint32_t>> _node_visit_counter;

        // PQ data
        // _n_chunks = # of chunks ndims is split into
        // data: char * _n_chunks
        // chunk_size = chunk size of each dimension chunk
        // pq_tables = float* [[2^8 * [chunk_size]] * _n_chunks]
        uint8_t *data = nullptr;
        uint64_t _n_chunks;
        FixedChunkPQTable _pq_table;

        // distance comparator
        std::shared_ptr<Distance<T>> _dist_cmp;
        std::shared_ptr<Distance<float>> _dist_cmp_float;

        // for very large datasets: we use PQ even for the disk resident index
        bool _use_disk_index_pq = false;
        uint64_t _disk_pq_n_chunks = 0;
        FixedChunkPQTable _disk_pq_table;

        // medoid/start info

        // graph has one entry point by default,
        // we can optionally have multiple starting points
        uint32_t *_medoids = nullptr;
        // defaults to 1
        size_t _num_medoids;
        // by default, it is empty. If there are multiple
        // centroids, we pick the medoid corresponding to the
        // closest centroid as the starting point of search
        float *_centroid_data = nullptr;

        // nhood_cache; the uint32_t in nhood_Cache are offsets into nhood_cache_buf
        unsigned *_nhood_cache_buf = nullptr;
        turbo::flat_hash_map<uint32_t, std::pair<uint32_t, uint32_t *>> _nhood_cache;

        // coord_cache; The T* in coord_cache are offsets into coord_cache_buf
        T *_coord_cache_buf = nullptr;
        turbo::flat_hash_map<uint32_t, T *> _coord_cache;

        // thread-specific scratch
        ConcurrentQueue<SSDThreadData<T> *> _thread_data;
        uint64_t _max_nthreads;
        bool _load_flag = false;
        bool _count_visited_nodes = false;
        bool _reorder_data_exists = false;
        uint64_t _reoreder_data_offset = 0;

    };
} // namespace polaris
