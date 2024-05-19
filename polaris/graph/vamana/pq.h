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

#include <polaris/graph/vamana/utils.h>
#include <polaris/graph/vamana/pq_common.h>

namespace polaris {
    class FixedChunkPQTable {
        float *tables = nullptr; // pq_tables = float array of size [256 * ndims]
        uint64_t ndims = 0;      // ndims = true dimension of vectors
        uint64_t n_chunks = 0;
        bool use_rotation = false;
        uint32_t *chunk_offsets = nullptr;
        float *centroid = nullptr;
        float *tables_tr = nullptr; // same as pq_tables, but col-major
        float *rotmat_tr = nullptr;

    public:
        FixedChunkPQTable();

        virtual ~FixedChunkPQTable();

        void load_pq_centroid_bin(const char *pq_table_file, size_t num_chunks);

        uint32_t get_num_chunks();

        void preprocess_query(float *query_vec);

        // assumes pre-processed query
        void populate_chunk_distances(const float *query_vec, float *dist_vec);

        float l2_distance(const float *query_vec, uint8_t *base_vec);

        float inner_product(const float *query_vec, uint8_t *base_vec);

        // assumes no rotation is involved
        void inflate_vector(uint8_t *base_vec, float *out_vec);

        void populate_chunk_inner_products(const float *query_vec, float *dist_vec);
    };

    void
    aggregate_coords(const std::vector<unsigned> &ids, const uint8_t *all_coords, const uint64_t ndims, uint8_t *out);

    void pq_dist_lookup(const uint8_t *pq_ids, const size_t n_pts, const size_t pq_nchunks, const float *pq_dists,
                        std::vector<float> &dists_out);

    // Need to replace calls to these with calls to vector& based functions above
    void aggregate_coords(const unsigned *ids, const uint64_t n_ids, const uint8_t *all_coords, const uint64_t ndims,
                          uint8_t *out);

    void pq_dist_lookup(const uint8_t *pq_ids, const size_t n_pts, const size_t pq_nchunks, const float *pq_dists,
                        float *dists_out);

    POLARIS_API collie::Status generate_pq_pivots(const float *const train_data, size_t num_train, unsigned dim,
                                       unsigned num_centers, unsigned num_pq_chunks, unsigned max_k_means_reps,
                                       std::string pq_pivots_path, bool make_zero_mean = false);

    POLARIS_API collie::Status generate_opq_pivots(const float *train_data, size_t num_train, unsigned dim, unsigned num_centers,
                                        unsigned num_pq_chunks, std::string opq_pivots_path,
                                        bool make_zero_mean = false);

    POLARIS_API int generate_pq_pivots_simplified(const float *train_data, size_t num_train, size_t dim,
                                                  size_t num_pq_chunks, std::vector<float> &pivot_data_vector);

    template<typename T>
    int generate_pq_data_from_pivots(const std::string &data_file, unsigned num_centers, unsigned num_pq_chunks,
                                     const std::string &pq_pivots_path, const std::string &pq_compressed_vectors_path,
                                     bool use_opq = false);

    POLARIS_API int generate_pq_data_from_pivots_simplified(const float *data, const size_t num,
                                                            const float *pivot_data, const size_t pivots_num,
                                                            const size_t dim, const size_t num_pq_chunks,
                                                            std::vector<uint8_t> &pq);

    template<typename T>
    void generate_disk_quantized_data(const std::string &data_file_to_use, const std::string &disk_pq_pivots_path,
                                      const std::string &disk_pq_compressed_vectors_path,
                                      const polaris::MetricType compareMetric, const double p_val,
                                      size_t &disk_pq_dims);

    template<typename T>
    void generate_quantized_data(const std::string &data_file_to_use, const std::string &pq_pivots_path,
                                 const std::string &pq_compressed_vectors_path, const polaris::MetricType compareMetric,
                                 const double p_val, const uint64_t num_pq_chunks, const bool use_opq,
                                 const std::string &codebook_prefix = "");
} // namespace polaris
