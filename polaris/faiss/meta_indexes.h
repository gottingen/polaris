/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// -*- c++ -*-

#pragma once

#include <polaris/faiss/index.h>
#include <polaris/faiss/index_id_map.h>
#include <polaris/faiss/index_replicas.h>
#include <polaris/faiss/index_shards.h>
#include <vector>

namespace polaris {

/** splits input vectors in segments and assigns each segment to a sub-index
 * used to distribute a MultiIndexQuantizer
 */
struct IndexSplitVectors : Index {
    bool own_fields;
    bool threaded;
    std::vector<Index*> sub_indexes;
    idx_t sum_d; /// sum of dimensions seen so far

    explicit IndexSplitVectors(idx_t d, bool threaded = false);

    void add_sub_index(Index*);
    void sync_with_sub_indexes();

    void add(idx_t n, const float* x) override;

    void search(
            idx_t n,
            const float* x,
            idx_t k,
            float* distances,
            idx_t* labels,
            const SearchParameters* params = nullptr) const override;

    void train(idx_t n, const float* x) override;

    void reset() override;

    ~IndexSplitVectors() override;
};

/** index that returns random results.
 * used mainly for time benchmarks
 */
struct IndexRandom : Index {
    int64_t seed;

    explicit IndexRandom(
            idx_t d,
            idx_t ntotal = 0,
            int64_t seed = 1234,
            MetricType mt = METRIC_L2);

    void add(idx_t n, const float* x) override;

    void search(
            idx_t n,
            const float* x,
            idx_t k,
            float* distances,
            idx_t* labels,
            const SearchParameters* params = nullptr) const override;

    void reconstruct(idx_t key, float* recons) const override;

    void reset() override;

    ~IndexRandom() override;
};

} // namespace polaris

