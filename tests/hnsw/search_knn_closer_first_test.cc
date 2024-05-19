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
// This is a test file for testing the interface
//  >>> virtual std::vector<std::pair<dist_t, labeltype>>
//  >>>    searchKnnCloserFirst(const void* query_data, size_t k) const;
// of class AlgorithmInterface

#include <polaris/graph/hnsw/hnswlib.h>
#include <polaris/unified_index.h>
#include <polaris/core/common.h>
#include <polaris/core/log.h>
#include <cassert>

#include <vector>
#include <iostream>

namespace {

using idx_t = polaris::vid_t;

void test() {
    int d = 4;
    idx_t n = 100;
    idx_t nq = 10;
    size_t k = 10;

    std::vector<float> data(n * d);
    std::vector<float> query(nq * d);

    std::mt19937 rng;
    rng.seed(47);
    std::uniform_real_distribution<> distrib;

    for (idx_t i = 0; i < n * d; ++i) {
        data[i] = distrib(rng);
    }
    for (idx_t i = 0; i < nq * d; ++i) {
        query[i] = distrib(rng);
    }
    POLARIS_LOG(INFO) << "Data generated";

    polaris::IndexConfig config = polaris::IndexConfigBuilder()
            .with_dimension(d)
            .with_max_points(n * 2)
            .with_metric(polaris::MetricType::METRIC_L2)
            .with_data_type(polaris::ObjectType::FLOAT)
            .build_hnsw();
    polaris::UnifiedIndex* hnsw_index = polaris::UnifiedIndex::create_index(polaris::IndexType::INDEX_HNSW);
    if(!hnsw_index) {
        POLARIS_LOG(ERROR) << "Failed to create hnsw index";
        return;
    }
    auto rs = hnsw_index->initialize(config);
    if(!rs.ok()) {
        POLARIS_LOG(ERROR) << "Failed to initialize hnsw index";
        return;
    }

    polaris::UnifiedIndex* flat_index = polaris::UnifiedIndex::create_index(polaris::IndexType::INDEX_HNSW_FLAT);
    rs = flat_index->initialize(config);
    if(!rs.ok()) {
        POLARIS_LOG(ERROR) << "Failed to initialize flat index";
        return;
    }

    hnswlib::L2Space space(d);
    hnswlib::AlgorithmInterface<float>* alg_brute  = new hnswlib::BruteforceSearch<float>(&space, 2 * n);
    auto * alg_hnsw = new hnswlib::HierarchicalNSW<float>(&space);
    rs = alg_hnsw->initialize(&space, 2 * n);
    if(!rs.ok()) {
        POLARIS_LOG(ERROR) << "Failed to initialize hnsw algorithm";
        return;
    }
    POLARIS_LOG(INFO) << "Index created";
    for (size_t i = 0; i < n; ++i) {
        auto rb =alg_brute->addPoint(data.data() + d * i, i);
        auto ra = alg_hnsw->addPoint(data.data() + d * i, i);
        rs = flat_index->add(i+1, data.data() + d * i);
        if(!rs.ok() || !rb.ok() || !ra.ok()){
            POLARIS_LOG(ERROR) << "Failed to add data to flat index";
            return;
        }
        rs = hnsw_index->add(i+1, data.data() + d * i);
        if(!rs.ok()) {
            POLARIS_LOG(ERROR) << "Failed to add data to hnsw index";
            return;
        }
    }
    POLARIS_LOG(INFO) << "Data added";

    // test searchKnnCloserFirst of BruteforceSearch
    for (size_t j = 0; j < nq; ++j) {
        const void* p = query.data() + j * d;
        polaris::SearchContext context;
        context.set_meta(polaris::ObjectType::FLOAT, d)
        .set_top_k(k)
        .set_query(p);

        auto gd = flat_index->search(context);
        auto res = alg_brute->searchKnnCloserFirst(p, k);
        if(context.top_k_queue.size() != res.size()) {
            std::cout << "context.top_k_queue.size() = " << context.top_k_queue.size() << std::endl;
            std::cout << "res.size() = " << res.size() << std::endl;
            exit(-1);
        }
        size_t t = context.top_k_queue.size();
        for (int i = 0; i < t; ++i) {
            assert(context.top_k_queue[i].id == res[i].second);
        }
    }
    for (size_t j = 0; j < nq; ++j) {
        const void* p = query.data() + j * d;
        polaris::SearchContext context;
        context.set_meta(polaris::ObjectType::FLOAT, d)
                .set_top_k(k)
                .set_query(p);
        auto gd = hnsw_index->search(context);
        auto res = alg_hnsw->searchKnnCloserFirst(p, k);
        if(context.top_k_queue.size() != res.size()) {
            std::cout << "context.top_k_queue.size() = " << context.top_k_queue.size() << std::endl;
            std::cout << "res.size() = " << res.size() << std::endl;
            exit(-1);
        }
        size_t t = context.top_k_queue.size();
        for (int i = 0; i < t; ++i) {
            assert(context.top_k_queue[i].id == res[i].second);
        }
    }

    delete alg_brute;
    delete alg_hnsw;
    delete hnsw_index;
    delete flat_index;
}

}  // namespace

int main() {
    std::cout << "Testing ..." << std::endl;
    test();
    std::cout << "Test ok" << std::endl;

    return 0;
}
