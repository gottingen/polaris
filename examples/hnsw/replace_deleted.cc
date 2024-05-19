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

#include <polaris/graph/hnsw/hnswlib.h>


int main() {
    int dim = 16;               // Dimension of the elements
    int max_elements = 10000;   // Maximum number of elements, should be known beforehand
    int M = 16;                 // Tightly connected with internal dimensionality of the data
                                // strongly affects the memory consumption
    int ef_construction = 200;  // Controls index search speed/build speed tradeoff

    // Initing index with allow_replace_deleted=true
    int seed = 100; 
    hnswlib::L2Space space(dim);
    hnswlib::HierarchicalNSW<float>* alg_hnsw = new hnswlib::HierarchicalNSW<float>(&space);
    auto rs = alg_hnsw->initialize(&space, max_elements, M, ef_construction, seed, true);
    if(!rs.ok()) {
        std::cout << "Error: " << rs.to_string() << std::endl;
        return 1;
    }

    // Generate random data
    std::mt19937 rng;
    rng.seed(47);
    std::uniform_real_distribution<> distrib_real;
    float* data = new float[dim * max_elements];
    for (int i = 0; i < dim * max_elements; i++) {
        data[i] = distrib_real(rng);
    }

    // Add data to index
    for (int i = 0; i < max_elements; i++) {
        auto r = alg_hnsw->addPoint(data + i * dim, i);
        if (!r.ok()) {
            std::cerr << "Failed to add element " << i << std::endl;
        }
    }

    // Mark first half of elements as deleted
    int num_deleted = max_elements / 2;
    for (int i = 0; i < num_deleted; i++) {
        auto r = alg_hnsw->mark_delete(i);
        if (!r.ok()) {
            std::cerr << "Failed to delete element " << i << std::endl;
        }
    }

    // Generate additional random data
    float* add_data = new float[dim * num_deleted];
    for (int i = 0; i < dim * num_deleted; i++) {
        add_data[i] = distrib_real(rng);
    }

    // Replace deleted data with new elements
    // Maximum number of elements is reached therefore we cannot add new items,
    // but we can replace the deleted ones by using replace_deleted=true
    for (int i = 0; i < num_deleted; i++) {
        polaris::vid_t label = max_elements + i;
        auto r = alg_hnsw->addPoint(add_data + i * dim, label, true);
        if (!r.ok()) {
            std::cerr << "Failed to replace deleted element " << i << std::endl;
        }
    }

    delete[] data;
    delete[] add_data;
    delete alg_hnsw;
    return 0;
}
