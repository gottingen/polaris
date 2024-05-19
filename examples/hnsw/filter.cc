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


// Filter that allows labels divisible by divisor
class PickDivisibleIds: public hnswlib::BaseFilterFunctor {
unsigned int divisor = 1;
 public:
    PickDivisibleIds(unsigned int divisor): divisor(divisor) {
        assert(divisor != 0);
    }
    bool operator()(polaris::vid_t label_id) {
        return label_id % divisor == 0;
    }
};


int main() {
    int dim = 16;               // Dimension of the elements
    int max_elements = 10000;   // Maximum number of elements, should be known beforehand
    int M = 16;                 // Tightly connected with internal dimensionality of the data
                                // strongly affects the memory consumption
    int ef_construction = 200;  // Controls index search speed/build speed tradeoff

    // Initing index
    hnswlib::L2Space space(dim);
    hnswlib::HierarchicalNSW<float>* alg_hnsw = new hnswlib::HierarchicalNSW<float>(&space);
    auto rs = alg_hnsw->initialize(&space, max_elements, M, ef_construction);
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
        auto rs = alg_hnsw->addPoint(data + i * dim, i);
        if(!rs.ok()) {
            std::cerr << "Failed to add element " << i << std::endl;
        }
    }

    // Create filter that allows only even labels
    PickDivisibleIds pickIdsDivisibleByTwo(2);

    // Query the elements for themselves with filter and check returned labels
    int k = 10;
    for (int i = 0; i < max_elements; i++) {
        std::vector<std::pair<float, polaris::vid_t>> result = alg_hnsw->searchKnnCloserFirst(data + i * dim, k, &pickIdsDivisibleByTwo);
        for (auto item: result) {
            if (item.second % 2 == 1) std::cout << "Error: found odd label\n";
        }
    }

    delete[] data;
    delete alg_hnsw;
    return 0;
}
