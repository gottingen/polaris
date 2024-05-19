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
//

#include <polaris/unified_index.h>
#include <polaris/unified/vamana_disk.h>
#include <polaris/unified/vamana.h>
#include <polaris/unified/hnsw.h>
#include <polaris/unified/hnsw_flat.h>
namespace polaris {

    UnifiedIndex *UnifiedIndex::create_index(IndexType type) {
        switch (type) {
            case IndexType::INDEX_VAMANA_DISK:
                return new VamanaDisk();
            case IndexType::INDEX_VAMANA:
                return new Vamana();
            case IndexType::INDEX_HNSW:
                return new Hnsw();
            case IndexType::INDEX_HNSW_FLAT:
                return new HnswFlat();
            default:
                return nullptr;
        }
    }

    inline size_t get_graph_num_frozen_points(const std::string &graph_file) {
        size_t expected_file_size;
        uint32_t max_observed_degree, start;
        size_t file_frozen_pts;

        std::ifstream in;
        in.exceptions(std::ios::badbit | std::ios::failbit);

        in.open(graph_file, std::ios::binary);
        in.read((char *) &expected_file_size, sizeof(size_t));
        in.read((char *) &max_observed_degree, sizeof(uint32_t));
        in.read((char *) &start, sizeof(uint32_t));
        in.read((char *) &file_frozen_pts, sizeof(size_t));

        return file_frozen_pts;
    }

    collie::Result<size_t> UnifiedIndex::get_frozen_points(IndexType it, const std::string &index_path) {
        if(it == IndexType::INDEX_VAMANA) {
            return get_graph_num_frozen_points(index_path);
        }
        return 0;
    }
}  // namespace polaris
