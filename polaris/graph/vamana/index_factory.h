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

#include <polaris/graph/vamana/index.h>
#include <polaris/storage/abstract_graph_store.h>
#include <polaris/storage/in_mem_graph_store.h>
#include <polaris/graph/vamana/pq_data_store.h>

namespace polaris {
    class IndexFactory {
    public:
        POLARIS_API explicit IndexFactory(const IndexConfig &config);

        POLARIS_API collie::Result<std::unique_ptr<AbstractIndex>> create_instance();

        POLARIS_API static std::unique_ptr<AbstractGraphStore> construct_graphstore(
                const GraphStoreStrategy stratagy, const size_t size, const size_t reserve_graph_degree);

        template<typename T>
        POLARIS_API static std::shared_ptr<AbstractDataStore<T>> construct_datastore(DataStoreStrategy stratagy,
                                                                                     size_t num_points,
                                                                                     size_t dimension, MetricType m);

        // For now PQDataStore incorporates within itself all variants of quantization that we support. In the
        // future it may be necessary to introduce an AbstractPQDataStore class to spearate various quantization
        // flavours.
        template<typename T>
        POLARIS_API static std::shared_ptr<PQDataStore<T>> construct_pq_datastore(DataStoreStrategy strategy,
                                                                                  size_t num_points, size_t dimension,
                                                                                  MetricType m, size_t num_pq_chunks,
                                                                                  bool use_opq);

        template<typename T>
        static Distance<T> *construct_inmem_distance_fn(MetricType m);

    private:
        void check_config();

        template<typename data_type>
        collie::Result<std::unique_ptr<AbstractIndex>> create_instance();

        collie::Result<std::unique_ptr<AbstractIndex>> create_instance(ObjectType obj_type);

        std::unique_ptr<IndexConfig> _config;
    };

} // namespace polaris
