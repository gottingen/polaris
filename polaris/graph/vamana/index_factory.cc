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

#include <polaris/graph/vamana/index_factory.h>
#include <polaris/graph/vamana/pq_l2_distance.h>
#include <polaris/distance/distance_impl.h>

namespace polaris {

    IndexFactory::IndexFactory(const IndexConfig &config) : _config(std::make_unique<IndexConfig>(config)) {
        check_config();
    }

    std::unique_ptr<AbstractIndex> IndexFactory::create_instance() {
        return create_instance(_config->data_type, _config->label_type);
    }

    void IndexFactory::check_config() {
        if (_config->dynamic_index && !_config->enable_tags) {
            throw PolarisException("ERROR: Dynamic Indexing must have tags enabled.", -1, __PRETTY_FUNCTION__, __FILE__,
                                   __LINE__);
        }

        if (_config->pq_dist_build) {
            if (_config->dynamic_index)
                throw PolarisException("ERROR: Dynamic Indexing not supported with PQ distance based "
                                       "index construction",
                                       -1, __PRETTY_FUNCTION__, __FILE__, __LINE__);
            if (_config->metric == polaris::MetricType::METRIC_INNER_PRODUCT)
                throw PolarisException("ERROR: Inner product metrics not yet supported "
                                       "with PQ distance "
                                       "base index",
                                       -1, __PRETTY_FUNCTION__, __FILE__, __LINE__);
        }

        if (_config->data_type != "float" && _config->data_type != "uint8" && _config->data_type != "int8") {
            throw PolarisException("ERROR: invalid data type : + " + _config->data_type +
                                   " is not supported. please select from [float, int8, uint8]",
                                   -1);
        }

        if (_config->tag_type != "int32" && _config->tag_type != "uint32" && _config->tag_type != "int64" &&
            _config->tag_type != "uint64") {
            throw PolarisException("ERROR: invalid data type : + " + _config->tag_type +
                                   " is not supported. please select from [int32, uint32, int64, uint64]",
                                   -1);
        }
    }

    template<typename T>
    Distance<T> *IndexFactory::construct_inmem_distance_fn(MetricType metric) {
        if (metric == polaris::MetricType::METRIC_COSINE && std::is_same<T, float>::value) {
            return (Distance<T> *) new AVXNormalizedCosineDistanceFloat();
        } else {
            return (Distance<T> *) get_distance_function<T>(metric);
        }
    }

    template<typename T>
    std::shared_ptr<AbstractDataStore<T>> IndexFactory::construct_datastore(DataStoreStrategy strategy,
                                                                            size_t total_internal_points,
                                                                            size_t dimension,
                                                                            MetricType metric) {
        std::unique_ptr<Distance<T>> distance;
        switch (strategy) {
            case DataStoreStrategy::MEMORY:
                distance.reset(construct_inmem_distance_fn<T>(metric));
                return std::make_shared<polaris::InMemDataStore<T>>((location_t) total_internal_points, dimension,
                                                                    std::move(distance));
            default:
                break;
        }
        return nullptr;
    }

    std::unique_ptr<AbstractGraphStore> IndexFactory::construct_graphstore(const GraphStoreStrategy strategy,
                                                                           const size_t size,
                                                                           const size_t reserve_graph_degree) {
        switch (strategy) {
            case GraphStoreStrategy::MEMORY:
                return std::make_unique<InMemGraphStore>(size, reserve_graph_degree);
            default:
                throw PolarisException("Error : Current GraphStoreStratagy is not supported.", -1);
        }
    }

    template<typename T>
    std::shared_ptr<PQDataStore<T>> IndexFactory::construct_pq_datastore(DataStoreStrategy strategy, size_t num_points,
                                                                         size_t dimension, MetricType m,
                                                                         size_t num_pq_chunks,
                                                                         bool use_opq) {
        std::unique_ptr<Distance<T>> distance_fn;
        std::unique_ptr<QuantizedDistance<T>> quantized_distance_fn;

        quantized_distance_fn = std::move(std::make_unique<PQL2Distance<T>>((uint32_t) num_pq_chunks, use_opq));
        switch (strategy) {
            case DataStoreStrategy::MEMORY:
                distance_fn.reset(construct_inmem_distance_fn<T>(m));
                return std::make_shared<polaris::PQDataStore<T>>(dimension, (location_t) (num_points), num_pq_chunks,
                                                                 std::move(distance_fn),
                                                                 std::move(quantized_distance_fn));
            default:
                // REFACTOR TODO: We do support diskPQ - so we may need to add a new class for SSDPQDataStore!
                break;
        }
        return nullptr;
    }

    template<typename data_type, typename label_type>
    std::unique_ptr<AbstractIndex> IndexFactory::create_instance() {
        size_t num_points = _config->max_points + _config->num_frozen_pts;
        size_t dim = _config->dimension;
        // auto graph_store = construct_graphstore(_config->graph_strategy, num_points);
        auto data_store = construct_datastore<data_type>(_config->data_strategy, num_points, dim, _config->metric);
        std::shared_ptr<AbstractDataStore<data_type>> pq_data_store = nullptr;

        if (_config->data_strategy == DataStoreStrategy::MEMORY && _config->pq_dist_build) {
            pq_data_store =
                    construct_pq_datastore<data_type>(_config->data_strategy, num_points + _config->num_frozen_pts, dim,
                                                      _config->metric, _config->num_pq_chunks, _config->use_opq);
        } else {
            pq_data_store = data_store;
        }
        size_t max_reserve_degree =
                (size_t) (defaults::GRAPH_SLACK_FACTOR * 1.05 *
                          (_config->index_write_params == nullptr ? 0 : _config->index_write_params->max_degree));
        std::unique_ptr<AbstractGraphStore> graph_store =
                construct_graphstore(_config->graph_strategy, num_points + _config->num_frozen_pts, max_reserve_degree);

        // REFACTOR TODO: Must construct in-memory PQDatastore if strategy == ONDISK and must construct
        // in-mem and on-disk PQDataStore if strategy == ONDISK and diskPQ is required.
        return std::make_unique<polaris::VamanaIndex<data_type, label_type>>(*_config, data_store,
                                                                                 std::move(graph_store), pq_data_store);
    }

    std::unique_ptr<AbstractIndex>
    IndexFactory::create_instance(const std::string &data_type, const std::string &label_type) {
        if (data_type == std::string("float")) {
            return create_instance<float>(label_type);
        } else if (data_type == std::string("uint8")) {
            return create_instance<uint8_t>(label_type);
        } else if (data_type == std::string("int8")) {
            return create_instance<int8_t>(label_type);
        } else
            throw PolarisException("Error: unsupported data_type please choose from [float/int8/uint8]", -1);
    }

    template<typename data_type>
    std::unique_ptr<AbstractIndex> IndexFactory::create_instance(const std::string &label_type) {
        if (label_type == std::string("uint16") || label_type == std::string("ushort")) {
            return create_instance<data_type, uint16_t>();
        } else if (label_type == std::string("uint32") || label_type == std::string("uint")) {
            return create_instance<data_type, uint32_t>();
        } else
            throw PolarisException("Error: unsupported label_type please choose from [uint/ushort]", -1);
    }


} // namespace polaris
