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

#include <polaris/utility/common_includes.h>
#include <polaris/core/common.h>
#include <polaris/core/vamana_parameters.h>
#include <turbo/status/result_status.h>

namespace polaris {

    struct IndexBasicConfig {
        IndexBasicConfig() = default;

        IndexBasicConfig(MetricType metric, ObjectType object_type, size_t dimension, size_t max_points)
                : metric(metric), object_type(object_type), dimension(dimension), max_points(max_points) {
        }

        IndexBasicConfig(const IndexBasicConfig &) = default;

        IndexBasicConfig &operator=(const IndexBasicConfig &) = default;

        MetricType metric{MetricType::METRIC_NONE};
        ObjectType object_type{ObjectType::ObjectTypeNone};
        size_t dimension{0};
        size_t max_points{0};
    };

    struct IndexConfig {
        IndexBasicConfig basic_config;
        VamanaIndexConfig vamana_config;
    private:
        IndexConfig(const IndexBasicConfig &basic_config, const VamanaIndexConfig &vamana_config)
                : basic_config(basic_config), vamana_config(vamana_config) {
        }

        friend class IndexConfigBuilder;
    };

    class IndexConfigBuilder {
    public:
        IndexConfigBuilder() = default;

        IndexConfigBuilder &with_metric(MetricType m) {
            this->_basic_config.metric = m;
            return *this;
        }

        IndexConfigBuilder &vamana_with_graph_load_store_strategy(GraphStoreStrategy graph_strategy) {
            this->_vamana_config.graph_strategy = graph_strategy;
            return *this;
        }

        IndexConfigBuilder &vamana_with_data_load_store_strategy(DataStoreStrategy data_strategy) {
            this->_vamana_config.data_strategy = data_strategy;
            return *this;
        }

        IndexConfigBuilder &with_dimension(size_t dimension) {
            this->_basic_config.dimension = dimension;
            return *this;
        }

        IndexConfigBuilder &with_max_points(size_t max_points) {
            this->_basic_config.max_points = max_points;
            return *this;
        }

        IndexConfigBuilder &vamana_is_dynamic_index(bool dynamic_index) {
            this->_vamana_config.dynamic_index = dynamic_index;
            return *this;
        }

        IndexConfigBuilder &vamana_is_pq_dist_build(bool pq_dist_build) {
            this->_vamana_config.pq_dist_build = pq_dist_build;
            return *this;
        }

        IndexConfigBuilder &vamana_is_concurrent_consolidate(bool concurrent_consolidate) {
            this->_vamana_config.concurrent_consolidate = concurrent_consolidate;
            return *this;
        }

        IndexConfigBuilder &vamana_is_use_opq(bool use_opq) {
            this->_vamana_config.use_opq = use_opq;
            return *this;
        }

        IndexConfigBuilder &vamana_with_num_pq_chunks(size_t num_pq_chunks) {
            this->_vamana_config.num_pq_chunks = num_pq_chunks;
            return *this;
        }

        IndexConfigBuilder &vamana_with_num_frozen_pts(size_t num_frozen_pts) {
            this->_vamana_config.num_frozen_pts = num_frozen_pts;
            return *this;
        }

        IndexConfigBuilder &with_data_type(ObjectType data_type) {
            this->_basic_config.object_type = data_type;
            return *this;
        }

        IndexConfigBuilder &vamana_with_index_write_params(IndexWriteParameters &index_write_params) {
            this->_vamana_config.index_write_params = std::make_shared<IndexWriteParameters>(index_write_params);
            return *this;
        }

        IndexConfigBuilder &vamana_with_index_write_params(std::shared_ptr<IndexWriteParameters> index_write_params_ptr) {
            if (index_write_params_ptr == nullptr) {
                std::cout << "Passed, empty build_params while creating index config" << std::endl;
                return *this;
            }
            this->_vamana_config.index_write_params = index_write_params_ptr;
            return *this;
        }

        IndexConfigBuilder &vamana_with_index_search_params(IndexSearchParams &search_params) {
            this->_vamana_config.index_search_params = std::make_shared<IndexSearchParams>(search_params);
            return *this;
        }

        IndexConfigBuilder &vamana_with_index_search_params(std::shared_ptr<IndexSearchParams> search_params_ptr) {
            if (search_params_ptr == nullptr) {
                std::cout << "Passed, empty search_params while creating index config" << std::endl;
                return *this;
            }
            this->_vamana_config.index_search_params = search_params_ptr;
            return *this;
        }

        IndexConfig build_vamana() {
            if (_basic_config.object_type == ObjectType::ObjectTypeNone)
                throw PolarisException("Error: data_type can not be empty", -1);

            if (_basic_config.metric == MetricType::METRIC_NONE)
                throw PolarisException("Error: metric can not be empty", -1);

            if (_basic_config.dimension == 0)
                throw PolarisException("Error: dimension can not be empty", -1);

            /*
            if (_basic_config.max_points == 0)
                throw PolarisException("Error: max_points can not be empty", -1);
                */

            if (_vamana_config.dynamic_index && _vamana_config.num_frozen_pts == 0) {
                _vamana_config.num_frozen_pts = 1;
            }

            if (_vamana_config.dynamic_index) {
                if (_vamana_config.index_search_params != nullptr && _vamana_config.index_search_params->initial_search_list_size == 0)
                    throw PolarisException("Error: please pass initial_search_list_size for building dynamic index.",
                                           -1);
            }

            // sanity check
            if (_vamana_config.dynamic_index && _vamana_config.num_frozen_pts == 0) {
                std::cout << "_num_frozen_pts passed as 0 for dynamic_index. Setting it to 1 for safety."
                          << std::endl;
                _vamana_config.num_frozen_pts = 1;
            }

            return IndexConfig{_basic_config, _vamana_config};
        }

        IndexConfigBuilder(const IndexConfigBuilder &) = delete;

        IndexConfigBuilder &operator=(const IndexConfigBuilder &) = delete;

    private:
        IndexBasicConfig _basic_config;
        VamanaIndexConfig _vamana_config;
    };
} // namespace polaris
