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
#include <polaris/unified/vamana_disk.h>
#include <polaris/graph/vamana/pq_flash_index.h>

namespace polaris {

    template <typename T>
    void vd_delete(void *index) {
        auto ptr = (PQFlashIndex<T>*)index;
        delete ptr;
    }
    VamanaDisk::~VamanaDisk() {
        if (index_ != nullptr) {
            switch (config_.basic_config.object_type) {
                case ObjectType::UINT8:
                    vd_delete<uint8_t>(index_);
                    break;
                case ObjectType::INT8:
                    vd_delete<int8_t>(index_);
                    break;
                case ObjectType::FLOAT:
                    vd_delete<float>(index_);
                    break;
                case ObjectType::UINT16:
                case ObjectType::INT16:
                case ObjectType::UINT32:
                case ObjectType::INT32:
                case ObjectType::UINT64:
                case ObjectType::INT64:
                case ObjectType::DOUBLE:
                case ObjectType::FLOAT16:
                case ObjectType::BFLOAT16:
                default:
                    break;
            }
            index_ = nullptr;
        }
    }

    turbo::Status VamanaDisk::initialize(const IndexConfig &config) {
        if (index_ != nullptr) {
            return turbo::make_status(turbo::kEAGAIN, "Index already initialized");
        }
        if (config.basic_config.object_type == ObjectType::ObjectTypeNone) {
            return turbo::make_status(turbo::kInvalidArgument, "Invalid object type");
        }
        if(config.basic_config.metric == MetricType::METRIC_NONE) {
            return turbo::make_status(turbo::kInvalidArgument, "Invalid metric type");
        }
        config_ = config;
        reader_ = std::make_shared<LinuxAlignedFileReader>();
        if (!reader_) {
            return turbo::make_status(turbo::kResourceExhausted, "Failed to create reader");
        }
        switch (config.basic_config.object_type) {
            case ObjectType::UINT8:
                index_ = new PQFlashIndex<uint8_t>(reader_, config.basic_config.metric);
                break;
            case ObjectType::INT8:
                index_ = new PQFlashIndex<int8_t>(reader_, config.basic_config.metric);
                break;
            case ObjectType::FLOAT:
                index_ = new PQFlashIndex<float>(reader_, config.basic_config.metric);
                break;
            case ObjectType::UINT16:
            case ObjectType::INT16:
            case ObjectType::UINT32:
            case ObjectType::INT32:
            case ObjectType::UINT64:
            case ObjectType::INT64:
            case ObjectType::DOUBLE:
            case ObjectType::FLOAT16:
            case ObjectType::BFLOAT16:
            default:
                return turbo::make_status(turbo::kInvalidArgument, "Unsupported object type");
        }
        return turbo::ok_status();
    }
    template <typename T>
    inline turbo::Status vd_load(void *index, uint32_t num_threads, uint32_t num_nodes_to_cache,  const std::string &index_path) {
        if (index == nullptr) {
            return turbo::make_status(turbo::kEINVAL, "Index not initialized");
        }
        auto ptr = (PQFlashIndex<T>*)index;
        ptr->load(num_threads, index_path.c_str());
        std::vector<uint32_t> node_list;
        ptr->cache_bfs_levels(num_nodes_to_cache, node_list);
        ptr->load_cache_list(node_list);
        node_list.clear();
        node_list.shrink_to_fit();
        return turbo::ok_status();
    }
    turbo::Status VamanaDisk::load(const std::string &index_path) {
        if (index_ == nullptr) {
            return turbo::make_status(turbo::kEINVAL, "Index not initialized");
        }

        switch (config_.basic_config.object_type) {
            case ObjectType::UINT8:
                return vd_load<uint8_t>(index_, config_.basic_config.load_threads, config_.disk_config.num_nodes_to_cache, index_path);
            case ObjectType::INT8:
                return vd_load<int8_t>(index_, config_.basic_config.load_threads, config_.disk_config.num_nodes_to_cache, index_path);
            case ObjectType::FLOAT:
                return vd_load<float>(index_, config_.basic_config.load_threads, config_.disk_config.num_nodes_to_cache, index_path);
            case ObjectType::UINT16:
            case ObjectType::INT16:
            case ObjectType::UINT32:
            case ObjectType::INT32:
            case ObjectType::UINT64:
            case ObjectType::INT64:
            case ObjectType::DOUBLE:
            case ObjectType::FLOAT16:
            case ObjectType::BFLOAT16:
            default:
                return turbo::make_status(turbo::kInvalidArgument, "Unsupported object type");
        }
    }

    turbo::Status VamanaDisk::save(const std::string &index_path) {
        return turbo::make_status(turbo::kUnimplemented, "Not implemented");
    }

    turbo::Status VamanaDisk::add(vid_t vid, const std::vector<uint8_t> &vec) {
        return turbo::make_status(turbo::kUnimplemented, "Not implemented");
    }

    turbo::Status VamanaDisk::lazy_remove(vid_t vid) {
        return turbo::make_status(turbo::kUnimplemented, "Not implemented");
    }

    template <typename T>
    inline turbo::Status vd_search(void *index,  SearchContext &context) {
        auto ptr = (PQFlashIndex<T>*)index;
        return ptr->search(context);
    }
    turbo::Status VamanaDisk::search(SearchContext &context) {
        if (index_ == nullptr) {
            return turbo::make_status(turbo::kEINVAL, "Index not initialized");
        }

        switch (config_.basic_config.object_type) {
            case ObjectType::UINT8:
                return vd_search<uint8_t>(index_, context);
            case ObjectType::INT8:
                return vd_search<int8_t>(index_, context);
            case ObjectType::FLOAT:
                return vd_search<float>(index_, context);
            case ObjectType::UINT16:
            case ObjectType::INT16:
            case ObjectType::UINT32:
            case ObjectType::INT32:
            case ObjectType::UINT64:
            case ObjectType::INT64:
            case ObjectType::DOUBLE:
            case ObjectType::FLOAT16:
            case ObjectType::BFLOAT16:
            default:
                return turbo::make_status(turbo::kInvalidArgument, "Unsupported object type");
        }
    }

    template <typename T>
    inline uint32_t vd_optimize_beam_width(void *index, void *tuning_sample, uint64_t tuning_sample_num,
                                 uint64_t tuning_sample_aligned_dim, uint32_t L, uint32_t nthreads,
                                 uint32_t start_bw) {
        auto ptr = (PQFlashIndex<T>*)index;
        return ptr->optimize_beamwidth((T*)tuning_sample, tuning_sample_num, tuning_sample_aligned_dim, L, nthreads, start_bw);
    }
    turbo::ResultStatus<uint32_t> VamanaDisk::optimize_beam_width(void *tuning_sample, uint64_t tuning_sample_num,
                                 uint64_t tuning_sample_aligned_dim, uint32_t L, uint32_t nthreads,
                                 uint32_t start_bw) {
        if (index_ == nullptr) {
            return turbo::Status(turbo::kEINVAL, "Index not initialized");
        }

        switch (config_.basic_config.object_type) {
            case ObjectType::UINT8:
                return vd_optimize_beam_width<uint8_t>(index_, tuning_sample, tuning_sample_num, tuning_sample_aligned_dim, L, nthreads, start_bw);
            case ObjectType::INT8:
                return vd_optimize_beam_width<int8_t>(index_, tuning_sample, tuning_sample_num, tuning_sample_aligned_dim, L, nthreads, start_bw);
            case ObjectType::FLOAT:
                return vd_optimize_beam_width<float>(index_, tuning_sample, tuning_sample_num, tuning_sample_aligned_dim, L, nthreads, start_bw);
            case ObjectType::UINT16:
            case ObjectType::INT16:
            case ObjectType::UINT32:
            case ObjectType::INT32:
            case ObjectType::UINT64:
            case ObjectType::INT64:
            case ObjectType::DOUBLE:
            case ObjectType::FLOAT16:
            case ObjectType::BFLOAT16:
            default:
                return turbo::make_status(turbo::kInvalidArgument, "Unsupported object type");
        }
    }
}  // namespace polaris
