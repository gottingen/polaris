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

#include <polaris/unified/hnsw.h>
#include <polaris/core/log.h>

namespace polaris {

    turbo::Status Hnsw::initialize(const IndexConfig &config) {
        config_ = config;
        if (config_.basic_config.metric == MetricType::METRIC_L2) {
            l2space_ = std::make_unique<hnswlib::L2Space>(config_.basic_config.dimension);
        } else if (config_.basic_config.metric == MetricType::METRIC_INNER_PRODUCT) {
            l2space_ = std::make_unique<hnswlib::InnerProductSpace>(config_.basic_config.dimension);
        } else if (config_.basic_config.metric == MetricType::METRIC_COSINE) {
            l2space_ = std::make_unique<hnswlib::InnerProductSpace>(config_.basic_config.dimension);
        }
        if (nullptr == l2space_) {
            return turbo::make_status(turbo::kInvalidArgument, "Invalid metric type");
        }

        try {
            appr_alg_ = std::make_unique<hnswlib::HierarchicalNSW<float>>(l2space_.get(), config_.basic_config.max_points,
                                                                config_.hnsw_config.m,
                                                                config_.hnsw_config.ef_construction,
                                                                config_.hnsw_config.random_seed);
        } catch (std::exception &e) {
            POLARIS_LOG(ERROR) << "HNSWLIB exception: " << e.what();
            return turbo::make_status(turbo::kResourceExhausted, "HNSWLIB exception: {}", e.what());
        } catch (...) {
            POLARIS_LOG(ERROR) << "Unkown exception";
            return turbo::make_status(turbo::kResourceExhausted, "Unkown exception");
        }

        appr_alg_->ef_ = config_.hnsw_config.ef;
        return turbo::ok_status();
    }

    turbo::Status Hnsw::load(const std::string &index_path) {
        if (!appr_alg_) {
            return turbo::make_status(turbo::kDataLoss, "index uninitialized");
        }
        return appr_alg_->load(index_path, l2space_.get());
    }

    turbo::Status Hnsw::build(const UnifiedBuildParameters &parameters) {
        return turbo::make_status(turbo::kUnimplemented, "Not implemented");
    }

    turbo::Status Hnsw::save(const std::string &index_path) {
        if (!appr_alg_) {
            return turbo::make_status(turbo::kDataLoss, "index uninitialized");
        }
        return appr_alg_->saveIndex(index_path);
    }

    turbo::Status Hnsw::add(vid_t vid, const void* vec) {
        return appr_alg_->addPoint(vec, vid);
    }

    turbo::Status Hnsw::get_vector(vid_t vid, void *vec) const {
        return  appr_alg_->get_vector(vid, vec);
    }

    size_t Hnsw::size() const {
        if (!appr_alg_) {
            return 0;
        }
        return appr_alg_->size();
    }

    turbo::Status Hnsw::lazy_remove(vid_t vid) {
        if (!appr_alg_) {
            return turbo::make_status(turbo::kDataLoss, "index uninitialized");
        }
        return appr_alg_->mark_delete(vid);
    }

    turbo::ResultStatus<consolidation_report> Hnsw::consolidate_deletes(const IndexWriteParameters &parameters) {
        return consolidation_report{};
    }

    turbo::Status Hnsw::search(SearchContext &context) {
        auto rs = appr_alg_->search(context);
        return rs;
    }
}  // namespace polaris
