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


#include <polaris/unified/hnsw_flat.h>
#include <polaris/core/log.h>

namespace polaris {

    collie::Status HnswFlat::initialize(const IndexConfig &config) {
        config_ = config;
        if (config_.basic_config.metric == MetricType::METRIC_L2) {
            l2space_ = std::make_unique<hnswlib::L2Space>(config_.basic_config.dimension);
        } else if (config_.basic_config.metric == MetricType::METRIC_INNER_PRODUCT) {
            l2space_ = std::make_unique<hnswlib::InnerProductSpace>(config_.basic_config.dimension);
        } else if (config_.basic_config.metric == MetricType::METRIC_COSINE) {
            l2space_ = std::make_unique<hnswlib::InnerProductSpace>(config_.basic_config.dimension);
        }
        if (nullptr == l2space_) {
            return collie::Status::invalid_argument("Invalid metric type");
        }

        try {
            appr_alg_ = std::make_unique<hnswlib::BruteforceSearch<float>>(l2space_.get(), config_.basic_config.max_points);
        } catch (std::exception &e) {
            POLARIS_LOG(ERROR) << "HNSWLIB exception: " << e.what();
            return collie::Status::resource_exhausted("HNSWLIB exception: {}", e.what());
        } catch (...) {
            POLARIS_LOG(ERROR) << "Unkown exception";
            return collie::Status::resource_exhausted("Unkown exception");
        }

        return collie::Status::ok_status();
    }

    collie::Status HnswFlat::load(const std::string &index_path) {
        if (!appr_alg_) {
            return collie::Status::data_loss("index uninitialized");
        }
        return appr_alg_->load(index_path, l2space_.get());
    }

    collie::Status HnswFlat::build(const UnifiedBuildParameters &parameters) {
        return collie::Status::unimplemented("Not implemented");
    }

    collie::Status HnswFlat::save(const std::string &index_path) {
        if (!appr_alg_) {
            return collie::Status::data_loss("index uninitialized");
        }
        return appr_alg_->saveIndex(index_path);
    }

    collie::Status HnswFlat::add(vid_t vid, const void *vec) {
        if (!appr_alg_) {
            return collie::Status::data_loss("index uninitialized");
        }
        return appr_alg_->addPoint(vec, vid);
    }

    size_t HnswFlat::size() const {
        if (!appr_alg_) {
            return 0;
        }
        return appr_alg_->size();
    }

    collie::Status HnswFlat::get_vector(vid_t vid, void *vec) const {
        return  appr_alg_->get_vector(vid, vec);
    }

    collie::Status HnswFlat::lazy_remove(vid_t vid) {
        if (!appr_alg_) {
            return collie::Status::data_loss("index uninitialized");
        }
        return appr_alg_->mark_delete(vid);
    }

    collie::Result<consolidation_report> HnswFlat::consolidate_deletes(const IndexWriteParameters &parameters) {
        return consolidation_report{};
    }

    collie::Status HnswFlat::search(SearchContext &context) {
        auto rs = appr_alg_->search(context);
        return rs;
    }
}  // namespace polaris
