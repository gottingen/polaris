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

#include <polaris/unified/vamana.h>

namespace polaris {

    turbo::Status Vamana::initialize(const IndexConfig &config) {
        config_ = config;
        auto index_factory = polaris::IndexFactory(config);
        index_ = index_factory.create_instance();
        if (index_ == nullptr) {
            return turbo::make_status(turbo::kInvalidArgument, "Invalid object type");
        }
        return turbo::ok_status();
    }

    turbo::Status Vamana::load(const std::string &index_path) {
        if (index_ == nullptr) {
            return turbo::make_status(turbo::kInvalidArgument, "Index not initialized");
        }
        index_->load(index_path.c_str(), config_.basic_config.load_threads, config_.basic_config.max_points);
        if (config_.basic_config.metric == polaris::MetricType::METRIC_FAST_L2) {
            index_->optimize_index_layout();
        }
        return turbo::ok_status();
    }

    turbo::Status Vamana::save(const std::string &index_path) {
        if (index_ == nullptr) {
            return turbo::make_status(turbo::kInvalidArgument, "Index not initialized");
        }
        index_->save(index_path.c_str());
        return turbo::ok_status();
    }

    turbo::Status Vamana::add(vid_t vid, const std::vector<uint8_t> &vec) {
        if (index_ == nullptr) {
            return turbo::make_status(turbo::kInvalidArgument, "Index not initialized");
        }
        return index_->insert_point(vec.data(), vid);
    }

    turbo::Status Vamana::lazy_remove(vid_t vid) {
        if (index_ == nullptr) {
            return turbo::make_status(turbo::kInvalidArgument, "Index not initialized");
        }
        return index_->lazy_delete(vid);
    }

    turbo::ResultStatus<consolidation_report> Vamana::consolidate_deletes(const IndexWriteParameters &parameters) {
        if(index_ == nullptr) {
            return turbo::make_status(turbo::kInvalidArgument, "Index not initialized");
        }
        return index_->consolidate_deletes(parameters);
    }

    turbo::Status Vamana::search(SearchContext &context) {
        if (index_ == nullptr) {
            return turbo::make_status(turbo::kInvalidArgument, "Index not initialized");
        }
        return index_->search(context);
    }

    [[nodiscard]] bool Vamana::supports_dynamic() const {
        return config_.vamana_config.dynamic_index;
    }

}  // namespace polaris
