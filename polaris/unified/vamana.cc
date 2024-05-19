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

    collie::Status Vamana::initialize(const IndexConfig &config) {
        config_ = config;
        auto index_factory = polaris::IndexFactory(config);
        index_ = index_factory.create_instance();
        if (index_ == nullptr) {
            return collie::Status::invalid_argument("Invalid object type");
        }
        return collie::Status::ok_status();
    }

    collie::Status Vamana::build(const UnifiedBuildParameters &parameters) {
        if (index_ == nullptr) {
            return collie::Status::invalid_argument("Index not initialized");
        }
        if (parameters.data_file.empty() && parameters.data == nullptr) {
            return collie::Status::invalid_argument("Data file not provided");
        }
        collie::Status rs;
        if (!parameters.tags_file.empty()) {
            rs =  index_->build(parameters.data_file, parameters.num_points_to_load, parameters.tags_file);
        } else if (!parameters.tags.empty() && !parameters.data_file.empty()) {
            rs = index_->build(parameters.data_file, parameters.num_points_to_load, parameters.tags);
        } else if(parameters.data != nullptr && !parameters.tags.empty()) {
            rs = index_->build(parameters.data, parameters.num_points_to_load, parameters.tags);
        } else if (!parameters.data_file.empty()) {
            rs = index_->build(parameters.data_file, parameters.num_points_to_load);
        } else {
            return collie::Status::invalid_argument("Invalid build parameters");
        }
        if (!rs.ok()) {
            return rs;
        }
        return save(parameters.output_path);
    }

    collie::Status Vamana::load(const std::string &index_path) {
        if (index_ == nullptr) {
            return collie::Status::invalid_argument("Index not initialized");
        }
        auto rs = index_->load(index_path.c_str(), config_.basic_config.load_threads, config_.basic_config.max_points);
        return rs;
    }

    size_t Vamana::size() const {
        if (index_ == nullptr) {
            return 0;
        }
        return index_->size();

    }

    collie::Status Vamana::save(const std::string &index_path) {
        if (index_ == nullptr) {
            return collie::Status::invalid_argument("Index not initialized");
        }
        index_->save(index_path.c_str());
        return collie::Status::ok_status();
    }

    collie::Status Vamana::add(vid_t vid, const void*vec) {
        if (index_ == nullptr) {
            return collie::Status::invalid_argument("Index not initialized");
        }
        return index_->insert_point(vec, vid);
    }

    collie::Status Vamana::get_vector(vid_t vid, void *vec) const {
        if (index_ == nullptr) {
            return collie::Status::invalid_argument("Index not initialized");
        }
        return index_->get_vector(vid, vec);

    }

    collie::Status Vamana::lazy_remove(vid_t vid) {
        if (index_ == nullptr) {
            return collie::Status::invalid_argument("Index not initialized");
        }
        return index_->lazy_delete(vid);
    }

    collie::Result<consolidation_report> Vamana::consolidate_deletes(const IndexWriteParameters &parameters) {
        if (index_ == nullptr) {
            return collie::Status::invalid_argument("Index not initialized");
        }
        return index_->consolidate_deletes(parameters);
    }

    collie::Status Vamana::search(SearchContext &context) {
        if (index_ == nullptr) {
            return collie::Status::invalid_argument("Index not initialized");
        }
        return index_->search(context);
    }

    [[nodiscard]] bool Vamana::supports_dynamic() const {
        return config_.vamana_config.dynamic_index;
    }

}  // namespace polaris
