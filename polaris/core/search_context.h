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
#pragma once
#include <polaris/core/defines.h>
#include <polaris/core/common.h>
#include <polaris/core/array_view.h>
#include <polaris/core/search_condition.h>
#include <polaris/distance/object_distance.h>
#include <turbo/times/time.h>
#include <cstdint>

namespace polaris {

    struct SearchContext {

        SearchContext &set_query(const void *data, size_t size) {
            query.assign(reinterpret_cast<const uint8_t *>(data), reinterpret_cast<const uint8_t *>(data) + size);
            query_view.set_data(query.data());
            return *this;
        }

        SearchContext &set_top_k(uint32_t k) {
            top_k = k;
            return *this;
        }

        SearchContext &set_search_list(uint32_t k) {
            search_list = k;
            return *this;
        }

        SearchContext &set_search_condition(BaseSearchCondition *condition) {
            search_condition = condition;
            return *this;
        }

        void start() {
            start_time = turbo::Time::time_now();
        }

        void done() {
            end_time = turbo::Time::time_now();
        }


        turbo::Time start_time;
        turbo::Time end_time;
        /// member variables
        aligned_vector<uint8_t> query;
        ArrayView query_view;
        uint32_t top_k{10};
        uint32_t search_list{100};

        bool with_local_ids{false};
        std::vector<localid_t> local_ids;

        bool with_raw_vectors{false};
        std::vector<std::vector<uint8_t>> raw_vectors;

        /// vid_t:distance pairs
        std::vector<ObjectDistance> top_k_queue;
        std::vector<ObjectDistance> extra_results;
        BaseSearchCondition *search_condition{NullSearchCondition::instance()};
        void * user_data{nullptr};
        uint32_t hops{0};
        uint32_t cmps{0};
        // vamana optimized_layout
        bool vamana_optimized_layout{false};

    };
}  // namespace polaris