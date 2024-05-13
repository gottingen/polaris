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
#pragma once

#include <polaris/distance/distance.h>
#include <polaris/core/vamana_parameters.h>
#include <polaris/core/search_context.h>
#include <turbo/status/status.h>
#include <polaris/graph/vamana/utils.h>
#include <polaris/utility/types.h>
#include <polaris/core/index_config.h>
#include <polaris/core/report.h>
#include <polaris/core/common.h>
#include <any>

namespace polaris {

    /* A templated independent class for intercation with VamanaIndex. Uses Type Erasure to add virtual implemetation of methods
    that can take any type(using std::any) and Provides a clean API that can be inherited by different type of VamanaIndex.
    */
    class AbstractIndex {
    public:
        AbstractIndex() = default;

        virtual ~AbstractIndex() = default;

        virtual turbo::Status build(const std::string &data_file, size_t num_points_to_load) = 0;
        virtual turbo::Status build(const void *data, size_t num_points_to_load, const std::vector<vid_t> &tags) = 0;


        virtual turbo::Status save(const char *filename, bool compact_before_save) = 0;
        virtual turbo::Status save(const std::string &filename) {
            return save(filename.c_str(), false);
        }

        [[nodiscard]] virtual turbo::Status load(const char *index_file, uint32_t num_threads, uint32_t search_l) = 0;

        virtual turbo::Status search(SearchContext &search_context) = 0;

        // insert point for unfiltered index build. do not use with filtered index
        virtual turbo::Status insert_point(const void *point, const vid_t tag) = 0;

        // delete point with tag, or return -1 if point can not be deleted
        virtual turbo::Status lazy_delete(const vid_t &tag) = 0;

        // batch delete tags and populates failed tags if unabke to delete given tags.
        virtual turbo::Status lazy_delete(const std::vector<vid_t> &tags, std::vector<vid_t> &failed_tags) = 0;

        virtual void get_active_tags(turbo::flat_hash_set<vid_t> &active_tags) = 0;

        virtual void set_start_points_at_random(const std::any &radius, uint32_t random_seed) = 0;

        void set_start_points_at_random(const std::any &radius) {
            set_start_points_at_random(radius, 0);
        }

        virtual consolidation_report consolidate_deletes(const IndexWriteParameters &parameters) = 0;

        virtual void optimize_index_layout() = 0;

        // memory should be allocated for vec before calling this function
        virtual turbo::Status get_vector_by_tag(vid_t &tag, void *vec) = 0;
    };
} // namespace polaris
