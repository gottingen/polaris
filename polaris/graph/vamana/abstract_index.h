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
#include <polaris/core/common.h>
#include <any>

namespace polaris {
    struct consolidation_report {
        enum status_code {
            SUCCESS = 0,
            FAIL = 1,
            LOCK_FAIL = 2,
            INCONSISTENT_COUNT_ERROR = 3
        };
        status_code _status;
        size_t _active_points, _max_points, _empty_slots, _slots_released, _delete_set_size, _num_calls_to_process_delete;
        double _time;

        consolidation_report(status_code status, size_t active_points, size_t max_points, size_t empty_slots,
                             size_t slots_released, size_t delete_set_size, size_t num_calls_to_process_delete,
                             double time_secs)
                : _status(status), _active_points(active_points), _max_points(max_points), _empty_slots(empty_slots),
                  _slots_released(slots_released), _delete_set_size(delete_set_size),
                  _num_calls_to_process_delete(num_calls_to_process_delete), _time(time_secs) {
        }
    };

    /* A templated independent class for intercation with VamanaIndex. Uses Type Erasure to add virtual implemetation of methods
    that can take any type(using std::any) and Provides a clean API that can be inherited by different type of VamanaIndex.
    */
    class AbstractIndex {
    public:
        AbstractIndex() = default;

        virtual ~AbstractIndex() = default;

        virtual turbo::Status build(const std::string &data_file, const size_t num_points_to_load) = 0;
        virtual turbo::Status build(const void *data, const size_t num_points_to_load, const std::vector<vid_t> &tags) = 0;

        virtual turbo::Status save(const char *filename, bool compact_before_save = false) = 0;


        virtual void load(const char *index_file, uint32_t num_threads, uint32_t search_l) = 0;

        virtual turbo::Status search(SearchContext &search_context) = 0;

        // For FastL2 search on optimized layout
        virtual turbo::Status search_with_optimized_layout(const void *query, size_t K, size_t L, uint32_t *indices) = 0;

        // Initialize space for res_vectors before calling.
        virtual turbo::ResultStatus<size_t> search_with_tags(const void *query, const uint64_t K, const uint32_t L, vid_t *tags,
                                float *distances, std::vector<void *> &res_vectors) = 0;

        // Added search overload that takes L as parameter, so that we
        // can customize L on a per-query basis without tampering with "Parameters"
        // IDtype is either uint32_t or uint64_t
        virtual turbo::ResultStatus<std::pair<uint32_t, uint32_t>> search(const void *query, const size_t K, const uint32_t L, localid_t *indices,
                                             float *distances = nullptr) = 0;

        // insert point for unfiltered index build. do not use with filtered index
        virtual turbo::Status insert_point(const void *point, const vid_t tag) = 0;

        // delete point with tag, or return -1 if point can not be deleted
        virtual turbo::Status lazy_delete(const vid_t &tag) = 0;

        // batch delete tags and populates failed tags if unabke to delete given tags.
        virtual turbo::Status lazy_delete(const std::vector<vid_t> &tags, std::vector<vid_t> &failed_tags) = 0;

        virtual void get_active_tags(turbo::flat_hash_set<vid_t> &active_tags) = 0;

        //template<typename data_type>
        virtual void set_start_points_at_random(std::any radius, uint32_t random_seed = 0) = 0;

        virtual consolidation_report consolidate_deletes(const IndexWriteParameters &parameters) = 0;

        virtual void optimize_index_layout() = 0;

        // memory should be allocated for vec before calling this function
        virtual int get_vector_by_tag(vid_t &tag, void *vec) = 0;
    };
} // namespace polaris
