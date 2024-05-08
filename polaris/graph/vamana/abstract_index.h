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

        virtual void build(const std::string &data_file, const size_t num_points_to_load,
                           IndexFilterParams &build_params) = 0;

        template<typename data_type>
        void build(const data_type *data, const size_t num_points_to_load, const std::vector<vid_t> &tags);

        virtual void save(const char *filename, bool compact_before_save = false) = 0;


        virtual void load(const char *index_file, uint32_t num_threads, uint32_t search_l) = 0;


        // For FastL2 search on optimized layout
        template<typename data_type>
        void search_with_optimized_layout(const data_type *query, size_t K, size_t L, uint32_t *indices);

        // Initialize space for res_vectors before calling.
        template<typename data_type>
        size_t search_with_tags(const data_type *query, const uint64_t K, const uint32_t L, vid_t *tags,
                                float *distances, std::vector<data_type *> &res_vectors, bool use_filters = false,
                                const std::string filter_label = "");

        // Added search overload that takes L as parameter, so that we
        // can customize L on a per-query basis without tampering with "Parameters"
        // IDtype is either uint32_t or uint64_t
        template<typename data_type, typename IDType>
        std::pair<uint32_t, uint32_t> search(const data_type *query, const size_t K, const uint32_t L, IDType *indices,
                                             float *distances = nullptr);

        // Filter support search
        // IndexType is either uint32_t or uint64_t
        template<typename IndexType>
        std::pair<uint32_t, uint32_t> search_with_filters(const DataType &query, const std::string &raw_label,
                                                          const size_t K, const uint32_t L, IndexType *indices,
                                                          float *distances);

        // insert points with labels, labels should be present for filtered index
        template<typename data_type, typename label_type>
        int insert_point(const data_type *point, const vid_t tag, const std::vector<label_type> &labels);

        // insert point for unfiltered index build. do not use with filtered index
        template<typename data_type>
        int insert_point(const data_type *point, const vid_t tag);

        // delete point with tag, or return -1 if point can not be deleted
        int lazy_delete(const vid_t &tag);

        // batch delete tags and populates failed tags if unabke to delete given tags.
        void lazy_delete(const std::vector<vid_t> &tags, std::vector<vid_t> &failed_tags);

        void get_active_tags(turbo::flat_hash_set<vid_t> &active_tags);

        template<typename data_type>
        void set_start_points_at_random(data_type radius, uint32_t random_seed = 0);

        virtual consolidation_report consolidate_deletes(const IndexWriteParameters &parameters) = 0;

        virtual void optimize_index_layout() = 0;

        // memory should be allocated for vec before calling this function
        template<typename data_type>
        int get_vector_by_tag(vid_t &tag, data_type *vec);

        template<typename label_type>
        void set_universal_label(const label_type universal_label);

    private:
        virtual void _build(const DataType &data, const size_t num_points_to_load, TagVector &tags) = 0;

        virtual std::pair<uint32_t, uint32_t> _search(const DataType &query, const size_t K, const uint32_t L,
                                                      std::any &indices, float *distances = nullptr) = 0;

        virtual std::pair<uint32_t, uint32_t>
        _search_with_filters(const DataType &query, const std::string &filter_label,
                             const size_t K, const uint32_t L, std::any &indices,
                             float *distances) = 0;

        virtual int _insert_point(const DataType &data_point, const TagType tag, Labelvector &labels) = 0;

        virtual int _insert_point(const DataType &data_point, const TagType tag) = 0;

        virtual int _lazy_delete(const TagType &tag) = 0;

        virtual void _lazy_delete(TagVector &tags, TagVector &failed_tags) = 0;

        virtual void _get_active_tags(TagRobinSet &active_tags) = 0;

        virtual void _set_start_points_at_random(DataType radius, uint32_t random_seed = 0) = 0;

        virtual int _get_vector_by_tag(TagType &tag, DataType &vec) = 0;

        virtual size_t _search_with_tags(const DataType &query, const uint64_t K, const uint32_t L, const TagType &tags,
                                         float *distances, DataVector &res_vectors, bool use_filters = false,
                                         const std::string filter_label = "") = 0;

        virtual void _search_with_optimized_layout(const DataType &query, size_t K, size_t L, uint32_t *indices) = 0;

        virtual void _set_universal_label(const LabelType universal_label) = 0;
    };
} // namespace polaris
