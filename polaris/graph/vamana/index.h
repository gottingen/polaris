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

#include <polaris/utility/common_includes.h>
#include <polaris/distance/distance.h>
#include <polaris/core/lock.h>
#include <polaris/utility/natural_number_map.h>
#include <polaris/utility/natural_number_set.h>
#include <polaris/graph/vamana/neighbor.h>
#include <polaris/core/vamana_parameters.h>
#include <polaris/graph/vamana/utils.h>
#include <polaris/utility/platform_macros.h>
#include <polaris/graph/vamana/scratch.h>
#include <polaris/storage/in_mem_data_store.h>
#include <polaris/storage/in_mem_graph_store.h>
#include <polaris/graph/vamana/abstract_index.h>

#include <polaris/graph/vamana/quantized_distance.h>
#include <polaris/graph/vamana/pq_data_store.h>

#define OVERHEAD_FACTOR 1.1
#define EXPAND_IF_FULL 0
#define DEFAULT_MAXC 750

namespace polaris {

    inline double estimate_ram_usage(size_t size, uint32_t dim, uint32_t datasize, uint32_t degree) {
        double size_of_data = ((double) size) * ROUND_UP(dim, 8) * datasize;
        double size_of_graph = ((double) size) * degree * sizeof(uint32_t) * defaults::GRAPH_SLACK_FACTOR;
        double size_of_locks = ((double) size) * sizeof(non_recursive_mutex);
        double size_of_outer_vector = ((double) size) * sizeof(ptrdiff_t);

        return OVERHEAD_FACTOR * (size_of_data + size_of_graph + size_of_locks + size_of_outer_vector);
    }

    template<typename T>
    class VamanaIndex : public AbstractIndex {
        /**************************************************************************
         *
         * Public functions acquire one or more of _update_lock, _consolidate_lock,
         * _tag_lock, _delete_lock before calling protected functions which DO NOT
         * acquire these locks. They might acquire locks on _locks[i]
         *
         **************************************************************************/

    public:
        // Constructor for Bulk operations and for creating the index object solely
        // for loading a prexisting index.
        POLARIS_API VamanaIndex(const IndexConfig &index_config, std::shared_ptr<AbstractDataStore<T>> data_store,
                          std::unique_ptr<AbstractGraphStore> graph_store,
                          std::shared_ptr<AbstractDataStore<T>> pq_data_store = nullptr);

        // Constructor for incremental index
        POLARIS_API VamanaIndex(MetricType m, const size_t dim, const size_t max_points,
                          const std::shared_ptr<IndexWriteParameters> index_parameters,
                          const std::shared_ptr<IndexSearchParams> index_search_params,
                          const size_t num_frozen_pts = 0, const bool dynamic_index = false,
                          const bool concurrent_consolidate = false,
                          const bool pq_dist_build = false, const size_t num_pq_chunks = 0,
                          const bool use_opq = false);

        POLARIS_API ~VamanaIndex() override;

        // Saves graph, data, metadata and associated tags.
        POLARIS_API turbo::Status save(const char *filename, bool compact_before_save) override;
        using AbstractIndex::save;
        // Load functions
        // Reads the number of frozen points from graph's metadata file section.
        POLARIS_API static size_t get_graph_num_frozen_points(const std::string &graph_file);

        POLARIS_API [[nodiscard]] turbo::Status load(const char *index_file, uint32_t num_threads, uint32_t search_l) override;
        // get some private variables
        POLARIS_API size_t get_num_points();

        POLARIS_API size_t get_max_points();

        // Batch build from a file. Optionally pass tags vector.
        POLARIS_API turbo::Status build(const char *filename, size_t num_points_to_load, const std::vector<vid_t> &tags);

        // Batch build from a file. Optionally pass tags file.
        POLARIS_API turbo::Status build(const char *filename, size_t num_points_to_load, const char *tag_filename);

        // Batch build from a data array, which must pad vectors to aligned_dim
        POLARIS_API turbo::Status build(const void *data, size_t num_points_to_load, const std::vector<vid_t> &tags) override;

        // Based on filter params builds a filtered or unfiltered index
        POLARIS_API turbo::Status build(const std::string &data_file, size_t num_points_to_load) override;

        // Set starting point of an index before inserting any points incrementally.
        // The data count should be equal to _num_frozen_pts * _aligned_dim.
        POLARIS_API void set_start_points(const T *data, size_t data_count);
        // Set starting points to random points on a sphere of certain radius.
        // A fixed random seed can be specified for scenarios where it's important
        // to have higher consistency between index builds.
        POLARIS_API void set_start_points_at_random(const std::any &radius, uint32_t random_seed) override;

        // For FastL2 search on a static index, we interleave the data with graph
        POLARIS_API void optimize_index_layout() override;

        POLARIS_API turbo::Status search(SearchContext &ctx) override;

        // Will fail if tag already in the index or if tag=0.
        POLARIS_API turbo::Status insert_point(const void *point, const vid_t tag) override;

        // call this before issuing deletions to sets relevant flags
        POLARIS_API int enable_delete();

        // Record deleted point now and restructure graph later. Return -1 if tag
        // not found, 0 if OK.
        POLARIS_API turbo::Status lazy_delete(const vid_t &tag) override;

        // Record deleted points now and restructure graph later. Add to failed_tags
        // if tag not found.
        POLARIS_API turbo::Status lazy_delete(const std::vector<vid_t> &tags, std::vector<vid_t> &failed_tags) override;

        // Call after a series of lazy deletions
        // Returns number of live points left after consolidation
        // If _conc_consolidates is set in the ctor, then this call can be invoked
        // alongside inserts and lazy deletes, else it acquires _update_lock
        POLARIS_API consolidation_report consolidate_deletes(const IndexWriteParameters &parameters) override;

        POLARIS_API bool is_index_saved();

        // repositions frozen points to the end of _data - if they have been moved
        // during deletion
        POLARIS_API void reposition_frozen_point_to_end();

        POLARIS_API void reposition_points(uint32_t old_location_start, uint32_t new_location_start,
                                           uint32_t num_locations);

        // POLARIS_API void save_index_as_one_file(bool flag);

        POLARIS_API void get_active_tags(turbo::flat_hash_set<vid_t> &active_tags) override;

        // memory should be allocated for vec before calling this function
        POLARIS_API turbo::Status get_vector_by_tag(vid_t &tag, void *vec) override;

        POLARIS_API void print_status();

        POLARIS_API void count_nodes_at_bfs_levels();

        // This variable MUST be updated if the number of entries in the metadata
        // change.
        POLARIS_API static const int METADATA_ROWS = 5;

        // ********************************
        //
        // Internals of the library
        //
        // ********************************
        // No copy/assign.
        VamanaIndex(const VamanaIndex<T> &) = delete;

        VamanaIndex<T> &operator=(const VamanaIndex<T> &) = delete;
    protected:
        POLARIS_API turbo::Status search_with_optimized_layout(SearchContext &ctx);
        // Use after _data and _nd have been populated
        // Acquire exclusive _update_lock before calling
        turbo::Status build_with_data_populated(const std::vector<vid_t> &tags);

        // generates 1 frozen point that will never be deleted from the graph
        // This is not visible to the user
        void generate_frozen_point();

        // determines navigating node of the graph by calculating medoid of datafopt
        uint32_t calculate_entry_point();

        // Returns the locations of start point and frozen points suitable for use
        // with iterate_to_fixed_point.
        std::vector<uint32_t> get_init_ids();

        std::pair<uint32_t, uint32_t> iterate_to_fixed_point(InMemQueryScratch<T> *scratch, uint32_t Lindex,
                                                             const std::vector<uint32_t> &init_ids,bool search_invocation);

        turbo::ResultStatus<std::pair<uint32_t, uint32_t>> iterate_to_fixed_point(InMemQueryScratch<T> *scratch, uint32_t Lindex,
                                                             const std::vector<uint32_t> &init_ids,
                                                             const BaseSearchCondition *condition,
                                                             bool search_invocation);

        void search_for_point_and_prune(int location, uint32_t Lindex, std::vector<uint32_t> &pruned_list,
                                        InMemQueryScratch<T> *scratch,uint32_t filteredLindex = 0);

        void prune_neighbors(const uint32_t location, std::vector<Neighbor> &pool, std::vector<uint32_t> &pruned_list,
                             InMemQueryScratch<T> *scratch);

        void prune_neighbors(const uint32_t location, std::vector<Neighbor> &pool, const uint32_t range,
                             const uint32_t max_candidate_size, const float alpha, std::vector<uint32_t> &pruned_list,
                             InMemQueryScratch<T> *scratch);

        // Prunes candidates in @pool to a shorter list @result
        // @pool must be sorted before calling
        void
        occlude_list(const uint32_t location, std::vector<Neighbor> &pool, const float alpha, const uint32_t degree,
                     const uint32_t maxc, std::vector<uint32_t> &result, InMemQueryScratch<T> *scratch,
                     const turbo::flat_hash_set<uint32_t> *const delete_set_ptr = nullptr);

        // add reverse links from all the visited nodes to node n.
        void inter_insert(uint32_t n, std::vector<uint32_t> &pruned_list, const uint32_t range,
                          InMemQueryScratch<T> *scratch);

        void inter_insert(uint32_t n, std::vector<uint32_t> &pruned_list, InMemQueryScratch<T> *scratch);

        // Acquire exclusive _update_lock before calling
        void link();

        // Acquire exclusive _tag_lock and _delete_lock before calling
        int reserve_location();

        // Acquire exclusive _tag_lock before calling
        size_t release_location(int location);

        size_t release_locations(const turbo::flat_hash_set<uint32_t> &locations);

        // Resize the index when no slots are left for insertion.
        // Acquire exclusive _update_lock and _tag_lock before calling.
        turbo::Status resize(size_t new_max_points);

        // Acquire unique lock on _update_lock, _consolidate_lock, _tag_lock
        // and _delete_lock before calling these functions.
        // Renumber nodes, update tag and location maps and compact the
        // graph, mode = _consolidated_order in case of lazy deletion and
        // _compacted_order in case of eager deletion
        POLARIS_API void compact_data();

        POLARIS_API void compact_frozen_point();

        // Remove deleted nodes from adjacency list of node loc
        // Replace removed neighbors with second order neighbors.
        // Also acquires _locks[i] for i = loc and out-neighbors of loc.
        void process_delete(const turbo::flat_hash_set<uint32_t> &old_delete_set, size_t loc, const uint32_t range,
                            const uint32_t maxc, const float alpha, InMemQueryScratch<T> *scratch);

        [[nodiscard]] turbo::Status initialize_query_scratch(uint32_t num_threads, uint32_t search_l, uint32_t indexing_l, uint32_t r,
                                      uint32_t maxc, size_t dim);

        // Do not call without acquiring appropriate locks
        // call public member functions save and load to invoke these.
        POLARIS_API size_t save_graph(std::string filename);

        POLARIS_API turbo::ResultStatus<size_t> save_data(std::string filename);

        POLARIS_API turbo::ResultStatus<size_t> save_tags(std::string filename);

        POLARIS_API turbo::ResultStatus<size_t> save_delete_list(const std::string &filename);

        POLARIS_API size_t load_graph(const std::string filename, size_t expected_num_points);

        POLARIS_API turbo::ResultStatus<size_t> load_data(std::string filename0);

        POLARIS_API size_t load_tags(const std::string tag_file_name);

        POLARIS_API size_t load_delete_set(const std::string &filename);

    private:

        IndexConfig _index_config;
        // Distance functions
        //MetricType _dist_metric = polaris::MetricType::METRIC_L2;

        // Data
        std::shared_ptr<AbstractDataStore<T>> _data_store;

        // Graph related data structures
        std::unique_ptr<AbstractGraphStore> _graph_store;

        char *_opt_graph = nullptr;

        // Dimensions
        //size_t _dim = 0;
        size_t _nd = 0;         // number of active points i.e. existing in the graph
        size_t _max_points = 0; // total number of points in given data set

        // _num_frozen_pts is the number of points which are used as initial
        // candidates when iterating to closest point(s). These are not visible
        // externally and won't be returned by search. At least 1 frozen point is
        // needed for a dynamic index. The frozen points have consecutive locations.
        // See also _start below.
        size_t _num_frozen_pts = 0;
        size_t _frozen_pts_used = 0;
        size_t _node_size;
        size_t _data_len;
        size_t _neighbor_len;

        //  Start point of the search. When _num_frozen_pts is greater than zero,
        //  this is the location of the first frozen point. Otherwise, this is a
        //  location of one of the points in index.
        uint32_t _start = 0;

        bool _has_built = false;
        bool _saturate_graph = false;
        bool _save_as_one_file = false; // plan to support in next version
        bool _dynamic_index = false;
        bool _normalize_vecs = false; // Using normalied L2 for cosine.
        bool _deletes_enabled = false;

        // Indexing parameters
        uint32_t _indexingQueueSize;
        uint32_t _indexingRange;
        uint32_t _indexingMaxC;
        float _indexingAlpha;
        uint32_t _indexingThreads;

        // Query scratch data structures
        ConcurrentQueue<InMemQueryScratch<T> *> _query_scratch;

        // Flags for PQ based distance calculation
        bool _pq_dist = false;
        bool _use_opq = false;
        size_t _num_pq_chunks = 0;
        // REFACTOR
        // uint8_t *_pq_data = nullptr;
        std::shared_ptr<QuantizedDistance<T>> _pq_distance_fn = nullptr;
        std::shared_ptr<AbstractDataStore<T>> _pq_data_store = nullptr;
        bool _pq_generated = false;
        FixedChunkPQTable _pq_table;

        //
        // Data structures, locks and flags for dynamic indexing and tags
        //

        // lazy_delete removes entry from _location_to_tag and _tag_to_location. If
        // _location_to_tag does not resolve a location, infer that it was deleted.
        turbo::flat_hash_map<vid_t, uint32_t> _tag_to_location;
        natural_number_map<uint32_t, vid_t> _location_to_tag;

        // _empty_slots has unallocated slots and those freed by consolidate_delete.
        // _delete_set has locations marked deleted by lazy_delete. Will not be
        // immediately available for insert. consolidate_delete will release these
        // slots to _empty_slots.
        natural_number_set<uint32_t> _empty_slots;
        std::unique_ptr<turbo::flat_hash_set<uint32_t>> _delete_set;

        bool _data_compacted = true;    // true if data has been compacted
        bool _is_saved = false;         // Checking if the index is already saved.
        bool _conc_consolidate = false; // use _lock while searching

        // Acquire locks in the order below when acquiring multiple locks
        std::shared_timed_mutex // RW mutex between save/load (exclusive lock) and
        _update_lock;       // search/inserts/deletes/consolidate (shared lock)
        std::shared_timed_mutex // Ensure only one consolidate or compact_data is
        _consolidate_lock;  // ever active
        std::shared_timed_mutex // RW lock for _tag_to_location,
        _tag_lock;          // _location_to_tag, _empty_slots, _nd, _max_points, _label_to_start_id
        std::shared_timed_mutex // RW Lock on _delete_set and _data_compacted
        _delete_lock;       // variable

        // Per node lock, cardinality=_max_points + _num_frozen_points
        std::vector<non_recursive_mutex> _locks;

        static const float INDEX_GROWTH_FACTOR;
    };
} // namespace polaris
