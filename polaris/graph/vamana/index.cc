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

#include <omp.h>

#include <type_traits>
#include <polaris/graph/vamana/index_factory.h>
#include <polaris/utility/memory_mapper.h>
#include <polaris/graph/vamana/timer.h>
#include <turbo/container/flat_hash_map.h>
#include <turbo/container/flat_hash_set.h>
#include <polaris/utility/platform_macros.h>
#include <polaris/utility/tag_uint128.h>
#include <polaris/distance/distance_impl.h>
#include <polaris/core/log.h>

#if defined(DISKANN_RELEASE_UNUSED_TCMALLOC_MEMORY_AT_CHECKPOINTS) && defined(DISKANN_BUILD)
#include <gperftools/malloc_extension.h>
#endif

#ifdef _WINDOWS
#include <xmmintrin.h>
#endif

#include <polaris/graph/vamana/index.h>

#define MAX_POINTS_FOR_USING_BITSET 10000000

namespace polaris {
    // Initialize an index with metric m, load the data of type T with filename
    // (bin), and initialize max_points
    template<typename T>
    VamanaIndex<T>::VamanaIndex(const IndexConfig &index_config, std::shared_ptr<AbstractDataStore<T>> data_store,
                                std::unique_ptr<AbstractGraphStore> graph_store,
                                std::shared_ptr<AbstractDataStore<T>> pq_data_store)
            : _index_config(index_config), _max_points(index_config.basic_config.max_points),
              _num_frozen_pts(index_config.vamana_config.num_frozen_pts),
              _dynamic_index(index_config.vamana_config.dynamic_index),
              _enable_tags(index_config.vamana_config.enable_tags),
              _indexingMaxC(DEFAULT_MAXC),
              _query_scratch(nullptr),
              _pq_dist(index_config.vamana_config.pq_dist_build),
              _use_opq(index_config.vamana_config.use_opq),
              _num_pq_chunks(index_config.vamana_config.num_pq_chunks),
              _delete_set(new turbo::flat_hash_set<uint32_t>),
              _conc_consolidate(index_config.vamana_config.concurrent_consolidate) {
        if (_dynamic_index && !_enable_tags) {
            throw PolarisException("ERROR: Dynamic Indexing must have tags enabled.", -1, __PRETTY_FUNCTION__, __FILE__,
                                   __LINE__);
        }

        if (_pq_dist) {
            if (_dynamic_index)
                throw PolarisException("ERROR: Dynamic Indexing not supported with PQ distance based "
                                       "index construction",
                                       -1, __PRETTY_FUNCTION__, __FILE__, __LINE__);
            if (_index_config.basic_config.metric == polaris::MetricType::METRIC_INNER_PRODUCT)
                throw PolarisException("ERROR: Inner product metrics not yet supported "
                                       "with PQ distance "
                                       "base index",
                                       -1, __PRETTY_FUNCTION__, __FILE__, __LINE__);
        }

        if (_dynamic_index && _num_frozen_pts == 0) {
            _num_frozen_pts = 1;
        }
        // Sanity check. While logically it is correct, max_points = 0 causes
        // downstream problems.
        if (_max_points == 0) {
            _max_points = 1;
        }
        const size_t total_internal_points = _max_points + _num_frozen_pts;

        _start = (uint32_t) _max_points;

        _data_store = data_store;
        _pq_data_store = pq_data_store;
        _graph_store = std::move(graph_store);

        _locks = std::vector<non_recursive_mutex>(total_internal_points);
        if (_enable_tags) {
            _location_to_tag.reserve(total_internal_points);
            _tag_to_location.reserve(total_internal_points);
        }

        if (_dynamic_index) {
            this->enable_delete(); // enable delete by default for dynamic index
        }

        if (index_config.vamana_config.index_write_params != nullptr) {
            _indexingQueueSize = index_config.vamana_config.index_write_params->search_list_size;
            _indexingRange = index_config.vamana_config.index_write_params->max_degree;
            _indexingMaxC = index_config.vamana_config.index_write_params->max_occlusion_size;
            _indexingAlpha = index_config.vamana_config.index_write_params->alpha;
            _indexingThreads = index_config.vamana_config.index_write_params->num_threads;
            _saturate_graph = index_config.vamana_config.index_write_params->saturate_graph;

            if (index_config.vamana_config.index_search_params != nullptr) {
                uint32_t num_scratch_spaces =
                        index_config.vamana_config.index_search_params->num_search_threads + _indexingThreads;
                initialize_query_scratch(num_scratch_spaces,
                                         index_config.vamana_config.index_search_params->initial_search_list_size,
                                         _indexingQueueSize, _indexingRange, _indexingMaxC, _data_store->get_dims());
            }
        }
    }

    template<typename T>
    VamanaIndex<T>::VamanaIndex(MetricType m, const size_t dim, const size_t max_points,
                                const std::shared_ptr<IndexWriteParameters> index_parameters,
                                const std::shared_ptr<IndexSearchParams> index_search_params,
                                const size_t num_frozen_pts,
                                const bool dynamic_index, const bool enable_tags, const bool concurrent_consolidate,
                                const bool pq_dist_build, const size_t num_pq_chunks, const bool use_opq)
            : VamanaIndex(
            IndexConfigBuilder()
                    .with_metric(m)
                    .with_dimension(dim)
                    .with_max_points(max_points)
                    .vamana_with_index_write_params(index_parameters)
                    .vamana_with_index_search_params(index_search_params)
                    .vamana_with_num_frozen_pts(num_frozen_pts)
                    .vamana_is_dynamic_index(dynamic_index)
                    .vamana_is_enable_tags(enable_tags)
                    .vamana_is_concurrent_consolidate(concurrent_consolidate)
                    .vamana_is_pq_dist_build(pq_dist_build)
                    .vamana_with_num_pq_chunks(num_pq_chunks)
                    .vamana_is_use_opq(use_opq)
                    .with_data_type(polaris_type_to_name<T>())
                    .build_vamana(),
            IndexFactory::construct_datastore<T>(DataStoreStrategy::MEMORY,
                                                 (max_points == 0 ? (size_t) 1 : max_points) +
                                                 (dynamic_index && num_frozen_pts == 0 ? (size_t) 1 : num_frozen_pts),
                                                 dim, m),
            IndexFactory::construct_graphstore(GraphStoreStrategy::MEMORY,
                                               (max_points == 0 ? (size_t) 1 : max_points) +
                                               (dynamic_index && num_frozen_pts == 0 ? (size_t) 1 : num_frozen_pts),
                                               (size_t) ((index_parameters == nullptr ? 0
                                                                                      : index_parameters->max_degree) *
                                                         defaults::GRAPH_SLACK_FACTOR * 1.05))) {
        if (_pq_dist) {
            _pq_data_store = IndexFactory::construct_pq_datastore<T>(DataStoreStrategy::MEMORY,
                                                                     max_points + num_frozen_pts,
                                                                     dim, m, num_pq_chunks, use_opq);
        } else {
            _pq_data_store = _data_store;
        }
    }

    template<typename T>
    VamanaIndex<T>::~VamanaIndex() {
        // Ensure that no other activity is happening before dtor()
        std::unique_lock<std::shared_timed_mutex> ul(_update_lock);
        std::unique_lock<std::shared_timed_mutex> cl(_consolidate_lock);
        std::unique_lock<std::shared_timed_mutex> tl(_tag_lock);
        std::unique_lock<std::shared_timed_mutex> dl(_delete_lock);

        for (auto &lock: _locks) {
            LockGuard lg(lock);
        }

        if (_opt_graph != nullptr) {
            delete[] _opt_graph;
        }

        if (!_query_scratch.empty()) {
            ScratchStoreManager<InMemQueryScratch<T>> manager(_query_scratch);
            manager.destroy();
        }
    }

    template<typename T>
    void VamanaIndex<T>::initialize_query_scratch(uint32_t num_threads, uint32_t search_l, uint32_t indexing_l,
                                                  uint32_t r, uint32_t maxc, size_t dim) {
        for (uint32_t i = 0; i < num_threads; i++) {
            auto scratch = new InMemQueryScratch<T>(search_l, indexing_l, r, maxc, dim, _data_store->get_aligned_dim(),
                                                    _data_store->get_alignment_factor(), _pq_dist);
            _query_scratch.push(scratch);
        }
    }

    template<typename T>
    size_t VamanaIndex<T>::save_tags(std::string tags_file) {
        if (!_enable_tags) {
            polaris::cout << "Not saving tags as they are not enabled." << std::endl;
            return 0;
        }

        size_t tag_bytes_written;
        vid_t *tag_data = new vid_t[_nd + _num_frozen_pts];
        for (uint32_t i = 0; i < _nd; i++) {
            vid_t tag;
            if (_location_to_tag.try_get(i, tag)) {
                tag_data[i] = tag;
            } else {
                // catering to future when tagT can be any type.
                std::memset((char *) &tag_data[i], 0, sizeof(vid_t));
            }
        }
        if (_num_frozen_pts > 0) {
            std::memset((char *) &tag_data[_start], 0, sizeof(vid_t) * _num_frozen_pts);
        }
        try {
            tag_bytes_written = save_bin<vid_t>(tags_file, tag_data, _nd + _num_frozen_pts, 1);
        }
        catch (std::system_error &e) {
            throw FileException(tags_file, e, __PRETTY_FUNCTION__, __FILE__, __LINE__);
        }
        delete[] tag_data;
        return tag_bytes_written;
    }

    template<typename T>
    size_t VamanaIndex<T>::save_data(std::string data_file) {
        // Note: at this point, either _nd == _max_points or any frozen points have
        // been temporarily moved to _nd, so _nd + _num_frozen_pts is the valid
        // location limit.
        return _data_store->save(data_file, (location_t) (_nd + _num_frozen_pts));
    }

    // save the graph index on a file as an adjacency list. For each point,
    // first store the number of neighbors, and then the neighbor list (each as
    // 4 byte uint32_t)
    template<typename T>
    size_t VamanaIndex<T>::save_graph(std::string graph_file) {
        return _graph_store->store(graph_file, _nd + _num_frozen_pts, _num_frozen_pts, _start);
    }

    template<typename T>
    size_t VamanaIndex<T>::save_delete_list(const std::string &filename) {
        if (_delete_set->size() == 0) {
            return 0;
        }
        std::unique_ptr<uint32_t[]> delete_list = std::make_unique<uint32_t[]>(_delete_set->size());
        uint32_t i = 0;
        for (auto &del: *_delete_set) {
            delete_list[i++] = del;
        }
        return save_bin<uint32_t>(filename, delete_list.get(), _delete_set->size(), 1);
    }

    template<typename T>
    void VamanaIndex<T>::save(const char *filename, bool compact_before_save) {
        polaris::Timer timer;

        std::unique_lock<std::shared_timed_mutex> ul(_update_lock);
        std::unique_lock<std::shared_timed_mutex> cl(_consolidate_lock);
        std::unique_lock<std::shared_timed_mutex> tl(_tag_lock);
        std::unique_lock<std::shared_timed_mutex> dl(_delete_lock);

        if (compact_before_save) {
            compact_data();
            compact_frozen_point();
        } else {
            if (!_data_compacted) {
                throw PolarisException("VamanaIndex save for non-compacted index is not yet implemented", -1,
                                       __PRETTY_FUNCTION__, __FILE__,
                                       __LINE__);
            }
        }

        if (!_save_as_one_file) {
            std::string graph_file = std::string(filename);
            std::string tags_file = std::string(filename) + ".tags";
            std::string data_file = std::string(filename) + ".data";
            std::string delete_list_file = std::string(filename) + ".del";

            // Because the save_* functions use append mode, ensure that
            // the files are deleted before save. Ideally, we should check
            // the error code for delete_file, but will ignore now because
            // delete should succeed if save will succeed.
            collie::filesystem::remove(graph_file);
            save_graph(graph_file);
            collie::filesystem::remove(data_file);
            save_data(data_file);
            collie::filesystem::remove(tags_file);
            save_tags(tags_file);
            collie::filesystem::remove(delete_list_file);
            save_delete_list(delete_list_file);
        } else {
            polaris::cout << "Save index in a single file currently not supported. "
                             "Not saving the index."
                          << std::endl;
        }

        // If frozen points were temporarily compacted to _nd, move back to
        // _max_points.
        reposition_frozen_point_to_end();

        polaris::cout << "Time taken for save: " << timer.elapsed() / 1000000.0 << "s." << std::endl;
    }

    template<typename T>
    size_t VamanaIndex<T>::load_tags(const std::string tag_filename) {
        if (_enable_tags && !collie::filesystem::exists(tag_filename)) {
            polaris::cerr << "Tag file " << tag_filename << " does not exist!" << std::endl;
            throw polaris::PolarisException("Tag file " + tag_filename + " does not exist!", -1, __PRETTY_FUNCTION__,
                                            __FILE__,
                                            __LINE__);
        }
        if (!_enable_tags) {
            polaris::cout << "Tags not loaded as tags not enabled." << std::endl;
            return 0;
        }

        size_t file_dim, file_num_points;
        vid_t *tag_data;
        load_bin<vid_t>(std::string(tag_filename), tag_data, file_num_points, file_dim);

        if (file_dim != 1) {
            std::stringstream stream;
            stream << "ERROR: Found " << file_dim << " dimensions for tags,"
                   << "but tag file must have 1 dimension." << std::endl;
            polaris::cerr << stream.str() << std::endl;
            delete[] tag_data;
            throw polaris::PolarisException(stream.str(), -1, __PRETTY_FUNCTION__, __FILE__, __LINE__);
        }

        const size_t num_data_points = file_num_points - _num_frozen_pts;
        _location_to_tag.reserve(num_data_points);
        _tag_to_location.reserve(num_data_points);
        for (uint32_t i = 0; i < (uint32_t) num_data_points; i++) {
            vid_t tag = *(tag_data + i);
            if (_delete_set->find(i) == _delete_set->end()) {
                _location_to_tag.set(i, tag);
                _tag_to_location[tag] = i;
            }
        }
        polaris::cout << "Tags loaded." << std::endl;
        delete[] tag_data;
        return file_num_points;
    }

    template<typename T>
    size_t VamanaIndex<T>::load_data(std::string filename) {
        size_t file_dim, file_num_points;
        if (!collie::filesystem::exists(filename)) {
            std::stringstream stream;
            stream << "ERROR: data file " << filename << " does not exist." << std::endl;
            polaris::cerr << stream.str() << std::endl;
            throw polaris::PolarisException(stream.str(), -1, __PRETTY_FUNCTION__, __FILE__, __LINE__);
        }
        polaris::get_bin_metadata(filename, file_num_points, file_dim);

        // since we are loading a new dataset, _empty_slots must be cleared
        _empty_slots.clear();

        if (file_dim != _index_config.basic_config.dimension) {
            std::stringstream stream;
            stream << "ERROR: Driver requests loading " << _index_config.basic_config.dimension << " dimension,"
                   << "but file has " << file_dim << " dimension." << std::endl;
            polaris::cerr << stream.str() << std::endl;
            throw polaris::PolarisException(stream.str(), -1, __PRETTY_FUNCTION__, __FILE__, __LINE__);
        }

        if (file_num_points > _max_points + _num_frozen_pts) {
            // update and tag lock acquired in load() before calling load_data
            resize(file_num_points - _num_frozen_pts);
        }

        _data_store->load(filename); // offset == 0.
        return file_num_points;
    }

    template<typename T>
    size_t VamanaIndex<T>::load_delete_set(const std::string &filename) {
        std::unique_ptr<uint32_t[]> delete_list;
        size_t npts, ndim;

        polaris::load_bin<uint32_t>(filename, delete_list, npts, ndim);
        assert(ndim == 1);
        for (uint32_t i = 0; i < npts; i++) {
            _delete_set->insert(delete_list[i]);
        }
        return npts;
    }

    // load the index from file and update the max_degree, cur (navigating
    // node loc), and _final_graph (adjacency list)
    template<typename T>
    void VamanaIndex<T>::load(const char *filename, uint32_t num_threads, uint32_t search_l) {
        std::unique_lock<std::shared_timed_mutex> ul(_update_lock);
        std::unique_lock<std::shared_timed_mutex> cl(_consolidate_lock);
        std::unique_lock<std::shared_timed_mutex> tl(_tag_lock);
        std::unique_lock<std::shared_timed_mutex> dl(_delete_lock);

        _has_built = true;

        size_t tags_file_num_pts = 0, graph_num_pts = 0, data_file_num_pts = 0;

        std::string mem_index_file(filename);

        if (!_save_as_one_file) {
            // For DLVS Store, we will not support saving the index in multiple
            // files.
            std::string data_file = std::string(filename) + ".data";
            std::string tags_file = std::string(filename) + ".tags";
            std::string delete_set_file = std::string(filename) + ".del";
            std::string graph_file = std::string(filename);
            data_file_num_pts = load_data(data_file);
            if (collie::filesystem::exists(delete_set_file)) {
                load_delete_set(delete_set_file);
            }
            if (_enable_tags) {
                tags_file_num_pts = load_tags(tags_file);
            }
            graph_num_pts = load_graph(graph_file, data_file_num_pts);
        } else {
            polaris::cout << "Single index file saving/loading support not yet "
                             "enabled. Not loading the index."
                          << std::endl;
            return;
        }

        if (data_file_num_pts != graph_num_pts || (data_file_num_pts != tags_file_num_pts && _enable_tags)) {
            std::stringstream stream;
            stream << "ERROR: When loading index, loaded " << data_file_num_pts << " points from datafile, "
                   << graph_num_pts << " from graph, and " << tags_file_num_pts
                   << " tags, with num_frozen_pts being set to " << _num_frozen_pts << " in constructor." << std::endl;
            polaris::cerr << stream.str() << std::endl;
            throw polaris::PolarisException(stream.str(), -1, __PRETTY_FUNCTION__, __FILE__, __LINE__);
        }

        _nd = data_file_num_pts - _num_frozen_pts;
        _empty_slots.clear();
        _empty_slots.reserve(_max_points);
        for (auto i = _nd; i < _max_points; i++) {
            _empty_slots.insert((uint32_t) i);
        }

        reposition_frozen_point_to_end();
        polaris::cout << "Num frozen points:" << _num_frozen_pts << " _nd: " << _nd << " _start: " << _start
                      << " size(_location_to_tag): " << _location_to_tag.size()
                      << " size(_tag_to_location):" << _tag_to_location.size() << " Max points: " << _max_points
                      << std::endl;

        // For incremental index, _query_scratch is initialized in the constructor.
        // For the bulk index, the params required to initialize _query_scratch
        // are known only at load time, hence this check and the call to
        // initialize_q_s().
        if (_query_scratch.size() == 0) {
            initialize_query_scratch(num_threads, search_l, search_l, (uint32_t) _graph_store->get_max_range_of_graph(),
                                     _indexingMaxC, _index_config.basic_config.dimension);
        }
    }

    template<typename T>
    size_t VamanaIndex<T>::get_graph_num_frozen_points(const std::string &graph_file) {
        size_t expected_file_size;
        uint32_t max_observed_degree, start;
        size_t file_frozen_pts;

        std::ifstream in;
        in.exceptions(std::ios::badbit | std::ios::failbit);

        in.open(graph_file, std::ios::binary);
        in.read((char *) &expected_file_size, sizeof(size_t));
        in.read((char *) &max_observed_degree, sizeof(uint32_t));
        in.read((char *) &start, sizeof(uint32_t));
        in.read((char *) &file_frozen_pts, sizeof(size_t));

        return file_frozen_pts;
    }


    template<typename T>
    size_t VamanaIndex<T>::load_graph(std::string filename, size_t expected_num_points) {
        auto res = _graph_store->load(filename, expected_num_points);
        _start = std::get<1>(res);
        _num_frozen_pts = std::get<2>(res);
        return std::get<0>(res);
    }

    template<typename T>
    int VamanaIndex<T>::_get_vector_by_tag(TagType &tag, DataType &vec) {
        try {
            vid_t tag_val = std::any_cast<vid_t>(tag);
            T *vec_val = std::any_cast<T *>(vec);
            return this->get_vector_by_tag(tag_val, vec_val);
        }
        catch (const std::bad_any_cast &e) {
            throw PolarisException(
                    "Error: bad any cast while performing _get_vector_by_tags() " + std::string(e.what()), -1);
        }
        catch (const std::exception &e) {
            throw PolarisException("Error: " + std::string(e.what()), -1);
        }
    }

    template<typename T>
    int VamanaIndex<T>::get_vector_by_tag(vid_t &tag, T *vec) {
        std::shared_lock<std::shared_timed_mutex> lock(_tag_lock);
        if (_tag_to_location.find(tag) == _tag_to_location.end()) {
            polaris::cout << "Tag " << get_tag_string(tag) << " does not exist" << std::endl;
            return -1;
        }

        location_t location = _tag_to_location[tag];
        _data_store->get_vector(location, vec);

        return 0;
    }

    template<typename T>
    uint32_t VamanaIndex<T>::calculate_entry_point() {
        // REFACTOR TODO: This function does not support multi-threaded calculation of medoid.
        // Must revisit if perf is a concern.
        return _data_store->calculate_medoid();
    }

    template<typename T>
    std::vector<uint32_t> VamanaIndex<T>::get_init_ids() {
        std::vector<uint32_t> init_ids;
        init_ids.reserve(1 + _num_frozen_pts);

        init_ids.emplace_back(_start);

        for (uint32_t frozen = (uint32_t) _max_points; frozen < _max_points + _num_frozen_pts; frozen++) {
            if (frozen != _start) {
                init_ids.emplace_back(frozen);
            }
        }

        return init_ids;
    }

    template<typename T>
    turbo::ResultStatus<std::pair<uint32_t, uint32_t>>
    VamanaIndex<T>::iterate_to_fixed_point(InMemQueryScratch<T> *scratch, const uint32_t Lsize,
                                           const std::vector<uint32_t> &init_ids,
                                           const BaseSearchCondition *condition,
                                           bool search_invocation) {
        std::vector<Neighbor> &expanded_nodes = scratch->pool();
        NeighborPriorityQueue &best_L_nodes = scratch->best_l_nodes();
        best_L_nodes.reserve(Lsize);
        turbo::flat_hash_set<uint32_t> &inserted_into_pool_rs = scratch->inserted_into_pool_rs();
        collie::dynamic_bitset<> &inserted_into_pool_bs = scratch->inserted_into_pool_bs();
        std::vector<uint32_t> &id_scratch = scratch->id_scratch();
        std::vector<float> &dist_scratch = scratch->dist_scratch();
        if (!id_scratch.empty()) {
            return turbo::make_status(turbo::kInvalidArgument,
                                      "ERROR: Clear scratch space before passing [id_scratch.size()].");
        }
        if (!condition) {
            return turbo::make_status(turbo::kInvalidArgument, "ERROR: Condition is not provided.");
        }

        T *aligned_query = scratch->aligned_query();

        float *pq_dists = nullptr;

        _pq_data_store->preprocess_query(aligned_query, scratch);

        if (!expanded_nodes.empty() || !id_scratch.empty()) {
            return turbo::make_status(turbo::kInvalidArgument, "ERROR: Clear scratch space before passing.");
        }

        // Decide whether to use bitset or robin set to mark visited nodes
        auto total_num_points = _max_points + _num_frozen_pts;
        bool fast_iterate = total_num_points <= MAX_POINTS_FOR_USING_BITSET;

        if (fast_iterate) {
            if (inserted_into_pool_bs.size() < total_num_points) {
                // hopefully using 2X will reduce the number of allocations.
                auto resize_size =
                        2 * total_num_points > MAX_POINTS_FOR_USING_BITSET ? MAX_POINTS_FOR_USING_BITSET : 2 *
                                                                                                           total_num_points;
                inserted_into_pool_bs.resize(resize_size);
            }
        }

        // Lambda to determine if a node has been visited
        auto is_not_visited = [this, fast_iterate, &inserted_into_pool_bs, &inserted_into_pool_rs](const uint32_t id) {
            return fast_iterate ? inserted_into_pool_bs[id] == 0
                                : inserted_into_pool_rs.find(id) == inserted_into_pool_rs.end();
        };

        // Lambda to batch compute query<-> node distances in PQ space
        auto compute_dists = [this, scratch, pq_dists](const std::vector<uint32_t> &ids,
                                                       std::vector<float> &dists_out) {
            _pq_data_store->get_distance(scratch->aligned_query(), ids, dists_out, scratch);
        };

        // Initialize the candidate pool with starting points
        vid_t tmp_vid;
        for (auto id: init_ids) {
            if (id >= _max_points + _num_frozen_pts) {
                polaris::cerr << "Out of range loc found as an edge : " << id << std::endl;
                throw polaris::PolarisException(std::string("Wrong loc") + std::to_string(id), -1, __PRETTY_FUNCTION__,
                                                __FILE__,
                                                __LINE__);
            }
            if (!_location_to_tag.try_get(id, tmp_vid)) {
                return turbo::make_status(turbo::kInvalidArgument, "ERROR: Tag not found for location.");
            }
            if (condition->is_in_blacklist(tmp_vid)) {
                continue;
            }

            if (is_not_visited(id)) {
                if (fast_iterate) {
                    inserted_into_pool_bs[id] = 1;
                } else {
                    inserted_into_pool_rs.insert(id);
                }

                float distance;
                uint32_t ids[] = {id};
                float distances[] = {std::numeric_limits<float>::max()};
                _pq_data_store->get_distance(aligned_query, ids, 1, distances, scratch);
                distance = distances[0];

                Neighbor nn = Neighbor(id, distance);
                best_L_nodes.insert(nn);
            }
        }

        uint32_t hops = 0;
        uint32_t cmps = 0;

        while (best_L_nodes.has_unexpanded_node()) {
            auto nbr = best_L_nodes.closest_unexpanded();
            auto n = nbr.id;

            // Add node to expanded nodes to create pool for prune later
            if (!search_invocation) {
                expanded_nodes.emplace_back(nbr);
            }

            // Find which of the nodes in des have not been visited before
            id_scratch.clear();
            dist_scratch.clear();
            if (_dynamic_index) {
                LockGuard guard(_locks[n]);
                for (auto id: _graph_store->get_neighbours(n)) {
                    assert(id < _max_points + _num_frozen_pts);

                    if (!_location_to_tag.try_get(id, tmp_vid)) {
                        return turbo::make_status(turbo::kInvalidArgument, "ERROR: Tag not found for location.");
                    }
                    if (condition->is_in_blacklist(tmp_vid)) {
                        continue;
                    }

                    if (is_not_visited(id)) {
                        id_scratch.push_back(id);
                    }
                }
            } else {
                _locks[n].lock();
                auto nbrs = _graph_store->get_neighbours(n);
                _locks[n].unlock();
                for (auto id: nbrs) {
                    assert(id < _max_points + _num_frozen_pts);

                    if (!_location_to_tag.try_get(id, tmp_vid)) {
                        return turbo::make_status(turbo::kInvalidArgument, "ERROR: Tag not found for location.");
                    }
                    if (condition->is_in_blacklist(tmp_vid)) {
                        continue;
                    }

                    if (is_not_visited(id)) {
                        id_scratch.push_back(id);
                    }
                }
            }

            // Mark nodes visited
            for (auto id: id_scratch) {
                if (fast_iterate) {
                    inserted_into_pool_bs[id] = 1;
                } else {
                    inserted_into_pool_rs.insert(id);
                }
            }

            assert(dist_scratch.capacity() >= id_scratch.size());
            compute_dists(id_scratch, dist_scratch);
            cmps += (uint32_t) id_scratch.size();

            // Insert <id, dist> pairs into the pool of candidates
            for (size_t m = 0; m < id_scratch.size(); ++m) {
                best_L_nodes.insert(Neighbor(id_scratch[m], dist_scratch[m]));
            }
        }
        return std::make_pair(hops, cmps);
    }

    template<typename T>
    std::pair<uint32_t, uint32_t> VamanaIndex<T>::iterate_to_fixed_point(
            InMemQueryScratch<T> *scratch, const uint32_t Lsize, const std::vector<uint32_t> &init_ids,
            bool search_invocation) {
        std::vector<Neighbor> &expanded_nodes = scratch->pool();
        NeighborPriorityQueue &best_L_nodes = scratch->best_l_nodes();
        best_L_nodes.reserve(Lsize);
        turbo::flat_hash_set<uint32_t> &inserted_into_pool_rs = scratch->inserted_into_pool_rs();
        collie::dynamic_bitset<> &inserted_into_pool_bs = scratch->inserted_into_pool_bs();
        std::vector<uint32_t> &id_scratch = scratch->id_scratch();
        std::vector<float> &dist_scratch = scratch->dist_scratch();
        assert(id_scratch.size() == 0);

        T *aligned_query = scratch->aligned_query();

        float *pq_dists = nullptr;

        _pq_data_store->preprocess_query(aligned_query, scratch);

        if (expanded_nodes.size() > 0 || id_scratch.size() > 0) {
            throw PolarisException("ERROR: Clear scratch space before passing.", -1, __PRETTY_FUNCTION__, __FILE__,
                                   __LINE__);
        }

        // Decide whether to use bitset or robin set to mark visited nodes
        auto total_num_points = _max_points + _num_frozen_pts;
        bool fast_iterate = total_num_points <= MAX_POINTS_FOR_USING_BITSET;

        if (fast_iterate) {
            if (inserted_into_pool_bs.size() < total_num_points) {
                // hopefully using 2X will reduce the number of allocations.
                auto resize_size =
                        2 * total_num_points > MAX_POINTS_FOR_USING_BITSET ? MAX_POINTS_FOR_USING_BITSET : 2 *
                                                                                                           total_num_points;
                inserted_into_pool_bs.resize(resize_size);
            }
        }

        // Lambda to determine if a node has been visited
        auto is_not_visited = [this, fast_iterate, &inserted_into_pool_bs, &inserted_into_pool_rs](const uint32_t id) {
            return fast_iterate ? inserted_into_pool_bs[id] == 0
                                : inserted_into_pool_rs.find(id) == inserted_into_pool_rs.end();
        };

        // Lambda to batch compute query<-> node distances in PQ space
        auto compute_dists = [this, scratch, pq_dists](const std::vector<uint32_t> &ids,
                                                       std::vector<float> &dists_out) {
            _pq_data_store->get_distance(scratch->aligned_query(), ids, dists_out, scratch);
        };

        // Initialize the candidate pool with starting points
        for (auto id: init_ids) {
            if (id >= _max_points + _num_frozen_pts) {
                polaris::cerr << "Out of range loc found as an edge : " << id << std::endl;
                throw polaris::PolarisException(std::string("Wrong loc") + std::to_string(id), -1, __PRETTY_FUNCTION__,
                                                __FILE__,
                                                __LINE__);
            }

            if (is_not_visited(id)) {
                if (fast_iterate) {
                    inserted_into_pool_bs[id] = 1;
                } else {
                    inserted_into_pool_rs.insert(id);
                }

                float distance;
                uint32_t ids[] = {id};
                float distances[] = {std::numeric_limits<float>::max()};
                _pq_data_store->get_distance(aligned_query, ids, 1, distances, scratch);
                distance = distances[0];

                Neighbor nn = Neighbor(id, distance);
                best_L_nodes.insert(nn);
            }
        }

        uint32_t hops = 0;
        uint32_t cmps = 0;

        while (best_L_nodes.has_unexpanded_node()) {
            auto nbr = best_L_nodes.closest_unexpanded();
            auto n = nbr.id;

            // Add node to expanded nodes to create pool for prune later
            if (!search_invocation) {
                expanded_nodes.emplace_back(nbr);
            }

            // Find which of the nodes in des have not been visited before
            id_scratch.clear();
            dist_scratch.clear();
            if (_dynamic_index) {
                LockGuard guard(_locks[n]);
                for (auto id: _graph_store->get_neighbours(n)) {
                    assert(id < _max_points + _num_frozen_pts);
                    if (is_not_visited(id)) {
                        id_scratch.push_back(id);
                    }
                }
            } else {
                _locks[n].lock();
                auto nbrs = _graph_store->get_neighbours(n);
                _locks[n].unlock();
                for (auto id: nbrs) {
                    assert(id < _max_points + _num_frozen_pts);
                    if (is_not_visited(id)) {
                        id_scratch.push_back(id);
                    }
                }
            }

            // Mark nodes visited
            for (auto id: id_scratch) {
                if (fast_iterate) {
                    inserted_into_pool_bs[id] = 1;
                } else {
                    inserted_into_pool_rs.insert(id);
                }
            }

            assert(dist_scratch.capacity() >= id_scratch.size());
            compute_dists(id_scratch, dist_scratch);
            cmps += (uint32_t) id_scratch.size();

            // Insert <id, dist> pairs into the pool of candidates
            for (size_t m = 0; m < id_scratch.size(); ++m) {
                best_L_nodes.insert(Neighbor(id_scratch[m], dist_scratch[m]));
            }
        }
        return std::make_pair(hops, cmps);
    }

    template<typename T>
    void VamanaIndex<T>::search_for_point_and_prune(int location, uint32_t Lindex,
                                                    std::vector<uint32_t> &pruned_list,
                                                    InMemQueryScratch<T> *scratch, uint32_t filteredLindex) {
        const std::vector<uint32_t> init_ids = get_init_ids();
        _data_store->get_vector(location, scratch->aligned_query());
        iterate_to_fixed_point(scratch, Lindex, init_ids, false);

        auto &pool = scratch->pool();

        for (uint32_t i = 0; i < pool.size(); i++) {
            if (pool[i].id == (uint32_t) location) {
                pool.erase(pool.begin() + i);
                i--;
            }
        }

        if (pruned_list.size() > 0) {
            throw polaris::PolarisException("ERROR: non-empty pruned_list passed", -1, __PRETTY_FUNCTION__, __FILE__,
                                            __LINE__);
        }

        prune_neighbors(location, pool, pruned_list, scratch);

        assert(!pruned_list.empty());
        assert(_graph_store->get_total_points() == _max_points + _num_frozen_pts);
    }

    template<typename T>
    void VamanaIndex<T>::occlude_list(const uint32_t location, std::vector<Neighbor> &pool, const float alpha,
                                      const uint32_t degree, const uint32_t maxc, std::vector<uint32_t> &result,
                                      InMemQueryScratch<T> *scratch,
                                      const turbo::flat_hash_set<uint32_t> *const delete_set_ptr) {
        if (pool.size() == 0)
            return;

        // Truncate pool at maxc and initialize scratch spaces
        assert(std::is_sorted(pool.begin(), pool.end()));
        assert(result.size() == 0);
        if (pool.size() > maxc)
            pool.resize(maxc);
        std::vector<float> &occlude_factor = scratch->occlude_factor();
        // occlude_list can be called with the same scratch more than once by
        // search_for_point_and_add_link through inter_insert.
        occlude_factor.clear();
        // Initialize occlude_factor to pool.size() many 0.0f values for correctness
        occlude_factor.insert(occlude_factor.end(), pool.size(), 0.0f);

        float cur_alpha = 1;
        while (cur_alpha <= alpha && result.size() < degree) {
            // used for MIPS, where we store a value of eps in cur_alpha to
            // denote pruned out entries which we can skip in later rounds.
            float eps = cur_alpha + 0.01f;

            for (auto iter = pool.begin(); result.size() < degree && iter != pool.end(); ++iter) {
                if (occlude_factor[iter - pool.begin()] > cur_alpha) {
                    continue;
                }
                // Set the entry to float::max so that is not considered again
                occlude_factor[iter - pool.begin()] = std::numeric_limits<float>::max();
                // Add the entry to the result if its not been deleted, and doesn't
                // add a self loop
                if (delete_set_ptr == nullptr || delete_set_ptr->find(iter->id) == delete_set_ptr->end()) {
                    if (iter->id != location) {
                        result.push_back(iter->id);
                    }
                }

                // Update occlude factor for points from iter+1 to pool.end()
                for (auto iter2 = iter + 1; iter2 != pool.end(); iter2++) {
                    auto t = iter2 - pool.begin();
                    if (occlude_factor[t] > alpha)
                        continue;

                    float djk = _data_store->get_distance(iter2->id, iter->id);
                    if (_index_config.basic_config.metric == polaris::MetricType::METRIC_L2 ||
                        _index_config.basic_config.metric == polaris::MetricType::METRIC_COSINE) {
                        occlude_factor[t] = (djk == 0) ? std::numeric_limits<float>::max()
                                                       : std::max(occlude_factor[t], iter2->distance / djk);
                    } else if (_index_config.basic_config.metric == polaris::MetricType::METRIC_INNER_PRODUCT) {
                        // Improvization for flipping max and min dist for MIPS
                        float x = -iter2->distance;
                        float y = -djk;
                        if (y > cur_alpha * x) {
                            occlude_factor[t] = std::max(occlude_factor[t], eps);
                        }
                    }
                }
            }
            cur_alpha *= 1.2f;
        }
    }

    template<typename T>
    void VamanaIndex<T>::prune_neighbors(const uint32_t location, std::vector<Neighbor> &pool,
                                         std::vector<uint32_t> &pruned_list, InMemQueryScratch<T> *scratch) {
        prune_neighbors(location, pool, _indexingRange, _indexingMaxC, _indexingAlpha, pruned_list, scratch);
    }

    template<typename T>
    void
    VamanaIndex<T>::prune_neighbors(const uint32_t location, std::vector<Neighbor> &pool, const uint32_t range,
                                    const uint32_t max_candidate_size, const float alpha,
                                    std::vector<uint32_t> &pruned_list, InMemQueryScratch<T> *scratch) {
        if (pool.size() == 0) {
            // if the pool is empty, behave like a noop
            pruned_list.clear();
            return;
        }

        // If using _pq_build, over-write the PQ distances with actual distances
        // REFACTOR PQ: TODO: How to get rid of this!?
        if (_pq_dist) {
            for (auto &ngh: pool)
                ngh.distance = _data_store->get_distance(ngh.id, location);
        }

        // sort the pool based on distance to query and prune it with occlude_list
        std::sort(pool.begin(), pool.end());
        pruned_list.clear();
        pruned_list.reserve(range);

        occlude_list(location, pool, alpha, range, max_candidate_size, pruned_list, scratch);
        assert(pruned_list.size() <= range);

        if (_saturate_graph && alpha > 1) {
            for (const auto &node: pool) {
                if (pruned_list.size() >= range)
                    break;
                if ((std::find(pruned_list.begin(), pruned_list.end(), node.id) == pruned_list.end()) &&
                    node.id != location)
                    pruned_list.push_back(node.id);
            }
        }
    }

    template<typename T>
    void VamanaIndex<T>::inter_insert(uint32_t n, std::vector<uint32_t> &pruned_list, const uint32_t range,
                                      InMemQueryScratch<T> *scratch) {
        const auto &src_pool = pruned_list;

        assert(!src_pool.empty());

        for (auto des: src_pool) {
            // des.loc is the loc of the neighbors of n
            assert(des < _max_points + _num_frozen_pts);
            // des_pool contains the neighbors of the neighbors of n
            std::vector<uint32_t> copy_of_neighbors;
            bool prune_needed = false;
            {
                LockGuard guard(_locks[des]);
                auto &des_pool = _graph_store->get_neighbours(des);
                if (std::find(des_pool.begin(), des_pool.end(), n) == des_pool.end()) {
                    if (des_pool.size() < (uint64_t) (defaults::GRAPH_SLACK_FACTOR * range)) {
                        // des_pool.emplace_back(n);
                        _graph_store->add_neighbour(des, n);
                        prune_needed = false;
                    } else {
                        copy_of_neighbors.reserve(des_pool.size() + 1);
                        copy_of_neighbors = des_pool;
                        copy_of_neighbors.push_back(n);
                        prune_needed = true;
                    }
                }
            } // des lock is released by this point

            if (prune_needed) {
                turbo::flat_hash_set<uint32_t> dummy_visited(0);
                std::vector<Neighbor> dummy_pool(0);

                size_t reserveSize = (size_t) (std::ceil(1.05 * defaults::GRAPH_SLACK_FACTOR * range));
                dummy_visited.reserve(reserveSize);
                dummy_pool.reserve(reserveSize);

                for (auto cur_nbr: copy_of_neighbors) {
                    if (dummy_visited.find(cur_nbr) == dummy_visited.end() && cur_nbr != des) {
                        float dist = _data_store->get_distance(des, cur_nbr);
                        dummy_pool.emplace_back(Neighbor(cur_nbr, dist));
                        dummy_visited.insert(cur_nbr);
                    }
                }
                std::vector<uint32_t> new_out_neighbors;
                prune_neighbors(des, dummy_pool, new_out_neighbors, scratch);
                {
                    LockGuard guard(_locks[des]);

                    _graph_store->set_neighbours(des, new_out_neighbors);
                }
            }
        }
    }

    template<typename T>
    void VamanaIndex<T>::inter_insert(uint32_t n, std::vector<uint32_t> &pruned_list,
                                      InMemQueryScratch<T> *scratch) {
        inter_insert(n, pruned_list, _indexingRange, scratch);
    }

    template<typename T>
    void VamanaIndex<T>::link() {
        uint32_t num_threads = _indexingThreads;
        if (num_threads != 0)
            omp_set_num_threads(num_threads);

        /* visit_order is a vector that is initialized to the entire graph */
        std::vector<uint32_t> visit_order;
        std::vector<polaris::Neighbor> pool, tmp;
        turbo::flat_hash_set<uint32_t> visited;
        visit_order.reserve(_nd + _num_frozen_pts);
        for (uint32_t i = 0; i < (uint32_t) _nd; i++) {
            visit_order.emplace_back(i);
        }

        // If there are any frozen points, add them all.
        for (uint32_t frozen = (uint32_t) _max_points; frozen < _max_points + _num_frozen_pts; frozen++) {
            visit_order.emplace_back(frozen);
        }

        // if there are frozen points, the first such one is set to be the _start
        if (_num_frozen_pts > 0)
            _start = (uint32_t) _max_points;
        else
            _start = calculate_entry_point();

        polaris::Timer link_timer;

#pragma omp parallel for schedule(dynamic, 2048)
        for (int64_t node_ctr = 0; node_ctr < (int64_t) (visit_order.size()); node_ctr++) {
            auto node = visit_order[node_ctr];

            // Find and add appropriate graph edges
            ScratchStoreManager<InMemQueryScratch<T>> manager(_query_scratch);
            auto scratch = manager.scratch_space();
            std::vector<uint32_t> pruned_list;
            search_for_point_and_prune(node, _indexingQueueSize, pruned_list, scratch);
            assert(pruned_list.size() > 0);

            {
                LockGuard guard(_locks[node]);

                _graph_store->set_neighbours(node, pruned_list);
                assert(_graph_store->get_neighbours((location_t) node).size() <= _indexingRange);
            }

            inter_insert(node, pruned_list, scratch);

            if (node_ctr % 100000 == 0) {
                polaris::cout << "\r" << (100.0 * node_ctr) / (visit_order.size()) << "% of index build completed."
                              << std::flush;
            }
        }

        if (_nd > 0) {
            polaris::cout << "Starting final cleanup.." << std::flush;
        }
#pragma omp parallel for schedule(dynamic, 2048)
        for (int64_t node_ctr = 0; node_ctr < (int64_t) (visit_order.size()); node_ctr++) {
            auto node = visit_order[node_ctr];
            if (_graph_store->get_neighbours((location_t) node).size() > _indexingRange) {
                ScratchStoreManager<InMemQueryScratch<T>> manager(_query_scratch);
                auto scratch = manager.scratch_space();

                turbo::flat_hash_set<uint32_t> dummy_visited(0);
                std::vector<Neighbor> dummy_pool(0);
                std::vector<uint32_t> new_out_neighbors;

                for (auto cur_nbr: _graph_store->get_neighbours((location_t) node)) {
                    if (dummy_visited.find(cur_nbr) == dummy_visited.end() && cur_nbr != node) {
                        float dist = _data_store->get_distance(node, cur_nbr);
                        dummy_pool.emplace_back(Neighbor(cur_nbr, dist));
                        dummy_visited.insert(cur_nbr);
                    }
                }
                prune_neighbors(node, dummy_pool, new_out_neighbors, scratch);

                _graph_store->clear_neighbours((location_t) node);
                _graph_store->set_neighbours((location_t) node, new_out_neighbors);
            }
        }
        if (_nd > 0) {
            polaris::cout << "done. Link time: " << ((double) link_timer.elapsed() / (double) 1000000) << "s"
                          << std::endl;
        }
    }

    // REFACTOR
    template<typename T>
    void VamanaIndex<T>::set_start_points(const T *data, size_t data_count) {
        std::unique_lock<std::shared_timed_mutex> ul(_update_lock);
        std::unique_lock<std::shared_timed_mutex> tl(_tag_lock);
        if (_nd > 0)
            throw PolarisException("Can not set starting point for a non-empty index", -1, __PRETTY_FUNCTION__,
                                   __FILE__, __LINE__);

        if (data_count != _num_frozen_pts * _index_config.basic_config.dimension)
            throw PolarisException("Invalid number of points", -1, __PRETTY_FUNCTION__, __FILE__, __LINE__);

        //     memcpy(_data + _aligned_dim * _max_points, data, _aligned_dim *
        //     sizeof(T) * _num_frozen_pts);
        for (location_t i = 0; i < _num_frozen_pts; i++) {
            _data_store->set_vector((location_t) (i + _max_points), data + i * _index_config.basic_config.dimension);
        }
        _has_built = true;
        polaris::cout << "VamanaIndex start points set: #" << _num_frozen_pts << std::endl;
    }

    template<typename T>
    void VamanaIndex<T>::_set_start_points_at_random(DataType radius, uint32_t random_seed) {
        try {
            T radius_to_use = std::any_cast<T>(radius);
            this->set_start_points_at_random(radius_to_use, random_seed);
        }
        catch (const std::bad_any_cast &e) {
            throw PolarisException(
                    "Error: bad any cast while performing _set_start_points_at_random() " + std::string(e.what()), -1);
        }
        catch (const std::exception &e) {
            throw PolarisException("Error: " + std::string(e.what()), -1);
        }
    }

    template<typename T>
    void VamanaIndex<T>::set_start_points_at_random(T radius, uint32_t random_seed) {
        std::mt19937 gen{random_seed};
        std::normal_distribution<> d{0.0, 1.0};

        std::vector<T> points_data;
        points_data.reserve(_index_config.basic_config.dimension * _num_frozen_pts);
        std::vector<double> real_vec(_index_config.basic_config.dimension);

        for (size_t frozen_point = 0; frozen_point < _num_frozen_pts; frozen_point++) {
            double norm_sq = 0.0;
            for (size_t i = 0; i < _index_config.basic_config.dimension; ++i) {
                auto r = d(gen);
                real_vec[i] = r;
                norm_sq += r * r;
            }

            const double norm = std::sqrt(norm_sq);
            for (auto iter: real_vec)
                points_data.push_back(static_cast<T>(iter * radius / norm));
        }

        set_start_points(points_data.data(), points_data.size());
    }

    template<typename T>
    void VamanaIndex<T>::build_with_data_populated(const std::vector<vid_t> &tags) {
        polaris::cout << "Starting index build with " << _nd << " points... " << std::endl;

        if (_nd < 1)
            throw PolarisException("Error: Trying to build an index with 0 points", -1, __PRETTY_FUNCTION__, __FILE__,
                                   __LINE__);

        if (_enable_tags && tags.size() != _nd) {
            std::stringstream stream;
            stream << "ERROR: Driver requests loading " << _nd << " points from file,"
                   << "but tags vector is of size " << tags.size() << "." << std::endl;
            polaris::cerr << stream.str() << std::endl;
            throw polaris::PolarisException(stream.str(), -1, __PRETTY_FUNCTION__, __FILE__, __LINE__);
        }
        if (_enable_tags) {
            for (size_t i = 0; i < tags.size(); ++i) {
                _tag_to_location[tags[i]] = (uint32_t) i;
                _location_to_tag.set(static_cast<uint32_t>(i), tags[i]);
            }
        }

        uint32_t index_R = _indexingRange;
        uint32_t num_threads_index = _indexingThreads;
        uint32_t index_L = _indexingQueueSize;
        uint32_t maxc = _indexingMaxC;

        if (_query_scratch.size() == 0) {
            initialize_query_scratch(5 + num_threads_index, index_L, index_L, index_R, maxc,
                                     _data_store->get_aligned_dim());
        }

        generate_frozen_point();
        link();

        size_t max = 0, min = SIZE_MAX, total = 0, cnt = 0;
        for (size_t i = 0; i < _nd; i++) {
            auto &pool = _graph_store->get_neighbours((location_t) i);
            max = std::max(max, pool.size());
            min = std::min(min, pool.size());
            total += pool.size();
            if (pool.size() < 2)
                cnt++;
        }
        polaris::cout << "VamanaIndex built with degree: max:" << max << "  avg:"
                      << (float) total / (float) (_nd + _num_frozen_pts)
                      << "  min:" << min << "  count(deg<2):" << cnt << std::endl;

        _has_built = true;
    }

    template<typename T>
    void VamanaIndex<T>::_build(const DataType &data, const size_t num_points_to_load, TagVector &tags) {
        try {
            this->build(std::any_cast<const T *>(data), num_points_to_load, tags.get<const std::vector<vid_t>>());
        }
        catch (const std::bad_any_cast &e) {
            throw PolarisException("Error: bad any cast in while building index. " + std::string(e.what()), -1);
        }
        catch (const std::exception &e) {
            throw PolarisException("Error" + std::string(e.what()), -1);
        }
    }

    template<typename T>
    void VamanaIndex<T>::build(const T *data, const size_t num_points_to_load, const std::vector<vid_t> &tags) {
        if (num_points_to_load == 0) {
            throw PolarisException("Do not call build with 0 points", -1, __PRETTY_FUNCTION__, __FILE__, __LINE__);
        }
        if (_pq_dist) {
            throw PolarisException("ERROR: DO not use this build interface with PQ distance", -1, __PRETTY_FUNCTION__,
                                   __FILE__,
                                   __LINE__);
        }

        std::unique_lock<std::shared_timed_mutex> ul(_update_lock);

        {
            std::unique_lock<std::shared_timed_mutex> tl(_tag_lock);
            _nd = num_points_to_load;

            _data_store->populate_data(data, (location_t) num_points_to_load);
        }

        build_with_data_populated(tags);
    }

    template<typename T>
    void VamanaIndex<T>::build(const char *filename, const size_t num_points_to_load,
                               const std::vector<vid_t> &tags) {
        // idealy this should call build_filtered_index based on params passed

        std::unique_lock<std::shared_timed_mutex> ul(_update_lock);

        // error checks
        if (num_points_to_load == 0)
            throw PolarisException("Do not call build with 0 points", -1, __PRETTY_FUNCTION__, __FILE__, __LINE__);

        if (!collie::filesystem::exists(filename)) {
            std::stringstream stream;
            stream << "ERROR: Data file " << filename << " does not exist." << std::endl;
            polaris::cerr << stream.str() << std::endl;
            throw polaris::PolarisException(stream.str(), -1, __PRETTY_FUNCTION__, __FILE__, __LINE__);
        }

        size_t file_num_points, file_dim;
        if (filename == nullptr) {
            throw polaris::PolarisException("Can not build with an empty file", -1, __PRETTY_FUNCTION__, __FILE__,
                                            __LINE__);
        }

        polaris::get_bin_metadata(filename, file_num_points, file_dim);
        if (file_num_points > _max_points) {
            std::stringstream stream;
            stream << "ERROR: Driver requests loading " << num_points_to_load << " points and file has "
                   << file_num_points
                   << " points, but "
                   << "index can support only " << _max_points << " points as specified in constructor." << std::endl;

            throw polaris::PolarisException(stream.str(), -1, __PRETTY_FUNCTION__, __FILE__, __LINE__);
        }

        if (num_points_to_load > file_num_points) {
            std::stringstream stream;
            stream << "ERROR: Driver requests loading " << num_points_to_load << " points and file has only "
                   << file_num_points << " points." << std::endl;

            throw polaris::PolarisException(stream.str(), -1, __PRETTY_FUNCTION__, __FILE__, __LINE__);
        }

        if (file_dim != _index_config.basic_config.dimension) {
            std::stringstream stream;
            stream << "ERROR: Driver requests loading " << _index_config.basic_config.dimension << " dimension,"
                   << "but file has " << file_dim << " dimension." << std::endl;
            polaris::cerr << stream.str() << std::endl;

            throw polaris::PolarisException(stream.str(), -1, __PRETTY_FUNCTION__, __FILE__, __LINE__);
        }

        // REFACTOR PQ TODO: We can remove this if and add a check in the InMemDataStore
        // to not populate_data if it has been called once.
        if (_pq_dist) {
            // REFACTOR TODO: Both in the previous code and in the current PQDataStore,
            // we are writing the PQ files in the same path as the input file. Now we
            // may not have write permissions to that folder, but we will always have
            // write permissions to the output folder. So we should write the PQ files
            // there. The problem is that the VamanaIndex class gets the output folder prefix
            // only at the time of save(), by which time we are too late. So leaving it
            // as-is for now.
            _pq_data_store->populate_data(filename, 0U);
        }

        _data_store->populate_data(filename, 0U);
        polaris::cout << "Using only first " << num_points_to_load << " from file.. " << std::endl;

        {
            std::unique_lock<std::shared_timed_mutex> tl(_tag_lock);
            _nd = num_points_to_load;
        }
        build_with_data_populated(tags);
    }

    template<typename T>
    void
    VamanaIndex<T>::build(const char *filename, const size_t num_points_to_load, const char *tag_filename) {
        std::vector<vid_t> tags;

        if (_enable_tags) {
            std::unique_lock<std::shared_timed_mutex> tl(_tag_lock);
            if (tag_filename == nullptr) {
                throw PolarisException("Tag filename is null, while _enable_tags is set", -1, __PRETTY_FUNCTION__,
                                       __FILE__, __LINE__);
            } else {
                if (collie::filesystem::exists(tag_filename)) {
                    polaris::cout << "Loading tags from " << tag_filename << " for vamana index build" << std::endl;
                    vid_t *tag_data = nullptr;
                    size_t npts, ndim;
                    polaris::load_bin(tag_filename, tag_data, npts, ndim);
                    if (npts < num_points_to_load) {
                        std::stringstream sstream;
                        sstream << "Loaded " << npts << " tags, insufficient to populate tags for "
                                << num_points_to_load
                                << "  points to load";
                        throw polaris::PolarisException(sstream.str(), -1, __PRETTY_FUNCTION__, __FILE__, __LINE__);
                    }
                    for (size_t i = 0; i < num_points_to_load; i++) {
                        tags.push_back(tag_data[i]);
                    }
                    delete[] tag_data;
                } else {
                    throw polaris::PolarisException(std::string("Tag file") + tag_filename + " does not exist", -1,
                                                    __PRETTY_FUNCTION__,
                                                    __FILE__, __LINE__);
                }
            }
        }
        build(filename, num_points_to_load, tags);
    }

    template<typename T>
    void VamanaIndex<T>::build(const std::string &data_file, const size_t num_points_to_load) {
        size_t points_to_load = num_points_to_load == 0 ? _max_points : num_points_to_load;

        auto s = std::chrono::high_resolution_clock::now();
        this->build(data_file.c_str(), points_to_load);
        std::chrono::duration<double> diff = std::chrono::high_resolution_clock::now() - s;
        std::cout << "Indexing time: " << diff.count() << "\n";
    }

    template<typename T>
    std::pair<uint32_t, uint32_t>
    VamanaIndex<T>::_search(const DataType &query, const size_t K, const uint32_t L,
                            localid_t *indices, float *distances) {
        try {
            auto typed_query = std::any_cast<const T *>(query);
            return this->search(typed_query, K, L, indices, distances);
        }
        catch (const std::bad_any_cast &e) {
            throw PolarisException("Error: bad any cast while searching. " + std::string(e.what()), -1);
        }
        catch (const std::exception &e) {
            throw PolarisException("Error: " + std::string(e.what()), -1);
        }
    }

    template<typename T>
    turbo::Status VamanaIndex<T>::search(SearchContext &ctx) {
        if (ctx.top_k > ctx.search_list) {
            return turbo::make_status(turbo::kInvalidArgument, "Top K cannot be greater than search list ({}:{})",
                                      ctx.top_k, ctx.search_list);
        }

        ScratchStoreManager<InMemQueryScratch<T>> manager(_query_scratch);
        auto scratch = manager.scratch_space();

        if (ctx.search_list > scratch->get_L()) {
            POLARIS_LOG(INFO)
            << "Attempting to expand query scratch_space. Was created " << "with Lsize: " << scratch->get_L()
            << " but search L is: " << ctx.search_list;
            scratch->resize_for_new_L(ctx.search_list);
            POLARIS_LOG(INFO) << "Resized scratch space to " << scratch->get_L();
        }

        std::shared_lock<std::shared_timed_mutex> ul(_update_lock);

        const std::vector<uint32_t> init_ids = get_init_ids();

        _data_store->preprocess_query(reinterpret_cast<T *>(ctx.query.data()), scratch);
        auto rs = iterate_to_fixed_point(scratch, ctx.search_list, init_ids, ctx.search_condition, true);
        if (!rs.ok()) {
            return rs.status();
        }

        NeighborPriorityQueue &best_L_nodes = scratch->best_l_nodes();
        assert(best_L_nodes.size() <= ctx.search_list);

        std::shared_lock<std::shared_timed_mutex> tl(_tag_lock);

        size_t pos = 0;
        ctx.top_k_queue.reserve(ctx.top_k);
        if (ctx.with_local_ids) {
            ctx.local_ids.reserve(ctx.top_k);
        }
        for (size_t i = 0; i < best_L_nodes.size(); ++i) {
            auto node = best_L_nodes[i];

            vid_t tag;
            if (_location_to_tag.try_get(node.id, tag)) {
                auto dis = _index_config.basic_config.metric == MetricType::METRIC_INNER_PRODUCT ? -1 * node.distance
                                                                                                 : node.distance;
                ctx.top_k_queue.emplace_back(tag, dis);
                if (ctx.with_local_ids) {
                    ctx.local_ids.emplace_back(node.id);
                }
                if (ctx.with_raw_vectors) {
                    std::vector<uint8_t> raw_vector;
                    raw_vector.resize(_data_store->get_aligned_dim() * sizeof(T));
                    _data_store->get_vector(node.id, reinterpret_cast<T *>(raw_vector.data()));
                    ctx.raw_vectors.push_back(std::move(raw_vector));
                }

                pos++;
                // If res_vectors.size() < k, clip at the value.
                if (pos == ctx.top_k) {
                    break;
                }
            }
        }

        //return pos;
        ctx.hops = rs.value().first;
        ctx.cmps = rs.value().second;
        return turbo::ok_status();
    }

    template<typename T>
    std::pair<uint32_t, uint32_t> VamanaIndex<T>::search(const T *query, const size_t K, const uint32_t L,
                                                         localid_t *indices, float *distances) {
        if (K > (uint64_t) L) {
            throw PolarisException("Set L to a value of at least K", -1, __PRETTY_FUNCTION__, __FILE__, __LINE__);
        }

        ScratchStoreManager<InMemQueryScratch<T>> manager(_query_scratch);
        auto scratch = manager.scratch_space();

        if (L > scratch->get_L()) {
            polaris::cout << "Attempting to expand query scratch_space. Was created "
                          << "with Lsize: " << scratch->get_L() << " but search L is: " << L << std::endl;
            scratch->resize_for_new_L(L);
            polaris::cout << "Resize completed. New scratch->L is " << scratch->get_L() << std::endl;
        }

        const std::vector<uint32_t> init_ids = get_init_ids();

        std::shared_lock<std::shared_timed_mutex> lock(_update_lock);

        _data_store->preprocess_query(query, scratch);

        auto retval = iterate_to_fixed_point(scratch, L, init_ids, true);

        NeighborPriorityQueue &best_L_nodes = scratch->best_l_nodes();

        size_t pos = 0;
        for (size_t i = 0; i < best_L_nodes.size(); ++i) {
            if (best_L_nodes[i].id < _max_points) {
                // safe because VamanaIndex uses uint32_t ids internally
                // and IDType will be uint32_t or uint64_t
                indices[pos] = (localid_t) best_L_nodes[i].id;
                if (distances != nullptr) {
                    distances[pos] =
                            _index_config.basic_config.metric == polaris::MetricType::METRIC_INNER_PRODUCT ? -1 *
                                                                                                             best_L_nodes[i].distance
                                                                                                           : best_L_nodes[i].distance;
                }
                pos++;
            }
            if (pos == K)
                break;
        }
        if (pos < K) {
            polaris::cerr << "Found pos: " << pos << "fewer than K elements " << K << " for query" << std::endl;
        }

        return retval;
    }

    template<typename T>
    size_t VamanaIndex<T>::_search_with_tags(const DataType &query, const uint64_t K, const uint32_t L,
                                             const TagType &tags, float *distances, DataVector &res_vectors) {
        try {
            return this->search_with_tags(std::any_cast<const T *>(query), K, L, std::any_cast<vid_t *>(tags),
                                          distances, res_vectors.get<std::vector<T *>>());
        }
        catch (const std::bad_any_cast &e) {
            throw PolarisException("Error: bad any cast while performing _search_with_tags() " + std::string(e.what()),
                                   -1);
        }
        catch (const std::exception &e) {
            throw PolarisException("Error: " + std::string(e.what()), -1);
        }
    }

    template<typename T>
    size_t VamanaIndex<T>::search_with_tags(const T *query, const uint64_t K, const uint32_t L, vid_t *tags,
                                            float *distances, std::vector<T *> &res_vectors) {
        if (K > (uint64_t) L) {
            throw PolarisException("Set L to a value of at least K", -1, __PRETTY_FUNCTION__, __FILE__, __LINE__);
        }
        ScratchStoreManager<InMemQueryScratch<T>> manager(_query_scratch);
        auto scratch = manager.scratch_space();

        if (L > scratch->get_L()) {
            polaris::cout << "Attempting to expand query scratch_space. Was created "
                          << "with Lsize: " << scratch->get_L() << " but search L is: " << L << std::endl;
            scratch->resize_for_new_L(L);
            polaris::cout << "Resize completed. New scratch->L is " << scratch->get_L() << std::endl;
        }

        std::shared_lock<std::shared_timed_mutex> ul(_update_lock);

        const std::vector<uint32_t> init_ids = get_init_ids();

        //_distance->preprocess_query(query, _data_store->get_dims(),
        // scratch->aligned_query());
        _data_store->preprocess_query(query, scratch);
        iterate_to_fixed_point(scratch, L, init_ids, true);

        NeighborPriorityQueue &best_L_nodes = scratch->best_l_nodes();
        assert(best_L_nodes.size() <= L);

        std::shared_lock<std::shared_timed_mutex> tl(_tag_lock);

        size_t pos = 0;
        for (size_t i = 0; i < best_L_nodes.size(); ++i) {
            auto node = best_L_nodes[i];

            vid_t tag;
            if (_location_to_tag.try_get(node.id, tag)) {
                tags[pos] = tag;

                if (res_vectors.size() > 0) {
                    _data_store->get_vector(node.id, res_vectors[pos]);
                }

                if (distances != nullptr) {
                    distances[pos] =
                            _index_config.basic_config.metric == MetricType::METRIC_INNER_PRODUCT ? -1 * node.distance
                                                                                                  : node.distance;
                }
                pos++;
                // If res_vectors.size() < k, clip at the value.
                if (pos == K || pos == res_vectors.size())
                    break;
            }
        }

        return pos;
    }

    template<typename T>
    size_t VamanaIndex<T>::get_num_points() {
        std::shared_lock<std::shared_timed_mutex> tl(_tag_lock);
        return _nd;
    }

    template<typename T>
    size_t VamanaIndex<T>::get_max_points() {
        std::shared_lock<std::shared_timed_mutex> tl(_tag_lock);
        return _max_points;
    }

    template<typename T>
    void VamanaIndex<T>::generate_frozen_point() {
        if (_num_frozen_pts == 0)
            return;

        if (_num_frozen_pts > 1) {
            throw PolarisException("More than one frozen point not supported in generate_frozen_point", -1,
                                   __PRETTY_FUNCTION__,
                                   __FILE__, __LINE__);
        }

        if (_nd == 0) {
            throw PolarisException("ERROR: Can not pick a frozen point since nd=0", -1, __PRETTY_FUNCTION__, __FILE__,
                                   __LINE__);
        }
        size_t res = calculate_entry_point();

        // REFACTOR PQ: Not sure if we should do this for both stores.
        if (_pq_dist) {
            // copy the PQ data corresponding to the point returned by
            // calculate_entry_point
            // memcpy(_pq_data + _max_points * _num_pq_chunks,
            //       _pq_data + res * _num_pq_chunks,
            //       _num_pq_chunks * DIV_ROUND_UP(NUM_PQ_BITS, 8));
            _pq_data_store->copy_vectors((location_t) res, (location_t) _max_points, 1);
        } else {
            _data_store->copy_vectors((location_t) res, (location_t) _max_points, 1);
        }
        _frozen_pts_used++;
    }

    template<typename T>
    int VamanaIndex<T>::enable_delete() {
        assert(_enable_tags);

        if (!_enable_tags) {
            polaris::cerr << "Tags must be instantiated for deletions" << std::endl;
            return -2;
        }

        if (this->_deletes_enabled) {
            return 0;
        }

        std::unique_lock<std::shared_timed_mutex> ul(_update_lock);
        std::unique_lock<std::shared_timed_mutex> tl(_tag_lock);
        std::unique_lock<std::shared_timed_mutex> dl(_delete_lock);

        if (_data_compacted) {
            for (uint32_t slot = (uint32_t) _nd; slot < _max_points; ++slot) {
                _empty_slots.insert(slot);
            }
        }
        this->_deletes_enabled = true;
        return 0;
    }

    template<typename T>
    inline void VamanaIndex<T>::process_delete(const turbo::flat_hash_set<uint32_t> &old_delete_set, size_t loc,
                                               const uint32_t range, const uint32_t maxc, const float alpha,
                                               InMemQueryScratch<T> *scratch) {
        turbo::flat_hash_set<uint32_t> &expanded_nodes_set = scratch->expanded_nodes_set();
        std::vector<Neighbor> &expanded_nghrs_vec = scratch->expanded_nodes_vec();

        // If this condition were not true, deadlock could result
        assert(old_delete_set.find((uint32_t) loc) == old_delete_set.end());

        std::vector<uint32_t> adj_list;
        {
            // Acquire and release lock[loc] before acquiring locks for neighbors
            std::unique_lock<non_recursive_mutex> adj_list_lock;
            if (_conc_consolidate)
                adj_list_lock = std::unique_lock<non_recursive_mutex>(_locks[loc]);
            adj_list = _graph_store->get_neighbours((location_t) loc);
        }

        bool modify = false;
        for (auto ngh: adj_list) {
            if (old_delete_set.find(ngh) == old_delete_set.end()) {
                expanded_nodes_set.insert(ngh);
            } else {
                modify = true;

                std::unique_lock<non_recursive_mutex> ngh_lock;
                if (_conc_consolidate)
                    ngh_lock = std::unique_lock<non_recursive_mutex>(_locks[ngh]);
                for (auto j: _graph_store->get_neighbours((location_t) ngh))
                    if (j != loc && old_delete_set.find(j) == old_delete_set.end())
                        expanded_nodes_set.insert(j);
            }
        }

        if (modify) {
            if (expanded_nodes_set.size() <= range) {
                std::unique_lock<non_recursive_mutex> adj_list_lock(_locks[loc]);
                _graph_store->clear_neighbours((location_t) loc);
                for (auto &ngh: expanded_nodes_set)
                    _graph_store->add_neighbour((location_t) loc, ngh);
            } else {
                // Create a pool of Neighbor candidates from the expanded_nodes_set
                expanded_nghrs_vec.reserve(expanded_nodes_set.size());
                for (auto &ngh: expanded_nodes_set) {
                    expanded_nghrs_vec.emplace_back(ngh, _data_store->get_distance((location_t) loc, (location_t) ngh));
                }
                std::sort(expanded_nghrs_vec.begin(), expanded_nghrs_vec.end());
                std::vector<uint32_t> &occlude_list_output = scratch->occlude_list_output();
                occlude_list((uint32_t) loc, expanded_nghrs_vec, alpha, range, maxc, occlude_list_output, scratch,
                             &old_delete_set);
                std::unique_lock<non_recursive_mutex> adj_list_lock(_locks[loc]);
                _graph_store->set_neighbours((location_t) loc, occlude_list_output);
            }
        }
    }

// Returns number of live points left after consolidation
    template<typename T>
    consolidation_report VamanaIndex<T>::consolidate_deletes(const IndexWriteParameters &params) {
        if (!_enable_tags)
            throw polaris::PolarisException("Point tag array not instantiated", -1, __PRETTY_FUNCTION__, __FILE__,
                                            __LINE__);

        {
            std::shared_lock<std::shared_timed_mutex> ul(_update_lock);
            std::shared_lock<std::shared_timed_mutex> tl(_tag_lock);
            std::shared_lock<std::shared_timed_mutex> dl(_delete_lock);
            if (_empty_slots.size() + _nd != _max_points) {
                std::string err = "#empty slots + nd != max points";
                polaris::cerr << err << std::endl;
                throw PolarisException(err, -1, __PRETTY_FUNCTION__, __FILE__, __LINE__);
            }

            if (_location_to_tag.size() + _delete_set->size() != _nd) {
                polaris::cerr << "Error: _location_to_tag.size (" << _location_to_tag.size()
                              << ")  + _delete_set->size ("
                              << _delete_set->size() << ") != _nd(" << _nd << ") ";
                return consolidation_report(polaris::consolidation_report::status_code::INCONSISTENT_COUNT_ERROR, 0, 0,
                                            0,
                                            0, 0, 0, 0);
            }

            if (_location_to_tag.size() != _tag_to_location.size()) {
                throw polaris::PolarisException("_location_to_tag and _tag_to_location not of same size", -1,
                                                __PRETTY_FUNCTION__,
                                                __FILE__, __LINE__);
            }
        }

        std::unique_lock<std::shared_timed_mutex> update_lock(_update_lock, std::defer_lock);
        if (!_conc_consolidate)
            update_lock.lock();

        std::unique_lock<std::shared_timed_mutex> cl(_consolidate_lock, std::defer_lock);
        if (!cl.try_lock()) {
            polaris::cerr << "Consildate delete function failed to acquire consolidate lock" << std::endl;
            return consolidation_report(polaris::consolidation_report::status_code::LOCK_FAIL, 0, 0, 0, 0, 0, 0, 0);
        }

        polaris::cout << "Starting consolidate_deletes... ";

        std::unique_ptr<turbo::flat_hash_set<uint32_t>> old_delete_set(new turbo::flat_hash_set<uint32_t>);
        {
            std::unique_lock<std::shared_timed_mutex> dl(_delete_lock);
            std::swap(_delete_set, old_delete_set);
        }

        if (old_delete_set->find(_start) != old_delete_set->end()) {
            throw polaris::PolarisException("ERROR: start node has been deleted", -1, __PRETTY_FUNCTION__, __FILE__,
                                            __LINE__);
        }

        const uint32_t range = params.max_degree;
        const uint32_t maxc = params.max_occlusion_size;
        const float alpha = params.alpha;
        const uint32_t num_threads = params.num_threads == 0 ? omp_get_num_procs() : params.num_threads;

        uint32_t num_calls_to_process_delete = 0;
        polaris::Timer timer;
#pragma omp parallel for num_threads(num_threads) schedule(dynamic, 8192) reduction(+ : num_calls_to_process_delete)
        for (int64_t loc = 0; loc < (int64_t) _max_points; loc++) {
            if (old_delete_set->find((uint32_t) loc) == old_delete_set->end() &&
                !_empty_slots.is_in_set((uint32_t) loc)) {
                ScratchStoreManager<InMemQueryScratch<T>> manager(_query_scratch);
                auto scratch = manager.scratch_space();
                process_delete(*old_delete_set, loc, range, maxc, alpha, scratch);
                num_calls_to_process_delete += 1;
            }
        }
        for (int64_t loc = _max_points; loc < (int64_t) (_max_points + _num_frozen_pts); loc++) {
            ScratchStoreManager<InMemQueryScratch<T>> manager(_query_scratch);
            auto scratch = manager.scratch_space();
            process_delete(*old_delete_set, loc, range, maxc, alpha, scratch);
            num_calls_to_process_delete += 1;
        }

        std::unique_lock<std::shared_timed_mutex> tl(_tag_lock);
        size_t ret_nd = release_locations(*old_delete_set);
        size_t max_points = _max_points;
        size_t empty_slots_size = _empty_slots.size();

        std::shared_lock<std::shared_timed_mutex> dl(_delete_lock);
        size_t delete_set_size = _delete_set->size();
        size_t old_delete_set_size = old_delete_set->size();

        if (!_conc_consolidate) {
            update_lock.unlock();
        }

        double duration = timer.elapsed() / 1000000.0;
        polaris::cout << " done in " << duration << " seconds." << std::endl;
        return consolidation_report(polaris::consolidation_report::status_code::SUCCESS, ret_nd, max_points,
                                    empty_slots_size, old_delete_set_size, delete_set_size, num_calls_to_process_delete,
                                    duration);
    }

    template<typename T>
    void VamanaIndex<T>::compact_frozen_point() {
        if (_nd < _max_points && _num_frozen_pts > 0) {
            reposition_points((uint32_t) _max_points, (uint32_t) _nd, (uint32_t) _num_frozen_pts);
            _start = (uint32_t) _nd;
        }
    }

// Should be called after acquiring _update_lock
    template<typename T>
    void VamanaIndex<T>::compact_data() {
        if (!_dynamic_index)
            throw PolarisException("Can not compact a non-dynamic index", -1, __PRETTY_FUNCTION__, __FILE__, __LINE__);

        if (_data_compacted) {
            polaris::cerr << "Warning! Calling compact_data() when _data_compacted is true!" << std::endl;
            return;
        }

        if (_delete_set->size() > 0) {
            throw PolarisException("Can not compact data when index has non-empty _delete_set of "
                                   "size: " +
                                   std::to_string(_delete_set->size()),
                                   -1, __PRETTY_FUNCTION__, __FILE__, __LINE__);
        }

        polaris::Timer timer;

        std::vector<uint32_t> new_location = std::vector<uint32_t>(_max_points + _num_frozen_pts, UINT32_MAX);

        uint32_t new_counter = 0;
        std::set<uint32_t> empty_locations;
        for (uint32_t old_location = 0; old_location < _max_points; old_location++) {
            if (_location_to_tag.contains(old_location)) {
                new_location[old_location] = new_counter;
                new_counter++;
            } else {
                empty_locations.insert(old_location);
            }
        }
        for (uint32_t old_location = (uint32_t) _max_points;
             old_location < _max_points + _num_frozen_pts; old_location++) {
            new_location[old_location] = old_location;
        }

        // If start node is removed, throw an exception
        if (_start < _max_points && !_location_to_tag.contains(_start)) {
            throw polaris::PolarisException("ERROR: Start node deleted.", -1, __PRETTY_FUNCTION__, __FILE__, __LINE__);
        }

        size_t num_dangling = 0;
        for (uint32_t old = 0; old < _max_points + _num_frozen_pts; ++old) {
            // compact _final_graph
            std::vector<uint32_t> new_adj_list;

            if ((new_location[old] < _max_points) // If point continues to exist
                || (old >= _max_points && old < _max_points + _num_frozen_pts)) {
                new_adj_list.reserve(_graph_store->get_neighbours((location_t) old).size());
                for (auto ngh_iter: _graph_store->get_neighbours((location_t) old)) {
                    if (empty_locations.find(ngh_iter) != empty_locations.end()) {
                        ++num_dangling;
                        polaris::cerr << "Error in compact_data(). _final_graph[" << old << "] has neighbor "
                                      << ngh_iter
                                      << " which is a location not associated with any tag." << std::endl;
                    } else {
                        new_adj_list.push_back(new_location[ngh_iter]);
                    }
                }
                //_graph_store->get_neighbours((location_t)old).swap(new_adj_list);
                _graph_store->set_neighbours((location_t) old, new_adj_list);

                // Move the data and adj list to the correct position
                if (new_location[old] != old) {
                    assert(new_location[old] < old);
                    _graph_store->swap_neighbours(new_location[old], (location_t) old);
                    _data_store->copy_vectors(old, new_location[old], 1);
                }
            } else {
                _graph_store->clear_neighbours((location_t) old);
            }
        }
        polaris::cerr << "#dangling references after data compaction: " << num_dangling << std::endl;

        _tag_to_location.clear();
        for (auto pos = _location_to_tag.find_first(); pos.is_valid(); pos = _location_to_tag.find_next(pos)) {
            const auto tag = _location_to_tag.get(pos);
            _tag_to_location[tag] = new_location[pos._key];
        }
        _location_to_tag.clear();
        for (const auto &iter: _tag_to_location) {
            _location_to_tag.set(iter.second, iter.first);
        }
        // remove all cleared up old
        for (size_t old = _nd; old < _max_points; ++old) {
            _graph_store->clear_neighbours((location_t) old);
        }

        _empty_slots.clear();
        // mark all slots after _nd as empty
        for (auto i = _nd; i < _max_points; i++) {
            _empty_slots.insert((uint32_t) i);
        }
        _data_compacted = true;
        polaris::cout << "Time taken for compact_data: " << timer.elapsed() / 1000000. << "s." << std::endl;
    }

    //
    // Caller must hold unique _tag_lock and _delete_lock before calling this
    //
    template<typename T>
    int VamanaIndex<T>::reserve_location() {
        if (_nd >= _max_points) {
            return -1;
        }
        uint32_t location;
        if (_data_compacted && _empty_slots.is_empty()) {
            // This code path is encountered when enable_delete hasn't been
            // called yet, so no points have been deleted and _empty_slots
            // hasn't been filled in. In that case, just keep assigning
            // consecutive locations.
            location = (uint32_t) _nd;
        } else {
            assert(_empty_slots.size() != 0);
            assert(_empty_slots.size() + _nd == _max_points);

            location = _empty_slots.pop_any();
            _delete_set->erase(location);
        }
        ++_nd;
        return location;
    }

    template<typename T>
    size_t VamanaIndex<T>::release_location(int location) {
        if (_empty_slots.is_in_set(location))
            throw PolarisException("Trying to release location, but location already in empty slots", -1,
                                   __PRETTY_FUNCTION__, __FILE__,
                                   __LINE__);
        _empty_slots.insert(location);

        _nd--;
        return _nd;
    }

    template<typename T>
    size_t VamanaIndex<T>::release_locations(const turbo::flat_hash_set<uint32_t> &locations) {
        for (auto location: locations) {
            if (_empty_slots.is_in_set(location))
                throw PolarisException("Trying to release location, but location "
                                       "already in empty slots",
                                       -1, __PRETTY_FUNCTION__, __FILE__, __LINE__);
            _empty_slots.insert(location);

            _nd--;
        }

        if (_empty_slots.size() + _nd != _max_points)
            throw PolarisException("#empty slots + nd != max points", -1, __PRETTY_FUNCTION__, __FILE__, __LINE__);

        return _nd;
    }

    template<typename T>
    void VamanaIndex<T>::reposition_points(uint32_t old_location_start, uint32_t new_location_start,
                                           uint32_t num_locations) {
        if (num_locations == 0 || old_location_start == new_location_start) {
            return;
        }

        // Update pointers to the moved nodes. Note: the computation is correct even
        // when new_location_start < old_location_start given the C++ uint32_t
        // integer arithmetic rules.
        const uint32_t location_delta = new_location_start - old_location_start;

        std::vector<location_t> updated_neighbours_location;
        for (uint32_t i = 0; i < _max_points + _num_frozen_pts; i++) {
            auto &i_neighbours = _graph_store->get_neighbours((location_t) i);
            std::vector<location_t> i_neighbours_copy(i_neighbours.begin(), i_neighbours.end());
            for (auto &loc: i_neighbours_copy) {
                if (loc >= old_location_start && loc < old_location_start + num_locations)
                    loc += location_delta;
            }
            _graph_store->set_neighbours(i, i_neighbours_copy);
        }

        // The [start, end) interval which will contain obsolete points to be
        // cleared.
        uint32_t mem_clear_loc_start = old_location_start;
        uint32_t mem_clear_loc_end_limit = old_location_start + num_locations;

        // Move the adjacency lists. Make sure that overlapping ranges are handled
        // correctly.
        if (new_location_start < old_location_start) {
            // New location before the old location: copy the entries in order
            // to avoid modifying locations that are yet to be copied.
            for (uint32_t loc_offset = 0; loc_offset < num_locations; loc_offset++) {
                assert(_graph_store->get_neighbours(new_location_start + loc_offset).empty());
                _graph_store->swap_neighbours(new_location_start + loc_offset, old_location_start + loc_offset);
            }
            // If ranges are overlapping, make sure not to clear the newly copied
            // data.
            if (mem_clear_loc_start < new_location_start + num_locations) {
                // Clear only after the end of the new range.
                mem_clear_loc_start = new_location_start + num_locations;
            }
        } else {
            // Old location after the new location: copy from the end of the range
            // to avoid modifying locations that are yet to be copied.
            for (uint32_t loc_offset = num_locations; loc_offset > 0; loc_offset--) {
                assert(_graph_store->get_neighbours(new_location_start + loc_offset - 1u).empty());
                _graph_store->swap_neighbours(new_location_start + loc_offset - 1u,
                                              old_location_start + loc_offset - 1u);
            }

            // If ranges are overlapping, make sure not to clear the newly copied
            // data.
            if (mem_clear_loc_end_limit > new_location_start) {
                // Clear only up to the beginning of the new range.
                mem_clear_loc_end_limit = new_location_start;
            }
        }
        _data_store->move_vectors(old_location_start, new_location_start, num_locations);
    }

    template<typename T>
    void VamanaIndex<T>::reposition_frozen_point_to_end() {
        if (_num_frozen_pts == 0)
            return;

        if (_nd == _max_points) {
            polaris::cout << "Not repositioning frozen point as it is already at the end." << std::endl;
            return;
        }

        reposition_points((uint32_t) _nd, (uint32_t) _max_points, (uint32_t) _num_frozen_pts);
        _start = (uint32_t) _max_points;
    }

    template<typename T>
    void VamanaIndex<T>::resize(size_t new_max_points) {
        const size_t new_internal_points = new_max_points + _num_frozen_pts;
        auto start = std::chrono::high_resolution_clock::now();
        assert(_empty_slots.size() == 0); // should not resize if there are empty slots.

        _data_store->resize((location_t) new_internal_points);
        _graph_store->resize_graph(new_internal_points);
        _locks = std::vector<non_recursive_mutex>(new_internal_points);

        if (_num_frozen_pts != 0) {
            reposition_points((uint32_t) _max_points, (uint32_t) new_max_points, (uint32_t) _num_frozen_pts);
            _start = (uint32_t) new_max_points;
        }

        _max_points = new_max_points;
        _empty_slots.reserve(_max_points);
        for (auto i = _nd; i < _max_points; i++) {
            _empty_slots.insert((uint32_t) i);
        }

        auto stop = std::chrono::high_resolution_clock::now();
        polaris::cout << "Resizing took: " << std::chrono::duration<double>(stop - start).count() << "s" << std::endl;
    }

    template<typename T>
    int VamanaIndex<T>::_insert_point(const DataType &point, const TagType tag) {
        try {
            return this->insert_point(std::any_cast<const T *>(point), std::any_cast<const vid_t>(tag));
        }
        catch (const std::bad_any_cast &anycast_e) {
            throw new PolarisException("Error:Trying to insert invalid data type" + std::string(anycast_e.what()), -1);
        }
        catch (const std::exception &e) {
            throw new PolarisException("Error:" + std::string(e.what()), -1);
        }
    }

    template<typename T>
    int VamanaIndex<T>::insert_point(const T *point, const vid_t tag) {

        assert(_has_built);
        if (tag == 0) {
            throw polaris::PolarisException("Do not insert point with tag 0. That is "
                                            "reserved for points hidden "
                                            "from the user.",
                                            -1, __PRETTY_FUNCTION__, __FILE__, __LINE__);
        }

        std::shared_lock<std::shared_timed_mutex> shared_ul(_update_lock);
        std::unique_lock<std::shared_timed_mutex> tl(_tag_lock);
        std::unique_lock<std::shared_timed_mutex> dl(_delete_lock);

        auto location = reserve_location();
        if (location == -1) {
#if EXPAND_IF_FULL
                                                                                                                                    dl.unlock();
        tl.unlock();
        shared_ul.unlock();

        {
            std::unique_lock<std::shared_timed_mutex> ul(_update_lock);
            tl.lock();
            dl.lock();

            if (_nd >= _max_points)
            {
                auto new_max_points = (size_t)(_max_points * INDEX_GROWTH_FACTOR);
                resize(new_max_points);
            }

            dl.unlock();
            tl.unlock();
            ul.unlock();
        }

        shared_ul.lock();
        tl.lock();
        dl.lock();

        location = reserve_location();
        if (location == -1)
        {
            throw polaris::PolarisException("Cannot reserve location even after "
                                        "expanding graph. Terminating.",
                                        -1, __PRETTY_FUNCTION__, __FILE__, __LINE__);
        }
#else
            return -1;
#endif
        } // cant insert as active pts >= max_pts
        dl.unlock();

        // Insert tag and mapping to location
        if (_enable_tags) {
            // if tags are enabled and tag is already inserted. so we can't reuse that tag.
            if (_tag_to_location.find(tag) != _tag_to_location.end()) {
                release_location(location);
                return -1;
            }

            _tag_to_location[tag] = location;
            _location_to_tag.set(location, tag);
        }
        tl.unlock();

        _data_store->set_vector(location, point); // update datastore

        // Find and add appropriate graph edges
        ScratchStoreManager<InMemQueryScratch<T>> manager(_query_scratch);
        auto scratch = manager.scratch_space();
        std::vector<uint32_t> pruned_list; // it is the set best candidates to connect to this point
        search_for_point_and_prune(location, _indexingQueueSize, pruned_list, scratch);
        assert(pruned_list.size() > 0); // should find atleast one neighbour (i.e frozen point acting as medoid)

        {
            std::shared_lock<std::shared_timed_mutex> tlock(_tag_lock, std::defer_lock);
            if (_conc_consolidate)
                tlock.lock();

            LockGuard guard(_locks[location]);
            _graph_store->clear_neighbours(location);

            std::vector<uint32_t> neighbor_links;
            for (auto link: pruned_list) {
                if (_conc_consolidate)
                    if (!_location_to_tag.contains(link))
                        continue;
                neighbor_links.emplace_back(link);
            }
            _graph_store->set_neighbours(location, neighbor_links);
            assert(_graph_store->get_neighbours(location).size() <= _indexingRange);

            if (_conc_consolidate)
                tlock.unlock();
        }

        inter_insert(location, pruned_list, scratch);

        return 0;
    }

    template<typename T>
    int VamanaIndex<T>::_lazy_delete(const TagType &tag) {
        try {
            return lazy_delete(std::any_cast<const vid_t>(tag));
        }
        catch (const std::bad_any_cast &e) {
            throw PolarisException(std::string("Error: ") + e.what(), -1);
        }
    }

    template<typename T>
    void VamanaIndex<T>::_lazy_delete(TagVector &tags, TagVector &failed_tags) {
        try {
            this->lazy_delete(tags.get<const std::vector<vid_t>>(), failed_tags.get<std::vector<vid_t>>());
        }
        catch (const std::bad_any_cast &e) {
            throw PolarisException("Error: bad any cast while performing _lazy_delete() " + std::string(e.what()), -1);
        }
        catch (const std::exception &e) {
            throw PolarisException("Error: " + std::string(e.what()), -1);
        }
    }

    template<typename T>
    int VamanaIndex<T>::lazy_delete(const vid_t &tag) {
        std::shared_lock<std::shared_timed_mutex> ul(_update_lock);
        std::unique_lock<std::shared_timed_mutex> tl(_tag_lock);
        std::unique_lock<std::shared_timed_mutex> dl(_delete_lock);
        _data_compacted = false;

        if (_tag_to_location.find(tag) == _tag_to_location.end()) {
            polaris::cerr << "Delete tag not found " << get_tag_string(tag) << std::endl;
            return -1;
        }
        assert(_tag_to_location[tag] < _max_points);

        const auto location = _tag_to_location[tag];
        _delete_set->insert(location);
        _location_to_tag.erase(location);
        _tag_to_location.erase(tag);
        return 0;
    }

    template<typename T>
    void VamanaIndex<T>::lazy_delete(const std::vector<vid_t> &tags, std::vector<vid_t> &failed_tags) {
        if (failed_tags.size() > 0) {
            throw PolarisException("failed_tags should be passed as an empty list", -1, __PRETTY_FUNCTION__, __FILE__,
                                   __LINE__);
        }
        std::shared_lock<std::shared_timed_mutex> ul(_update_lock);
        std::unique_lock<std::shared_timed_mutex> tl(_tag_lock);
        std::unique_lock<std::shared_timed_mutex> dl(_delete_lock);
        _data_compacted = false;

        for (auto tag: tags) {
            if (_tag_to_location.find(tag) == _tag_to_location.end()) {
                failed_tags.push_back(tag);
            } else {
                const auto location = _tag_to_location[tag];
                _delete_set->insert(location);
                _location_to_tag.erase(location);
                _tag_to_location.erase(tag);
            }
        }
    }

    template<typename T>
    bool VamanaIndex<T>::is_index_saved() {
        return _is_saved;
    }

    template<typename T>
    void VamanaIndex<T>::_get_active_tags(TagRobinSet &active_tags) {
        try {
            this->get_active_tags(active_tags.get<turbo::flat_hash_set<vid_t>>());
        }
        catch (const std::bad_any_cast &e) {
            throw PolarisException("Error: bad_any cast while performing _get_active_tags() " + std::string(e.what()),
                                   -1);
        }
        catch (const std::exception &e) {
            throw PolarisException("Error :" + std::string(e.what()), -1);
        }
    }

    template<typename T>
    void VamanaIndex<T>::get_active_tags(turbo::flat_hash_set<vid_t> &active_tags) {
        active_tags.clear();
        std::shared_lock<std::shared_timed_mutex> tl(_tag_lock);
        for (auto iter: _tag_to_location) {
            active_tags.insert(iter.first);
        }
    }

    template<typename T>
    void VamanaIndex<T>::print_status() {
        std::shared_lock<std::shared_timed_mutex> ul(_update_lock);
        std::shared_lock<std::shared_timed_mutex> cl(_consolidate_lock);
        std::shared_lock<std::shared_timed_mutex> tl(_tag_lock);
        std::shared_lock<std::shared_timed_mutex> dl(_delete_lock);

        polaris::cout << "------------------- VamanaIndex object: " << (uint64_t) this << " -------------------"
                      << std::endl;
        polaris::cout << "Number of points: " << _nd << std::endl;
        polaris::cout << "Graph size: " << _graph_store->get_total_points() << std::endl;
        polaris::cout << "Location to tag size: " << _location_to_tag.size() << std::endl;
        polaris::cout << "Tag to location size: " << _tag_to_location.size() << std::endl;
        polaris::cout << "Number of empty slots: " << _empty_slots.size() << std::endl;
        polaris::cout << std::boolalpha << "Data compacted: " << this->_data_compacted << std::endl;
        polaris::cout << "---------------------------------------------------------"
                         "------------"
                      << std::endl;
    }

    template<typename T>
    void VamanaIndex<T>::count_nodes_at_bfs_levels() {
        std::unique_lock<std::shared_timed_mutex> ul(_update_lock);

        collie::dynamic_bitset<> visited(_max_points + _num_frozen_pts);

        size_t MAX_BFS_LEVELS = 32;
        auto bfs_sets = new turbo::flat_hash_set<uint32_t>[MAX_BFS_LEVELS];

        bfs_sets[0].insert(_start);
        visited.set(_start);

        for (uint32_t i = (uint32_t) _max_points; i < _max_points + _num_frozen_pts; ++i) {
            if (i != _start) {
                bfs_sets[0].insert(i);
                visited.set(i);
            }
        }

        for (size_t l = 0; l < MAX_BFS_LEVELS - 1; ++l) {
            polaris::cout << "Number of nodes at BFS level " << l << " is " << bfs_sets[l].size() << std::endl;
            if (bfs_sets[l].size() == 0)
                break;
            for (auto node: bfs_sets[l]) {
                for (auto nghbr: _graph_store->get_neighbours((location_t) node)) {
                    if (!visited.test(nghbr)) {
                        visited.set(nghbr);
                        bfs_sets[l + 1].insert(nghbr);
                    }
                }
            }
        }

        delete[] bfs_sets;
    }

    // REFACTOR: This should be an OptimizedDataStore class
    template<typename T>
    void VamanaIndex<T>::optimize_index_layout() { // use after build or load
        if (_dynamic_index) {
            throw polaris::PolarisException("Optimize_index_layout not implemented for dyanmic indices", -1,
                                            __PRETTY_FUNCTION__,
                                            __FILE__, __LINE__);
        }

        float *cur_vec = new float[_data_store->get_aligned_dim()];
        std::memset(cur_vec, 0, _data_store->get_aligned_dim() * sizeof(float));
        _data_len = (_data_store->get_aligned_dim() + 1) * sizeof(float);
        _neighbor_len = (_graph_store->get_max_observed_degree() + 1) * sizeof(uint32_t);
        _node_size = _data_len + _neighbor_len;
        _opt_graph = new char[_node_size * _nd];
        auto dist_fast = (DistanceFastL2<T> *) (_data_store->get_dist_fn());
        for (uint32_t i = 0; i < _nd; i++) {
            char *cur_node_offset = _opt_graph + i * _node_size;
            _data_store->get_vector(i, (T *) cur_vec);
            float cur_norm = dist_fast->norm((T *) cur_vec, (uint32_t) _data_store->get_aligned_dim());
            std::memcpy(cur_node_offset, &cur_norm, sizeof(float));
            std::memcpy(cur_node_offset + sizeof(float), cur_vec, _data_len - sizeof(float));

            cur_node_offset += _data_len;
            uint32_t k = (uint32_t) _graph_store->get_neighbours(i).size();
            std::memcpy(cur_node_offset, &k, sizeof(uint32_t));
            std::memcpy(cur_node_offset + sizeof(uint32_t), _graph_store->get_neighbours(i).data(),
                        k * sizeof(uint32_t));
            // std::vector<uint32_t>().swap(_graph_store->get_neighbours(i));
            _graph_store->clear_neighbours(i);
        }
        _graph_store->clear_graph();
        _graph_store->resize_graph(0);
        delete[] cur_vec;
    }

    template<typename T>
    void VamanaIndex<T>::_search_with_optimized_layout(const DataType &query, size_t K, size_t L,
                                                       uint32_t *indices) {
        try {
            return this->search_with_optimized_layout(std::any_cast<const T *>(query), K, L, indices);
        }
        catch (const std::bad_any_cast &e) {
            throw PolarisException("Error: bad any cast while performing "
                                   "_search_with_optimized_layout() " +
                                   std::string(e.what()),
                                   -1);
        }
        catch (const std::exception &e) {
            throw PolarisException("Error: " + std::string(e.what()), -1);
        }
    }

    template<typename T>
    void VamanaIndex<T>::search_with_optimized_layout(const T *query, size_t K, size_t L, uint32_t *indices) {
        DistanceFastL2<T> *dist_fast = (DistanceFastL2<T> *) (_data_store->get_dist_fn());

        NeighborPriorityQueue retset(L);
        std::vector<uint32_t> init_ids(L);

        collie::dynamic_bitset<> flags{_nd, 0};
        uint32_t tmp_l = 0;
        uint32_t *neighbors = (uint32_t *) (_opt_graph + _node_size * _start + _data_len);
        uint32_t MaxM_ep = *neighbors;
        neighbors++;

        for (; tmp_l < L && tmp_l < MaxM_ep; tmp_l++) {
            init_ids[tmp_l] = neighbors[tmp_l];
            flags[init_ids[tmp_l]] = true;
        }

        while (tmp_l < L) {
            uint32_t id = rand() % _nd;
            if (flags[id])
                continue;
            flags[id] = true;
            init_ids[tmp_l] = id;
            tmp_l++;
        }

        for (uint32_t i = 0; i < init_ids.size(); i++) {
            uint32_t id = init_ids[i];
            if (id >= _nd)
                continue;
            _mm_prefetch(_opt_graph + _node_size * id, _MM_HINT_T0);
        }
        L = 0;
        for (uint32_t i = 0; i < init_ids.size(); i++) {
            uint32_t id = init_ids[i];
            if (id >= _nd)
                continue;
            T *x = (T *) (_opt_graph + _node_size * id);
            float norm_x = *x;
            x++;
            float dist = dist_fast->compare(x, query, norm_x, (uint32_t) _data_store->get_aligned_dim());
            retset.insert(Neighbor(id, dist));
            flags[id] = true;
            L++;
        }

        while (retset.has_unexpanded_node()) {
            auto nbr = retset.closest_unexpanded();
            auto n = nbr.id;
            _mm_prefetch(_opt_graph + _node_size * n + _data_len, _MM_HINT_T0);
            neighbors = (uint32_t *) (_opt_graph + _node_size * n + _data_len);
            uint32_t MaxM = *neighbors;
            neighbors++;
            for (uint32_t m = 0; m < MaxM; ++m)
                _mm_prefetch(_opt_graph + _node_size * neighbors[m], _MM_HINT_T0);
            for (uint32_t m = 0; m < MaxM; ++m) {
                uint32_t id = neighbors[m];
                if (flags[id])
                    continue;
                flags[id] = 1;
                T *data = (T *) (_opt_graph + _node_size * id);
                float norm = *data;
                data++;
                float dist = dist_fast->compare(query, data, norm, (uint32_t) _data_store->get_aligned_dim());
                Neighbor nn(id, dist);
                retset.insert(nn);
            }
        }

        for (size_t i = 0; i < K; i++) {
            indices[i] = retset[i].id;
        }
    }

    /*  Internals of the library */
    template<typename T> const float VamanaIndex<T>::INDEX_GROWTH_FACTOR = 1.5f;

    // EXPORTS
    template POLARIS_API
    class VamanaIndex<float>;

    template POLARIS_API
    class VamanaIndex<int8_t>;

    template POLARIS_API
    class VamanaIndex<uint8_t>;

} // namespace polaris
