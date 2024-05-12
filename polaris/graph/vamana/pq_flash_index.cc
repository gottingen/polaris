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

#include <polaris/utility/common_includes.h>
#include <polaris/core/log.h>
#include <polaris/graph/vamana/timer.h>
#include <polaris/graph/vamana/pq.h>
#include <polaris/datasets/bin.h>
#include <polaris/graph/vamana/disk_utils.h>
#include <polaris/graph/vamana/partition.h>
#include <polaris/graph/vamana/pq_scratch.h>
#include <polaris/graph/vamana/pq_flash_index.h>
#include <polaris/distance/cosine_similarity.h>
#include <polaris/io/linux_aligned_file_reader.h>
#include <mkl.h>
#include <omp.h>

#define READ_U64(stream, val) stream.read((char *)&val, sizeof(uint64_t))
#define READ_U32(stream, val) stream.read((char *)&val, sizeof(uint32_t))
#define READ_UNSIGNED(stream, val) stream.read((char *)&val, sizeof(unsigned))

// sector # beyond the end of graph where data for id is present for reordering
#define VECTOR_SECTOR_NO(id) (((uint64_t)(id)) / _nvecs_per_sector + _reorder_data_start_sector)

// sector # beyond the end of graph where data for id is present for reordering
#define VECTOR_SECTOR_OFFSET(id) ((((uint64_t)(id)) % _nvecs_per_sector) * _data_dim * sizeof(float))

namespace polaris {

    template<typename T>
    PQFlashIndex<T>::PQFlashIndex(std::shared_ptr<AlignedFileReader> &fileReader, polaris::MetricType m)
            : reader(fileReader), metric(m), _thread_data(nullptr) {
        polaris::MetricType metric_to_invoke = m;
        if (m == polaris::MetricType::METRIC_COSINE || m == polaris::MetricType::METRIC_INNER_PRODUCT) {
            if (std::is_floating_point<T>::value) {
                POLARIS_LOG(INFO)
                << "Since data is floating point, we assume that it has been appropriately pre-processed "
                   "(normalization for cosine, and convert-to-l2 by adding extra dimension for MIPS). So we "
                   "shall invoke an l2 distance function.";
                metric_to_invoke = polaris::MetricType::METRIC_L2;
            } else {
                POLARIS_LOG(INFO) << "WARNING: Cannot normalize integral data types."
                                  << " This may result in erroneous results or poor recall."
                                  << " Consider using L2 distance with integral data types.";
            }
        }

        this->_dist_cmp.reset(polaris::get_distance_function<T>(metric_to_invoke));
        this->_dist_cmp_float.reset(polaris::get_distance_function<float>(metric_to_invoke));
    }

    template<typename T>
    PQFlashIndex<T>::~PQFlashIndex() {
        if (data != nullptr) {
            delete[] data;
        }

        if (_centroid_data != nullptr)
            aligned_free(_centroid_data);
        // delete backing bufs for nhood and coord cache
        if (_nhood_cache_buf != nullptr) {
            delete[] _nhood_cache_buf;
            polaris::aligned_free(_coord_cache_buf);
        }

        if (_load_flag) {
            POLARIS_LOG(INFO) << "Clearing scratch";
            ScratchStoreManager<SSDThreadData<T>> manager(this->_thread_data);
            manager.destroy();
            this->reader->deregister_all_threads();
            reader->close();
        }

        if (_medoids != nullptr) {
            delete[] _medoids;
        }
    }

    template<typename T>
    inline uint64_t PQFlashIndex<T>::get_node_sector(uint64_t node_id) {
        return 1 + (_nnodes_per_sector > 0 ? node_id / _nnodes_per_sector
                                           : node_id * DIV_ROUND_UP(_max_node_len, defaults::SECTOR_LEN));
    }

    template<typename T>
    inline char *PQFlashIndex<T>::offset_to_node(char *sector_buf, uint64_t node_id) {
        return sector_buf + (_nnodes_per_sector == 0 ? 0 : (node_id % _nnodes_per_sector) * _max_node_len);
    }

    template<typename T>
    inline uint32_t *PQFlashIndex<T>::offset_to_node_nhood(char *node_buf) {
        return (unsigned *) (node_buf + _disk_bytes_per_point);
    }

    template<typename T>
    inline T *PQFlashIndex<T>::offset_to_node_coords(char *node_buf) {
        return (T *) (node_buf);
    }

    template<typename T>
    void PQFlashIndex<T>::setup_thread_data(uint64_t nthreads, uint64_t visited_reserve) {
        POLARIS_LOG(INFO)<< "Setting up thread-specific contexts for nthreads: " << nthreads;
// omp parallel for to generate unique thread IDs
#pragma omp parallel for num_threads((int)nthreads)
        for (int64_t thread = 0; thread < (int64_t) nthreads; thread++) {
#pragma omp critical
            {
                SSDThreadData<T> *data = new SSDThreadData<T>(this->_aligned_dim, visited_reserve);
                this->reader->register_thread();
                data->ctx = this->reader->get_ctx();
                this->_thread_data.push(data);
            }
        }
        _load_flag = true;
    }

    template<typename T>
    std::vector<bool> PQFlashIndex<T>::read_nodes(const std::vector<uint32_t> &node_ids,
                                                  std::vector<T *> &coord_buffers,
                                                  std::vector<std::pair<uint32_t, uint32_t *>> &nbr_buffers) {
        std::vector<AlignedRead> read_reqs;
        std::vector<bool> retval(node_ids.size(), true);

        char *buf = nullptr;
        auto num_sectors = _nnodes_per_sector > 0 ? 1 : DIV_ROUND_UP(_max_node_len, defaults::SECTOR_LEN);
        alloc_aligned((void **) &buf, node_ids.size() * num_sectors * defaults::SECTOR_LEN, defaults::SECTOR_LEN);

        // create read requests
        for (size_t i = 0; i < node_ids.size(); ++i) {
            auto node_id = node_ids[i];

            AlignedRead read;
            read.len = num_sectors * defaults::SECTOR_LEN;
            read.buf = buf + i * num_sectors * defaults::SECTOR_LEN;
            read.offset = get_node_sector(node_id) * defaults::SECTOR_LEN;
            read_reqs.push_back(read);
        }

        // borrow thread data and issue reads
        ScratchStoreManager<SSDThreadData<T>> manager(this->_thread_data);
        auto this_thread_data = manager.scratch_space();
        IOContext &ctx = this_thread_data->ctx;
        reader->read(read_reqs, ctx);

        // copy reads into buffers
        for (uint32_t i = 0; i < read_reqs.size(); i++) {

            char *node_buf = offset_to_node((char *) read_reqs[i].buf, node_ids[i]);

            if (coord_buffers[i] != nullptr) {
                T *node_coords = offset_to_node_coords(node_buf);
                memcpy(coord_buffers[i], node_coords, _disk_bytes_per_point);
            }

            if (nbr_buffers[i].second != nullptr) {
                uint32_t *node_nhood = offset_to_node_nhood(node_buf);
                auto num_nbrs = *node_nhood;
                nbr_buffers[i].first = num_nbrs;
                memcpy(nbr_buffers[i].second, node_nhood + 1, num_nbrs * sizeof(uint32_t));
            }
        }

        aligned_free(buf);

        return retval;
    }

    template<typename T>
    POLARIS_API turbo::Status
    PQFlashIndex<T>::build(const char *dataFilePath, const char *indexFilePath, const IndexConfig &config,
                           const std::vector<vid_t> &tags,
                           const std::string &codebook_prefix) {

        if (!std::is_same<T, float>::value &&
            (config.basic_config.metric == polaris::MetricType::METRIC_INNER_PRODUCT ||
             config.basic_config.metric == polaris::MetricType::METRIC_COSINE)) {
            std::stringstream stream;
            stream << "Disk-index build currently only supports floating point data for Max "
                      "Inner Product Search/ cosine similarity. "
                   << std::endl;
            return turbo::make_status(turbo::kInvalidArgument, stream.str());
        }

        // if there is a 6th parameter, it means we compress the disk index
        // vectors also using PQ data (for very large dimensionality data). If the
        // provided parameter is 0, it means we store full vectors.
        size_t disk_pq_dims = config.disk_config.pq_dims;
        bool use_disk_pq = disk_pq_dims > 0 ? true : false;
        size_t build_pq_bytes = config.disk_config.build_pq_bytes;

        bool reorder_data = config.disk_config.append_reorder_data;

        std::string base_file(dataFilePath);
        std::string data_file_to_use = base_file;
        std::string index_prefix_path(indexFilePath);
        std::string pq_pivots_path_base = codebook_prefix;
        std::string pq_pivots_path = collie::filesystem::exists(pq_pivots_path_base) ? pq_pivots_path_base +
                                                                                       "_pq_pivots.bin"
                                                                                     : index_prefix_path +
                                                                                       "_pq_pivots.bin";
        std::string pq_compressed_vectors_path = index_prefix_path + "_pq_compressed.bin";
        std::string mem_index_path = index_prefix_path + "_mem.index";
        std::string disk_index_path = index_prefix_path + "_disk.index";
        std::string medoids_path = disk_index_path + "_medoids.bin";
        std::string centroids_path = disk_index_path + "_centroids.bin";
        std::string tags_file = index_prefix_path + "_tags.bin";


        std::string sample_base_prefix = index_prefix_path + "_sample";
        // optional, used if disk index file must store pq data
        std::string disk_pq_pivots_path = index_prefix_path + "_disk.index_pq_pivots.bin";
        // optional, used if disk index must store pq data
        std::string disk_pq_compressed_vectors_path = index_prefix_path + "_disk.index_pq_compressed.bin";
        std::string prepped_base =
                index_prefix_path +
                "_prepped_base.bin"; // temp file for storing pre-processed base file for cosine/ mips metrics
        bool created_temp_file_for_processed_data = false;

        // output a new base file which contains extra dimension with sqrt(1 -
        // ||x||^2/M^2) for every x, M is max norm of all points. Extra space on
        // disk needed!
        if (config.basic_config.metric == polaris::MetricType::METRIC_INNER_PRODUCT) {
            Timer timer;
            POLARIS_LOG(INFO) << "Using Inner Product search, so need to pre-process base "
                         "data into temp file. Please ensure there is additional "
                         "(n*(d+1)*4) bytes for storing pre-processed base vectors, "
                         "apart from the interim indices created by DiskANN and the final index.";
            data_file_to_use = prepped_base;
            float max_norm_of_base = polaris::prepare_base_for_inner_products<T>(base_file, prepped_base);
            std::string norm_file = disk_index_path + "_max_base_norm.bin";
            auto rs = polaris::save_bin<float>(norm_file, &max_norm_of_base, 1, 1);
            if (!rs.ok()) {
                return rs.status();
            }
            POLARIS_LOG(INFO) << timer.elapsed_seconds_for_step("preprocessing data for inner product");
            created_temp_file_for_processed_data = true;
        } else if (config.basic_config.metric == polaris::MetricType::METRIC_COSINE) {
            Timer timer;
            POLARIS_LOG(INFO) << "Normalizing data for cosine to temporary file, please ensure there is additional "
                         "(n*d*4) bytes for storing normalized base vectors, "
                         "apart from the interim indices created by DiskANN and the final index.";
            data_file_to_use = prepped_base;
            polaris::normalize_data_file(base_file, prepped_base);
            POLARIS_LOG(INFO) << timer.elapsed_seconds_for_step("preprocessing data for cosine");
            created_temp_file_for_processed_data = true;
        }
        uint32_t R = config.disk_config.R;
        uint32_t L = config.disk_config.L;

        double final_index_ram_limit = get_memory_budget(config.disk_config.B);
        if (final_index_ram_limit <= 0) {
            return turbo::make_status(turbo::kInvalidArgument,
                                      "Insufficient memory budget (or string was not in right format). Should be > 0.");
        }
        double indexing_ram_budget = config.disk_config.M;
        if (indexing_ram_budget <= 0) {
            std::cerr << "Not building index. Please provide more RAM budget" << std::endl;
            return turbo::make_status(turbo::kInvalidArgument, "Not building index. Please provide more RAM budget");
        }
        uint32_t num_threads = config.disk_config.num_threads;

        if (num_threads != 0) {
            omp_set_num_threads(num_threads);
            mkl_set_num_threads(num_threads);
        }

        POLARIS_LOG(INFO)
        << "Starting index build: R=" << R << " L=" << L << " Query RAM budget: " << final_index_ram_limit
        << " Indexing ram budget: " << indexing_ram_budget << " T: " << num_threads;

        auto s = std::chrono::high_resolution_clock::now();
        size_t points_num, dim;
        Timer timer;
        polaris::get_bin_metadata(data_file_to_use.c_str(), points_num, dim);
        const double p_val = ((double) MAX_PQ_TRAINING_SET_SIZE / (double) points_num);

        if (use_disk_pq) {
            generate_disk_quantized_data<T>(data_file_to_use, disk_pq_pivots_path, disk_pq_compressed_vectors_path,
                                            config.basic_config.metric, p_val, disk_pq_dims);
        }
        size_t num_pq_chunks = (size_t) (std::floor)(uint64_t(final_index_ram_limit / points_num));

        num_pq_chunks = num_pq_chunks <= 0 ? 1 : num_pq_chunks;
        num_pq_chunks = num_pq_chunks > dim ? dim : num_pq_chunks;
        num_pq_chunks = num_pq_chunks > MAX_PQ_CHUNKS ? MAX_PQ_CHUNKS : num_pq_chunks;

        if (config.disk_config.pq_chunks <= MAX_PQ_CHUNKS && config.disk_config.pq_chunks > 0) {
            POLARIS_LOG(INFO)<< "Use quantized dimension (QD) to overwrite derived quantized "
                         "dimension from search_DRAM_budget (B)";
            num_pq_chunks = config.disk_config.pq_chunks;
        }

        POLARIS_LOG(INFO)
        << "Compressing " << dim << "-dimensional data into " << num_pq_chunks << " bytes per vector.";

        generate_quantized_data<T>(data_file_to_use, pq_pivots_path, pq_compressed_vectors_path,
                                   config.basic_config.metric, p_val,
                                   num_pq_chunks, config.disk_config.use_opq, codebook_prefix);
        POLARIS_LOG(INFO) << timer.elapsed_seconds_for_step("generating quantized data");

// Gopal. Splitting diskann_dll into separate DLLs for search and build.
// This code should only be available in the "build" DLL.
#if defined(DISKANN_RELEASE_UNUSED_TCMALLOC_MEMORY_AT_CHECKPOINTS) && defined(DISKANN_BUILD)
        MallocExtension::instance()->ReleaseFreeMemory();
#endif
        // Whether it is cosine or inner product, we still L2 metric due to the pre-processing.
        timer.reset();
        polaris::build_merged_vamana_index<T>(data_file_to_use.c_str(), polaris::MetricType::METRIC_L2, L, R, p_val,
                                              indexing_ram_budget, mem_index_path, medoids_path, centroids_path,
                                              build_pq_bytes, config.disk_config.use_opq, num_threads);
        POLARIS_LOG(INFO)<< timer.elapsed_seconds_for_step("building merged vamana index");

        timer.reset();
        if (!use_disk_pq) {
            polaris::create_disk_layout<T>(data_file_to_use.c_str(), mem_index_path, disk_index_path);
        } else {
            if (!reorder_data)
                polaris::create_disk_layout<uint8_t>(disk_pq_compressed_vectors_path, mem_index_path, disk_index_path);
            else
                polaris::create_disk_layout<uint8_t>(disk_pq_compressed_vectors_path, mem_index_path, disk_index_path,
                                                     data_file_to_use.c_str());
        }
        POLARIS_LOG(INFO)<< timer.elapsed_seconds_for_step("generating disk layout");

        double ten_percent_points = std::ceil(points_num * 0.1);
        double num_sample_points =
                ten_percent_points > MAX_SAMPLE_POINTS_FOR_WARMUP ? MAX_SAMPLE_POINTS_FOR_WARMUP : ten_percent_points;
        double sample_sampling_rate = num_sample_points / points_num;
        gen_random_slice<T>(data_file_to_use.c_str(), sample_base_prefix, sample_sampling_rate);
        if (created_temp_file_for_processed_data)
            std::remove(prepped_base.c_str());
        std::remove(mem_index_path.c_str());
        if (use_disk_pq)
            std::remove(disk_pq_compressed_vectors_path.c_str());

        auto r = save_bin(tags_file, tags.data(), tags.size(), 1);
        if (!r.ok()) {
            return turbo::make_status(turbo::kInternal, "Failed to save tags file");
        }

        auto e = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> diff = e - s;
        POLARIS_LOG(INFO)<< "Indexing time: " << diff.count();

        return turbo::ok_status();

    }

    template<typename T>
    POLARIS_API turbo::Status
    PQFlashIndex<T>::build(const char *dataFilePath, const char *indexFilePath, const std::string &tags_file,
                           const IndexConfig &indexConfig,
                           const std::string &codebook_prefix
    ) {
        std::vector<vid_t> tags;
        if (!tags_file.empty()) {
            size_t file_dim, file_num_points;
            vid_t *tag_data;
            load_bin<vid_t>(tags_file, tag_data, file_num_points, file_dim);
            if (file_dim != 1) {
                return turbo::make_status(turbo::kInvalidArgument, "Tags file must be a 1D array.");
            }
            tags.assign(tag_data, tag_data + file_num_points);
            delete[] tag_data;
        } else {
            return turbo::make_status(turbo::kInvalidArgument, "Tags file must be provided.");
        }
        return build(dataFilePath, indexFilePath, indexConfig, tags, codebook_prefix);
    }

    template<typename T>
    POLARIS_API turbo::Status
    PQFlashIndex<T>::build(const char *dataFilePath, const char *indexFilePath, const IndexConfig &indexConfig,
                           const std::string &codebook_prefix) {
        std::vector<vid_t> tags;
        size_t base_num, base_dim;
        polaris::get_bin_metadata(dataFilePath, base_num, base_dim);
        if (base_num <= 0) {
            return turbo::make_status(turbo::kInvalidArgument, "Data file is empty.");
        }
        tags.resize(base_num);
        for (size_t i = 0; i < base_num; i++) {
            tags[i] = i + 1;
        }
        return build(dataFilePath, indexFilePath, indexConfig, tags, codebook_prefix);
    }

    template<typename T>
    void PQFlashIndex<T>::load_cache_list(std::vector<uint32_t> &node_list) {
        POLARIS_LOG(INFO)<< "Loading the cache list into memory..";
        size_t num_cached_nodes = node_list.size();

        // borrow thread data
        ScratchStoreManager<SSDThreadData<T>> manager(this->_thread_data);
        //auto this_thread_data = manager.scratch_space();
        //IOContext &ctx = this_thread_data->ctx;

        // Allocate space for neighborhood cache
        _nhood_cache_buf = new uint32_t[num_cached_nodes * (_max_degree + 1)];
        memset(_nhood_cache_buf, 0, num_cached_nodes * (_max_degree + 1));

        // Allocate space for coordinate cache
        size_t coord_cache_buf_len = num_cached_nodes * _aligned_dim;
        polaris::alloc_aligned((void **) &_coord_cache_buf, coord_cache_buf_len * sizeof(T), 8 * sizeof(T));
        memset(_coord_cache_buf, 0, coord_cache_buf_len * sizeof(T));

        size_t BLOCK_SIZE = 8;
        size_t num_blocks = DIV_ROUND_UP(num_cached_nodes, BLOCK_SIZE);
        for (size_t block = 0; block < num_blocks; block++) {
            size_t start_idx = block * BLOCK_SIZE;
            size_t end_idx = (std::min)(num_cached_nodes, (block + 1) * BLOCK_SIZE);

            // Copy offset into buffers to read into
            std::vector<uint32_t> nodes_to_read;
            std::vector<T *> coord_buffers;
            std::vector<std::pair<uint32_t, uint32_t *>> nbr_buffers;
            for (size_t node_idx = start_idx; node_idx < end_idx; node_idx++) {
                nodes_to_read.push_back(node_list[node_idx]);
                coord_buffers.push_back(_coord_cache_buf + node_idx * _aligned_dim);
                nbr_buffers.emplace_back(0, _nhood_cache_buf + node_idx * (_max_degree + 1));
            }

            // issue the reads
            auto read_status = read_nodes(nodes_to_read, coord_buffers, nbr_buffers);

            // check for success and insert into the cache.
            for (size_t i = 0; i < read_status.size(); i++) {
                if (read_status[i] == true) {
                    _coord_cache.insert(std::make_pair(nodes_to_read[i], coord_buffers[i]));
                    _nhood_cache.insert(std::make_pair(nodes_to_read[i], nbr_buffers[i]));
                }
            }
        }
        POLARIS_LOG(INFO)<< "..done.";
    }

    template<typename T>
    void PQFlashIndex<T>::generate_cache_list_from_sample_queries(std::string sample_bin, uint64_t l_search,
                                                                  uint64_t beamwidth,
                                                                  uint64_t num_nodes_to_cache,
                                                                  uint32_t nthreads,
                                                                  std::vector<uint32_t> &node_list) {
        if (num_nodes_to_cache >= this->_num_points) {
            // for small num_points and big num_nodes_to_cache, use below way to get the node_list quickly
            node_list.resize(this->_num_points);
            for (uint32_t i = 0; i < this->_num_points; ++i) {
                node_list[i] = i;
            }
            return;
        }

        this->_count_visited_nodes = true;
        this->_node_visit_counter.clear();
        this->_node_visit_counter.resize(this->_num_points);
        for (uint32_t i = 0; i < _node_visit_counter.size(); i++) {
            this->_node_visit_counter[i].first = i;
            this->_node_visit_counter[i].second = 0;
        }

        uint64_t sample_num, sample_dim, sample_aligned_dim;
        T *samples;

        if (collie::filesystem::exists(sample_bin)) {
            polaris::load_aligned_bin<T>(sample_bin, samples, sample_num, sample_dim, sample_aligned_dim);
        } else {
            POLARIS_LOG(WARN)<< "Sample bin file not found. Not generating cache.";
            return;
        }

        std::vector<uint64_t> tmp_result_ids_64(sample_num, 0);
        std::vector<float> tmp_result_dists(sample_num, 0);

#pragma omp parallel for schedule(dynamic, 1) num_threads(nthreads)
        for (int64_t i = 0; i < (int64_t) sample_num; i++) {
            // run a search on the sample query with a random label (sampled from base label distribution), and it will
            // concurrently update the node_visit_counter to track most visited nodes. The last false is to not use the
            // "use_reorder_data" option which enables a final reranking if the disk index itself contains only PQ data.
            cached_beam_search(samples + (i * sample_aligned_dim), 1, l_search, tmp_result_ids_64.data() + i,
                               tmp_result_dists.data() + i, beamwidth, false);
        }

        std::sort(this->_node_visit_counter.begin(), _node_visit_counter.end(),
                  [](std::pair<uint32_t, uint32_t> &left, std::pair<uint32_t, uint32_t> &right) {
                      return left.second > right.second;
                  });
        node_list.clear();
        node_list.shrink_to_fit();
        num_nodes_to_cache = std::min(num_nodes_to_cache, this->_node_visit_counter.size());
        node_list.reserve(num_nodes_to_cache);
        for (uint64_t i = 0; i < num_nodes_to_cache; i++) {
            node_list.push_back(this->_node_visit_counter[i].first);
        }
        this->_count_visited_nodes = false;

        polaris::aligned_free(samples);
    }

    template<typename T>
    void PQFlashIndex<T>::cache_bfs_levels(uint64_t num_nodes_to_cache, std::vector<uint32_t> &node_list,
                                           const bool shuffle) {
        std::random_device rng;
        std::mt19937 urng(rng());

        turbo::flat_hash_set<uint32_t> node_set;

        // Do not cache more than 10% of the nodes in the index
        uint64_t tenp_nodes = (uint64_t) (std::round(this->_num_points * 0.1));
        if (num_nodes_to_cache > tenp_nodes) {
            POLARIS_LOG(INFO)<< "Reducing nodes to cache from: " << num_nodes_to_cache << " to: " << tenp_nodes
                          << "(10 percent of total nodes:" << this->_num_points << ")";
            num_nodes_to_cache = tenp_nodes == 0 ? 1 : tenp_nodes;
        }
        POLARIS_LOG(INFO)<< "Caching " << num_nodes_to_cache << "...";

        // borrow thread data
        ScratchStoreManager<SSDThreadData<T>> manager(this->_thread_data);
        //auto this_thread_data = manager.scratch_space();
        //IOContext &ctx = this_thread_data->ctx;

        std::unique_ptr<turbo::flat_hash_set<uint32_t>> cur_level, prev_level;
        cur_level = std::make_unique<turbo::flat_hash_set<uint32_t>>();
        prev_level = std::make_unique<turbo::flat_hash_set<uint32_t>>();

        for (uint64_t miter = 0; miter < _num_medoids && cur_level->size() < num_nodes_to_cache; miter++) {
            cur_level->insert(_medoids[miter]);
        }

        uint64_t lvl = 1;
        uint64_t prev_node_set_size = 0;
        while ((node_set.size() + cur_level->size() < num_nodes_to_cache) && cur_level->size() != 0) {
            // swap prev_level and cur_level
            std::swap(prev_level, cur_level);
            // clear cur_level
            cur_level->clear();

            std::vector<uint32_t> nodes_to_expand;

            for (const uint32_t &id: *prev_level) {
                if (node_set.find(id) != node_set.end()) {
                    continue;
                }
                node_set.insert(id);
                nodes_to_expand.push_back(id);
            }

            if (shuffle)
                std::shuffle(nodes_to_expand.begin(), nodes_to_expand.end(), urng);
            else
                std::sort(nodes_to_expand.begin(), nodes_to_expand.end());

            POLARIS_LOG(INFO)<< "Level: " << lvl;
            bool finish_flag = false;

            uint64_t BLOCK_SIZE = 1024;
            uint64_t nblocks = DIV_ROUND_UP(nodes_to_expand.size(), BLOCK_SIZE);
            for (size_t block = 0; block < nblocks && !finish_flag; block++) {
                POLARIS_LOG(INFO)<< ".";
                size_t start = block * BLOCK_SIZE;
                size_t end = (std::min)((block + 1) * BLOCK_SIZE, nodes_to_expand.size());

                std::vector<uint32_t> nodes_to_read;
                std::vector<T *> coord_buffers(end - start, nullptr);
                std::vector<std::pair<uint32_t, uint32_t *>> nbr_buffers;

                for (size_t cur_pt = start; cur_pt < end; cur_pt++) {
                    nodes_to_read.push_back(nodes_to_expand[cur_pt]);
                    nbr_buffers.emplace_back(0, new uint32_t[_max_degree + 1]);
                }

                // issue read requests
                auto read_status = read_nodes(nodes_to_read, coord_buffers, nbr_buffers);

                // process each nhood buf
                for (uint32_t i = 0; i < read_status.size(); i++) {
                    if (read_status[i] == false) {
                        continue;
                    } else {
                        uint32_t nnbrs = nbr_buffers[i].first;
                        uint32_t *nbrs = nbr_buffers[i].second;

                        // explore next level
                        for (uint32_t j = 0; j < nnbrs && !finish_flag; j++) {
                            if (node_set.find(nbrs[j]) == node_set.end()) {
                                cur_level->insert(nbrs[j]);
                            }
                            if (cur_level->size() + node_set.size() >= num_nodes_to_cache) {
                                finish_flag = true;
                            }
                        }
                    }
                    delete[] nbr_buffers[i].second;
                }
            }

            POLARIS_LOG(INFO)<< ". #nodes: " << node_set.size() - prev_node_set_size
                          << ", #nodes thus far: " << node_set.size();
            prev_node_set_size = node_set.size();
            lvl++;
        }

        assert(node_set.size() + cur_level->size() == num_nodes_to_cache || cur_level->size() == 0);

        node_list.clear();
        node_list.reserve(node_set.size() + cur_level->size());
        for (auto node: node_set)
            node_list.push_back(node);
        for (auto node: *cur_level)
            node_list.push_back(node);

        POLARIS_LOG(INFO)
        << "Level: " << lvl << ". #nodes: " << node_list.size() - prev_node_set_size << ", #nodes thus far: "
        << node_list.size();
        POLARIS_LOG(INFO) << "done";
    }

    template<typename T>
    void PQFlashIndex<T>::use_medoids_data_as_centroids() {
        if (_centroid_data != nullptr)
            aligned_free(_centroid_data);
        alloc_aligned(((void **) &_centroid_data), _num_medoids * _aligned_dim * sizeof(float), 32);
        std::memset(_centroid_data, 0, _num_medoids * _aligned_dim * sizeof(float));

        // borrow ctx
        ScratchStoreManager<SSDThreadData<T>> manager(this->_thread_data);
        //auto data = manager.scratch_space();
        //IOContext &ctx = data->ctx;
        POLARIS_LOG(INFO) << "Loading centroid data from medoids vector data of " << _num_medoids << " medoid(s)";

        std::vector<uint32_t> nodes_to_read;
        std::vector<T *> medoid_bufs;
        std::vector<std::pair<uint32_t, uint32_t *>> nbr_bufs;

        for (uint64_t cur_m = 0; cur_m < _num_medoids; cur_m++) {
            nodes_to_read.push_back(_medoids[cur_m]);
            medoid_bufs.push_back(new T[_data_dim]);
            nbr_bufs.emplace_back(0, nullptr);
        }

        auto read_status = read_nodes(nodes_to_read, medoid_bufs, nbr_bufs);

        for (uint64_t cur_m = 0; cur_m < _num_medoids; cur_m++) {
            if (read_status[cur_m] == true) {
                if (!_use_disk_index_pq) {
                    for (uint32_t i = 0; i < _data_dim; i++)
                        _centroid_data[cur_m * _aligned_dim + i] = medoid_bufs[cur_m][i];
                } else {
                    _disk_pq_table.inflate_vector((uint8_t *) medoid_bufs[cur_m],
                                                  (_centroid_data + cur_m * _aligned_dim));
                }
            } else {
                throw PolarisException("Unable to read a medoid", -1, __PRETTY_FUNCTION__, __FILE__, __LINE__);
            }
            delete[] medoid_bufs[cur_m];
        }
    }

    template<typename T>
    void PQFlashIndex<T>::reset_stream_for_reading(std::basic_istream<char> &infile) {
        infile.clear();
        infile.seekg(0);
    }

    template<typename T>
    turbo::Status PQFlashIndex<T>::load(uint32_t num_threads, const char *index_prefix) {
        std::string pq_table_bin = std::string(index_prefix) + "_pq_pivots.bin";
        std::string pq_compressed_vectors = std::string(index_prefix) + "_pq_compressed.bin";
        std::string _disk_index_file = std::string(index_prefix) + "_disk.index";
        std::string tags_file = std::string(index_prefix) + "_tags.bin";
        size_t file_dim, file_num_points;
        vid_t *tag_data;
        load_bin<vid_t>(std::string(tags_file), tag_data, file_num_points, file_dim);
        if (file_dim != 1) {
            return turbo::make_status(turbo::kInvalidArgument, "Tags file should have 1 dimension.");
        }
        _location_to_tag.reserve(file_num_points);
        for (uint32_t i = 0; i < (uint32_t) file_num_points; i++) {
            vid_t tag = *(tag_data + i);
            _location_to_tag.set(i, tag);
            _tag_to_location[tag] = i;
        }
        return load_from_separate_paths(num_threads, _disk_index_file.c_str(), pq_table_bin.c_str(),
                                        pq_compressed_vectors.c_str());
    }

    template<typename T>
    turbo::Status PQFlashIndex<T>::load_from_separate_paths(uint32_t num_threads, const char *index_filepath,
                                                            const char *pivots_filepath,
                                                            const char *compressed_filepath) {
        std::string pq_table_bin = pivots_filepath;
        std::string pq_compressed_vectors = compressed_filepath;
        std::string _disk_index_file = index_filepath;
        std::string medoids_file = std::string(_disk_index_file) + "_medoids.bin";
        std::string centroids_file = std::string(_disk_index_file) + "_centroids.bin";

        size_t pq_file_dim, pq_file_num_centroids;
        get_bin_metadata(pq_table_bin, pq_file_num_centroids, pq_file_dim, METADATA_SIZE);

        this->_disk_index_file = _disk_index_file;

        if (pq_file_num_centroids != 256) {
            return turbo::make_status(turbo::kInvalidArgument, "Number of PQ centroids:{} is not 256.",
                                      pq_file_num_centroids);
        }

        this->_data_dim = pq_file_dim;
        // will change later if we use PQ on disk or if we are using
        // inner product without PQ
        this->_disk_bytes_per_point = this->_data_dim * sizeof(T);
        this->_aligned_dim = ROUND_UP(pq_file_dim, 8);

        size_t npts_u64, nchunks_u64;
        polaris::load_bin<uint8_t>(pq_compressed_vectors, this->data, npts_u64, nchunks_u64);

        this->_num_points = npts_u64;
        this->_n_chunks = nchunks_u64;
        _pq_table.load_pq_centroid_bin(pq_table_bin.c_str(), nchunks_u64);

        POLARIS_LOG(INFO) << "Loaded PQ centroids and in-memory compressed vectors. #points: " << _num_points
                          << " #dim: " << _data_dim << " #aligned_dim: " << _aligned_dim << " #chunks: " << _n_chunks;

        if (_n_chunks > MAX_PQ_CHUNKS) {
            return turbo::make_status(turbo::kInvalidArgument,
                                      "Error loading index. Ensure that max PQ bytes for in-memory PQ data does not exceed {}.",
                                      MAX_PQ_CHUNKS);
        }

        std::string disk_pq_pivots_path = this->_disk_index_file + "_pq_pivots.bin";
        if (collie::filesystem::exists(disk_pq_pivots_path)) {
            _use_disk_index_pq = true;
            // giving 0 chunks to make the _pq_table infer from the
            // chunk_offsets file the correct value
            _disk_pq_table.load_pq_centroid_bin(disk_pq_pivots_path.c_str(), 0);
            _disk_pq_n_chunks = _disk_pq_table.get_num_chunks();
            _disk_bytes_per_point =
                    _disk_pq_n_chunks * sizeof(uint8_t); // revising disk_bytes_per_point since DISK PQ is used.
            POLARIS_LOG(INFO) << "Disk index uses PQ data compressed down to "
                              << _disk_pq_n_chunks << " bytes per point.";
        }

        std::ifstream index_metadata(_disk_index_file, std::ios::binary);

        uint32_t nr, nc; // metadata itself is stored as bin format (nr is number of
        // metadata, nc should be 1)
        READ_U32(index_metadata, nr);
        READ_U32(index_metadata, nc);

        uint64_t disk_nnodes;
        uint64_t disk_ndims; // can be disk PQ dim if disk_PQ is set to true
        READ_U64(index_metadata, disk_nnodes);
        READ_U64(index_metadata, disk_ndims);

        if (disk_nnodes != _num_points) {
            POLARIS_LOG(ERROR) << "Mismatch in #points for compressed data file and disk "
                                  "index file: "
                               << disk_nnodes << " vs " << _num_points;
            return turbo::make_status(turbo::kInvalidArgument,
                                      "Mismatch in #points for compressed data file and disk index file: {} vs {}",
                                      disk_nnodes, _num_points);
        }

        size_t medoid_id_on_file;
        READ_U64(index_metadata, medoid_id_on_file);
        READ_U64(index_metadata, _max_node_len);
        READ_U64(index_metadata, _nnodes_per_sector);
        _max_degree = ((_max_node_len - _disk_bytes_per_point) / sizeof(uint32_t)) - 1;

        if (_max_degree > defaults::MAX_GRAPH_DEGREE) {
            return turbo::make_status(turbo::kInvalidArgument,
                                      "Error loading index. Ensure that max graph degree (R) does not exceed {}.",
                                      defaults::MAX_GRAPH_DEGREE);
        }

        // setting up concept of frozen points in disk index for streaming-DiskANN
        READ_U64(index_metadata, this->_num_frozen_points);
        uint64_t file_frozen_id;
        READ_U64(index_metadata, file_frozen_id);
        if (this->_num_frozen_points == 1)
            this->_frozen_location = file_frozen_id;
        if (this->_num_frozen_points == 1) {
            POLARIS_LOG(INFO) << " Detected frozen point in index at location " << this->_frozen_location
                              << ". Will not output it at search time.";
        }

        READ_U64(index_metadata, this->_reorder_data_exists);
        if (this->_reorder_data_exists) {
            if (this->_use_disk_index_pq == false) {
                return turbo::make_status(turbo::kInvalidArgument,
                                          "Reordering is designed for used with disk PQ compression option.");
            }
            READ_U64(index_metadata, this->_reorder_data_start_sector);
            READ_U64(index_metadata, this->_ndims_reorder_vecs);
            READ_U64(index_metadata, this->_nvecs_per_sector);
        }

        POLARIS_LOG(INFO)<< "VamanaIndex File Meta-data: "
        << "# nodes per sector: " << _nnodes_per_sector
        << ", max node len (bytes): " << _max_node_len
        << ", max node degree: " << _max_degree;
        index_metadata.close();
        // open AlignedFileReader handle to index_file
        std::string index_fname(_disk_index_file);
        reader->open(index_fname);
        this->setup_thread_data(num_threads);
        this->_max_nthreads = num_threads;
        if (collie::filesystem::exists(medoids_file)) {
            size_t tmp_dim;
            polaris::load_bin<uint32_t>(medoids_file, _medoids, _num_medoids, tmp_dim);

            if (tmp_dim != 1) {
                return turbo::make_status(turbo::kInvalidArgument,
                                          "Error loading medoids file. Expected bin format of m times 1 vector of uint32_t.");
            }
            if (!collie::filesystem::exists(centroids_file)) {
                POLARIS_LOG(INFO) << "Centroid data file not found. Using corresponding vectors "
                                 "for the medoids ";
                use_medoids_data_as_centroids();
            } else {
                size_t num_centroids, aligned_tmp_dim;
                polaris::load_aligned_bin<float>(centroids_file, _centroid_data, num_centroids, tmp_dim,
                                                 aligned_tmp_dim);
                if (aligned_tmp_dim != _aligned_dim || num_centroids != _num_medoids) {
                    std::stringstream stream;
                    stream << "Error loading centroids data file. Expected bin format "
                              "of "
                              "m times data_dim vector of float, where m is number of "
                              "medoids "
                              "in medoids file.";
                    POLARIS_LOG(ERROR) << stream.str();
                    return turbo::make_status(turbo::kInvalidArgument, stream.str());
                }
            }
        } else {
            _num_medoids = 1;
            _medoids = new uint32_t[1];
            _medoids[0] = (uint32_t) (medoid_id_on_file);
            use_medoids_data_as_centroids();
        }

        std::string norm_file = std::string(_disk_index_file) + "_max_base_norm.bin";
        if (collie::filesystem::exists(norm_file) && metric == polaris::MetricType::METRIC_INNER_PRODUCT) {
            uint64_t dumr, dumc;
            float *norm_val;
            polaris::load_bin<float>(norm_file, norm_val, dumr, dumc);
            this->_max_base_norm = norm_val[0];
            POLARIS_LOG(INFO) << "Setting re-scaling factor of base vectors to " << this->_max_base_norm;
            delete[] norm_val;
        }
        POLARIS_LOG(INFO) << "loading done..";
        return turbo::ok_status();
    }

    template<typename T>
    void PQFlashIndex<T>::cached_beam_search(const T *query1, const uint64_t k_search, const uint64_t l_search,
                                             uint64_t *indices, float *distances, const uint64_t beam_width,
                                             const bool use_reorder_data, QueryStats *stats) {
        cached_beam_search(query1, k_search, l_search, indices, distances, beam_width,
                           std::numeric_limits<uint32_t>::max(),
                           use_reorder_data, stats);
    }

    template<typename T>
    turbo::Status PQFlashIndex<T>::search(SearchContext &sctx) {
        auto *stats = sctx.stats;
        uint64_t num_sector_per_nodes = DIV_ROUND_UP(_max_node_len, defaults::SECTOR_LEN);
        if (sctx.vd_beam_width > num_sector_per_nodes * defaults::MAX_N_SECTOR_READS) {
            return turbo::make_status(turbo::kInvalidArgument,
                                      "Beamwidth can not be higher than defaults::MAX_N_SECTOR_READS");
        }

        ScratchStoreManager<SSDThreadData<T>> manager(this->_thread_data);
        auto data = manager.scratch_space();
        IOContext &ctx = data->ctx;
        auto query_scratch = &(data->scratch);
        auto pq_query_scratch = query_scratch->pq_scratch();

        // reset query scratch
        query_scratch->reset();

        // copy query to thread specific aligned and allocated memory (for distance
        // calculations we need aligned data)
        float query_norm = 0;
        T *aligned_query_T = query_scratch->aligned_query_T();
        float *query_float = pq_query_scratch->aligned_query_float;
        float *query_rotated = pq_query_scratch->rotated_query;

        // normalization step. for cosine, we simply normalize the query
        // for mips, we normalize the first d-1 dims, and add a 0 for last dim, since an extra coordinate was used to
        // convert MIPS to L2 search
        auto query1 = (T *) sctx.query.data();
        if (metric == polaris::MetricType::METRIC_INNER_PRODUCT || metric == polaris::MetricType::METRIC_COSINE) {
            uint64_t inherent_dim = (metric == polaris::MetricType::METRIC_COSINE) ? this->_data_dim : (uint64_t) (
                    this->_data_dim - 1);
            for (size_t i = 0; i < inherent_dim; i++) {
                aligned_query_T[i] = query1[i];
                query_norm += query1[i] * query1[i];
            }
            if (metric == polaris::MetricType::METRIC_INNER_PRODUCT)
                aligned_query_T[this->_data_dim - 1] = 0;

            query_norm = std::sqrt(query_norm);

            for (size_t i = 0; i < inherent_dim; i++) {
                aligned_query_T[i] = (T) (aligned_query_T[i] / query_norm);
            }
            pq_query_scratch->initialize(this->_data_dim, aligned_query_T);
        } else {
            for (size_t i = 0; i < this->_data_dim; i++) {
                aligned_query_T[i] = query1[i];
            }
            pq_query_scratch->initialize(this->_data_dim, aligned_query_T);
        }

        // pointers to buffers for data
        T *data_buf = query_scratch->coord_scratch;
        _mm_prefetch((char *) data_buf, _MM_HINT_T1);

        // sector scratch
        char *sector_scratch = query_scratch->sector_scratch;
        uint64_t &sector_scratch_idx = query_scratch->sector_idx;
        const uint64_t num_sectors_per_node =
                _nnodes_per_sector > 0 ? 1 : DIV_ROUND_UP(_max_node_len, defaults::SECTOR_LEN);

        // query <-> PQ chunk centers distances
        _pq_table.preprocess_query(query_rotated); // center the query and rotate if
        // we have a rotation matrix
        float *pq_dists = pq_query_scratch->aligned_pqtable_dist_scratch;
        _pq_table.populate_chunk_distances(query_rotated, pq_dists);

        // query <-> neighbor list
        float *dist_scratch = pq_query_scratch->aligned_dist_scratch;
        uint8_t *pq_coord_scratch = pq_query_scratch->aligned_pq_coord_scratch;

        // lambda to batch compute query<-> node distances in PQ space
        auto compute_dists = [this, pq_coord_scratch, pq_dists](const uint32_t *ids, const uint64_t n_ids,
                                                                float *dists_out) {
            polaris::aggregate_coords(ids, n_ids, this->data, this->_n_chunks, pq_coord_scratch);
            polaris::pq_dist_lookup(pq_coord_scratch, n_ids, this->_n_chunks, pq_dists, dists_out);
        };
        Timer query_timer, io_timer, cpu_timer;

        turbo::flat_hash_set<uint64_t> &visited = query_scratch->visited;
        NeighborPriorityQueue &retset = query_scratch->retset;
        retset.reserve(sctx.search_list);
        std::vector<Neighbor> &full_retset = query_scratch->full_retset;

        uint32_t best_medoid = 0;
        float best_dist = (std::numeric_limits<float>::max)();
        for (uint64_t cur_m = 0; cur_m < _num_medoids; cur_m++) {
            float cur_expanded_dist =
                    _dist_cmp_float->compare(query_float, _centroid_data + _aligned_dim * cur_m,
                                             (uint32_t) _aligned_dim);
            if (cur_expanded_dist < best_dist) {
                best_medoid = _medoids[cur_m];
                best_dist = cur_expanded_dist;
            }
        }

        compute_dists(&best_medoid, 1, dist_scratch);
        retset.insert(Neighbor(best_medoid, dist_scratch[0]));
        visited.insert(best_medoid);

        uint32_t cmps = 0;
        uint32_t hops = 0;
        uint32_t num_ios = 0;

        // cleared every iteration
        std::vector<uint32_t> frontier;
        frontier.reserve(2 * sctx.vd_beam_width);
        std::vector<std::pair<uint32_t, char *>> frontier_nhoods;
        frontier_nhoods.reserve(2 * sctx.vd_beam_width);
        std::vector<AlignedRead> frontier_read_reqs;
        frontier_read_reqs.reserve(2 * sctx.vd_beam_width);
        std::vector<std::pair<uint32_t, std::pair<uint32_t, uint32_t *>>> cached_nhoods;
        cached_nhoods.reserve(2 * sctx.vd_beam_width);

        while (retset.has_unexpanded_node() && num_ios < sctx.vd_io_limit) {
            // clear iteration state
            frontier.clear();
            frontier_nhoods.clear();
            frontier_read_reqs.clear();
            cached_nhoods.clear();
            sector_scratch_idx = 0;
            // find new beam
            uint32_t num_seen = 0;
            while (retset.has_unexpanded_node() && frontier.size() < sctx.vd_beam_width &&
                   num_seen < sctx.vd_beam_width) {
                auto nbr = retset.closest_unexpanded();
                num_seen++;
                auto iter = _nhood_cache.find(nbr.id);
                if (iter != _nhood_cache.end()) {
                    cached_nhoods.push_back(std::make_pair(nbr.id, iter->second));
                    if (stats != nullptr) {
                        stats->n_cache_hits++;
                    }
                } else {
                    frontier.push_back(nbr.id);
                }
                if (this->_count_visited_nodes) {
                    reinterpret_cast<std::atomic<uint32_t> &>(this->_node_visit_counter[nbr.id].second).fetch_add(1);
                }
            }

            // read nhoods of frontier ids
            if (!frontier.empty()) {
                if (stats != nullptr)
                    stats->n_hops++;
                for (uint64_t i = 0; i < frontier.size(); i++) {
                    auto id = frontier[i];
                    std::pair<uint32_t, char *> fnhood;
                    fnhood.first = id;
                    fnhood.second = sector_scratch + num_sectors_per_node * sector_scratch_idx * defaults::SECTOR_LEN;
                    sector_scratch_idx++;
                    frontier_nhoods.push_back(fnhood);
                    frontier_read_reqs.emplace_back(get_node_sector((size_t) id) * defaults::SECTOR_LEN,
                                                    num_sectors_per_node * defaults::SECTOR_LEN, fnhood.second);
                    if (stats != nullptr) {
                        stats->n_4k++;
                        stats->n_ios++;
                    }
                    num_ios++;
                }
                io_timer.reset();
                reader->read(frontier_read_reqs, ctx); // synchronous IO linux
                if (stats != nullptr) {
                    stats->io_us += (float) io_timer.elapsed();
                }
            }

            // process cached nhoods
            vid_t tmp_id;
            for (auto &cached_nhood: cached_nhoods) {
                auto global_cache_iter = _coord_cache.find(cached_nhood.first);
                T *node_fp_coords_copy = global_cache_iter->second;
                float cur_expanded_dist;
                if (!_use_disk_index_pq) {
                    cur_expanded_dist = _dist_cmp->compare(aligned_query_T, node_fp_coords_copy,
                                                           (uint32_t) _aligned_dim);
                } else {
                    if (metric == polaris::MetricType::METRIC_INNER_PRODUCT)
                        cur_expanded_dist = _disk_pq_table.inner_product(query_float, (uint8_t *) node_fp_coords_copy);
                    else
                        cur_expanded_dist = _disk_pq_table.l2_distance( // disk_pq does not support OPQ yet
                                query_float, (uint8_t *) node_fp_coords_copy);
                }
                if (!_location_to_tag.try_get(cached_nhood.first, tmp_id)) {
                    return turbo::make_status(turbo::kInternal, "Error getting tag for node");
                }
                if (!sctx.search_condition->is_in_blacklist(tmp_id)) {
                    full_retset.push_back(Neighbor((uint32_t) cached_nhood.first, cur_expanded_dist));
                }
                uint64_t nnbrs = cached_nhood.second.first;
                uint32_t *node_nbrs = cached_nhood.second.second;

                // compute node_nbrs <-> query dists in PQ space
                cpu_timer.reset();
                compute_dists(node_nbrs, nnbrs, dist_scratch);
                if (stats != nullptr) {
                    stats->n_cmps += (uint32_t) nnbrs;
                    stats->cpu_us += (float) cpu_timer.elapsed();
                }

                // process prefetched nhood
                for (uint64_t m = 0; m < nnbrs; ++m) {
                    uint32_t id = node_nbrs[m];
                    if (visited.insert(id).second) {
                        cmps++;
                        float dist = dist_scratch[m];
                        Neighbor nn(id, dist);
                        retset.insert(nn);
                    }
                }
            }
            for (auto &frontier_nhood: frontier_nhoods) {
                char *node_disk_buf = offset_to_node(frontier_nhood.second, frontier_nhood.first);
                uint32_t *node_buf = offset_to_node_nhood(node_disk_buf);
                uint64_t nnbrs = (uint64_t) (*node_buf);
                T *node_fp_coords = offset_to_node_coords(node_disk_buf);
                memcpy(data_buf, node_fp_coords, _disk_bytes_per_point);
                float cur_expanded_dist;
                if (!_use_disk_index_pq) {
                    cur_expanded_dist = _dist_cmp->compare(aligned_query_T, data_buf, (uint32_t) _aligned_dim);
                } else {
                    if (metric == polaris::MetricType::METRIC_INNER_PRODUCT)
                        cur_expanded_dist = _disk_pq_table.inner_product(query_float, (uint8_t *) data_buf);
                    else
                        cur_expanded_dist = _disk_pq_table.l2_distance(query_float, (uint8_t *) data_buf);
                }
                if (!_location_to_tag.try_get(frontier_nhood.first, tmp_id)) {
                    return turbo::make_status(turbo::kInternal, "Error getting tag for node");
                }

                if (!sctx.search_condition->is_in_blacklist(tmp_id)) {
                    full_retset.push_back(Neighbor((uint32_t) frontier_nhood.first, cur_expanded_dist));
                }
                uint32_t *node_nbrs = (node_buf + 1);
                // compute node_nbrs <-> query dist in PQ space
                cpu_timer.reset();
                compute_dists(node_nbrs, nnbrs, dist_scratch);
                if (stats != nullptr) {
                    stats->n_cmps += (uint32_t) nnbrs;
                    stats->cpu_us += (float) cpu_timer.elapsed();
                }

                cpu_timer.reset();
                // process prefetch-ed nhood
                for (uint64_t m = 0; m < nnbrs; ++m) {
                    uint32_t id = node_nbrs[m];
                    if (visited.insert(id).second) {
                        cmps++;
                        float dist = dist_scratch[m];
                        if (stats != nullptr) {
                            stats->n_cmps++;
                        }

                        Neighbor nn(id, dist);
                        retset.insert(nn);
                    }
                }

                if (stats != nullptr) {
                    stats->cpu_us += (float) cpu_timer.elapsed();
                }
            }

            hops++;
        }

        // re-sort by distance
        std::sort(full_retset.begin(), full_retset.end());

        if (sctx.vd_use_reorder_data) {
            if (!(this->_reorder_data_exists)) {
                throw PolarisException("Requested use of reordering data which does "
                                       "not exist in index "
                                       "file",
                                       -1, __PRETTY_FUNCTION__, __FILE__, __LINE__);
            }

            std::vector<AlignedRead> vec_read_reqs;

            if (full_retset.size() > sctx.top_k * FULL_PRECISION_REORDER_MULTIPLIER)
                full_retset.erase(full_retset.begin() + sctx.top_k * FULL_PRECISION_REORDER_MULTIPLIER,
                                  full_retset.end());

            for (size_t i = 0; i < full_retset.size(); ++i) {
                // MULTISECTORFIX
                vec_read_reqs.emplace_back(VECTOR_SECTOR_NO(((size_t) full_retset[i].id)) * defaults::SECTOR_LEN,
                                           defaults::SECTOR_LEN, sector_scratch + i * defaults::SECTOR_LEN);

                if (stats != nullptr) {
                    stats->n_4k++;
                    stats->n_ios++;
                }
            }

            io_timer.reset();
            reader->read(vec_read_reqs, ctx); // synchronous IO linux
            if (stats != nullptr) {
                stats->io_us += io_timer.elapsed();
            }

            for (size_t i = 0; i < full_retset.size(); ++i) {
                auto id = full_retset[i].id;
                // MULTISECTORFIX
                auto location = (sector_scratch + i * defaults::SECTOR_LEN) + VECTOR_SECTOR_OFFSET(id);
                full_retset[i].distance = _dist_cmp->compare(aligned_query_T, (T *) location,
                                                             (uint32_t) this->_data_dim);
            }

            std::sort(full_retset.begin(), full_retset.end());
        }

        // copy k_search values
        vid_t rid;
        if (sctx.with_local_ids) {
            sctx.local_ids.reserve(sctx.top_k);
        }
        for (uint64_t i = 0; i < full_retset.size(); i++) {
            _location_to_tag.try_get(full_retset[i].id, rid);
            distance_t dist = full_retset[i].distance;
            if (metric == polaris::MetricType::METRIC_INNER_PRODUCT) {
                // flip the sign to convert min to max
                dist = (-dist);
                // rescale to revert back to original norms (cancelling the
                // effect of base and query pre-processing)
                if (_max_base_norm != 0)
                    dist *= (_max_base_norm * query_norm);
            }
            sctx.top_k_queue.emplace_back(rid, dist);
            if (sctx.with_local_ids) {
                sctx.local_ids.push_back(full_retset[i].id);
            }
            if (sctx.top_k_queue.size() == sctx.top_k) {
                break;
            }
        }

        if (stats != nullptr) {
            stats->total_us = (float) query_timer.elapsed();
        }
        return turbo::ok_status();
    }

    template<typename T>
    void PQFlashIndex<T>::cached_beam_search(const T *query1, const uint64_t k_search, const uint64_t l_search,
                                             uint64_t *indices, float *distances, const uint64_t beam_width,
                                             const uint32_t io_limit, const bool use_reorder_data,
                                             QueryStats *stats) {

        uint64_t num_sector_per_nodes = DIV_ROUND_UP(_max_node_len, defaults::SECTOR_LEN);
        if (beam_width > num_sector_per_nodes * defaults::MAX_N_SECTOR_READS)
            throw PolarisException("Beamwidth can not be higher than defaults::MAX_N_SECTOR_READS", -1,
                                   __PRETTY_FUNCTION__, __FILE__,
                                   __LINE__);

        ScratchStoreManager<SSDThreadData<T>> manager(this->_thread_data);
        auto data = manager.scratch_space();
        IOContext &ctx = data->ctx;
        auto query_scratch = &(data->scratch);
        auto pq_query_scratch = query_scratch->pq_scratch();

        // reset query scratch
        query_scratch->reset();

        // copy query to thread specific aligned and allocated memory (for distance
        // calculations we need aligned data)
        float query_norm = 0;
        T *aligned_query_T = query_scratch->aligned_query_T();
        float *query_float = pq_query_scratch->aligned_query_float;
        float *query_rotated = pq_query_scratch->rotated_query;

        // normalization step. for cosine, we simply normalize the query
        // for mips, we normalize the first d-1 dims, and add a 0 for last dim, since an extra coordinate was used to
        // convert MIPS to L2 search
        if (metric == polaris::MetricType::METRIC_INNER_PRODUCT || metric == polaris::MetricType::METRIC_COSINE) {
            uint64_t inherent_dim = (metric == polaris::MetricType::METRIC_COSINE) ? this->_data_dim : (uint64_t) (
                    this->_data_dim - 1);
            for (size_t i = 0; i < inherent_dim; i++) {
                aligned_query_T[i] = query1[i];
                query_norm += query1[i] * query1[i];
            }
            if (metric == polaris::MetricType::METRIC_INNER_PRODUCT)
                aligned_query_T[this->_data_dim - 1] = 0;

            query_norm = std::sqrt(query_norm);

            for (size_t i = 0; i < inherent_dim; i++) {
                aligned_query_T[i] = (T) (aligned_query_T[i] / query_norm);
            }
            pq_query_scratch->initialize(this->_data_dim, aligned_query_T);
        } else {
            for (size_t i = 0; i < this->_data_dim; i++) {
                aligned_query_T[i] = query1[i];
            }
            pq_query_scratch->initialize(this->_data_dim, aligned_query_T);
        }

        // pointers to buffers for data
        T *data_buf = query_scratch->coord_scratch;
        _mm_prefetch((char *) data_buf, _MM_HINT_T1);

        // sector scratch
        char *sector_scratch = query_scratch->sector_scratch;
        uint64_t &sector_scratch_idx = query_scratch->sector_idx;
        const uint64_t num_sectors_per_node =
                _nnodes_per_sector > 0 ? 1 : DIV_ROUND_UP(_max_node_len, defaults::SECTOR_LEN);

        // query <-> PQ chunk centers distances
        _pq_table.preprocess_query(query_rotated); // center the query and rotate if
        // we have a rotation matrix
        float *pq_dists = pq_query_scratch->aligned_pqtable_dist_scratch;
        _pq_table.populate_chunk_distances(query_rotated, pq_dists);

        // query <-> neighbor list
        float *dist_scratch = pq_query_scratch->aligned_dist_scratch;
        uint8_t *pq_coord_scratch = pq_query_scratch->aligned_pq_coord_scratch;

        // lambda to batch compute query<-> node distances in PQ space
        auto compute_dists = [this, pq_coord_scratch, pq_dists](const uint32_t *ids, const uint64_t n_ids,
                                                                float *dists_out) {
            polaris::aggregate_coords(ids, n_ids, this->data, this->_n_chunks, pq_coord_scratch);
            polaris::pq_dist_lookup(pq_coord_scratch, n_ids, this->_n_chunks, pq_dists, dists_out);
        };
        Timer query_timer, io_timer, cpu_timer;

        turbo::flat_hash_set<uint64_t> &visited = query_scratch->visited;
        NeighborPriorityQueue &retset = query_scratch->retset;
        retset.reserve(l_search);
        std::vector<Neighbor> &full_retset = query_scratch->full_retset;

        uint32_t best_medoid = 0;
        float best_dist = (std::numeric_limits<float>::max)();
        for (uint64_t cur_m = 0; cur_m < _num_medoids; cur_m++) {
            float cur_expanded_dist =
                    _dist_cmp_float->compare(query_float, _centroid_data + _aligned_dim * cur_m,
                                             (uint32_t) _aligned_dim);
            if (cur_expanded_dist < best_dist) {
                best_medoid = _medoids[cur_m];
                best_dist = cur_expanded_dist;
            }
        }

        compute_dists(&best_medoid, 1, dist_scratch);
        retset.insert(Neighbor(best_medoid, dist_scratch[0]));
        visited.insert(best_medoid);

        uint32_t cmps = 0;
        uint32_t hops = 0;
        uint32_t num_ios = 0;

        // cleared every iteration
        std::vector<uint32_t> frontier;
        frontier.reserve(2 * beam_width);
        std::vector<std::pair<uint32_t, char *>> frontier_nhoods;
        frontier_nhoods.reserve(2 * beam_width);
        std::vector<AlignedRead> frontier_read_reqs;
        frontier_read_reqs.reserve(2 * beam_width);
        std::vector<std::pair<uint32_t, std::pair<uint32_t, uint32_t *>>> cached_nhoods;
        cached_nhoods.reserve(2 * beam_width);

        while (retset.has_unexpanded_node() && num_ios < io_limit) {
            // clear iteration state
            frontier.clear();
            frontier_nhoods.clear();
            frontier_read_reqs.clear();
            cached_nhoods.clear();
            sector_scratch_idx = 0;
            // find new beam
            uint32_t num_seen = 0;
            while (retset.has_unexpanded_node() && frontier.size() < beam_width && num_seen < beam_width) {
                auto nbr = retset.closest_unexpanded();
                num_seen++;
                auto iter = _nhood_cache.find(nbr.id);
                if (iter != _nhood_cache.end()) {
                    cached_nhoods.push_back(std::make_pair(nbr.id, iter->second));
                    if (stats != nullptr) {
                        stats->n_cache_hits++;
                    }
                } else {
                    frontier.push_back(nbr.id);
                }
                if (this->_count_visited_nodes) {
                    reinterpret_cast<std::atomic<uint32_t> &>(this->_node_visit_counter[nbr.id].second).fetch_add(1);
                }
            }

            // read nhoods of frontier ids
            if (!frontier.empty()) {
                if (stats != nullptr)
                    stats->n_hops++;
                for (uint64_t i = 0; i < frontier.size(); i++) {
                    auto id = frontier[i];
                    std::pair<uint32_t, char *> fnhood;
                    fnhood.first = id;
                    fnhood.second = sector_scratch + num_sectors_per_node * sector_scratch_idx * defaults::SECTOR_LEN;
                    sector_scratch_idx++;
                    frontier_nhoods.push_back(fnhood);
                    frontier_read_reqs.emplace_back(get_node_sector((size_t) id) * defaults::SECTOR_LEN,
                                                    num_sectors_per_node * defaults::SECTOR_LEN, fnhood.second);
                    if (stats != nullptr) {
                        stats->n_4k++;
                        stats->n_ios++;
                    }
                    num_ios++;
                }
                io_timer.reset();
                reader->read(frontier_read_reqs, ctx); // synchronous IO linux
                if (stats != nullptr) {
                    stats->io_us += (float) io_timer.elapsed();
                }
            }

            // process cached nhoods
            for (auto &cached_nhood: cached_nhoods) {
                auto global_cache_iter = _coord_cache.find(cached_nhood.first);
                T *node_fp_coords_copy = global_cache_iter->second;
                float cur_expanded_dist;
                if (!_use_disk_index_pq) {
                    cur_expanded_dist = _dist_cmp->compare(aligned_query_T, node_fp_coords_copy,
                                                           (uint32_t) _aligned_dim);
                } else {
                    if (metric == polaris::MetricType::METRIC_INNER_PRODUCT)
                        cur_expanded_dist = _disk_pq_table.inner_product(query_float, (uint8_t *) node_fp_coords_copy);
                    else
                        cur_expanded_dist = _disk_pq_table.l2_distance( // disk_pq does not support OPQ yet
                                query_float, (uint8_t *) node_fp_coords_copy);
                }
                full_retset.push_back(Neighbor((uint32_t) cached_nhood.first, cur_expanded_dist));

                uint64_t nnbrs = cached_nhood.second.first;
                uint32_t *node_nbrs = cached_nhood.second.second;

                // compute node_nbrs <-> query dists in PQ space
                cpu_timer.reset();
                compute_dists(node_nbrs, nnbrs, dist_scratch);
                if (stats != nullptr) {
                    stats->n_cmps += (uint32_t) nnbrs;
                    stats->cpu_us += (float) cpu_timer.elapsed();
                }

                // process prefetched nhood
                for (uint64_t m = 0; m < nnbrs; ++m) {
                    uint32_t id = node_nbrs[m];
                    if (visited.insert(id).second) {
                        cmps++;
                        float dist = dist_scratch[m];
                        Neighbor nn(id, dist);
                        retset.insert(nn);
                    }
                }
            }
            for (auto &frontier_nhood: frontier_nhoods) {
                char *node_disk_buf = offset_to_node(frontier_nhood.second, frontier_nhood.first);
                uint32_t *node_buf = offset_to_node_nhood(node_disk_buf);
                uint64_t nnbrs = (uint64_t) (*node_buf);
                T *node_fp_coords = offset_to_node_coords(node_disk_buf);
                memcpy(data_buf, node_fp_coords, _disk_bytes_per_point);
                float cur_expanded_dist;
                if (!_use_disk_index_pq) {
                    cur_expanded_dist = _dist_cmp->compare(aligned_query_T, data_buf, (uint32_t) _aligned_dim);
                } else {
                    if (metric == polaris::MetricType::METRIC_INNER_PRODUCT)
                        cur_expanded_dist = _disk_pq_table.inner_product(query_float, (uint8_t *) data_buf);
                    else
                        cur_expanded_dist = _disk_pq_table.l2_distance(query_float, (uint8_t *) data_buf);
                }
                full_retset.push_back(Neighbor(frontier_nhood.first, cur_expanded_dist));
                uint32_t *node_nbrs = (node_buf + 1);
                // compute node_nbrs <-> query dist in PQ space
                cpu_timer.reset();
                compute_dists(node_nbrs, nnbrs, dist_scratch);
                if (stats != nullptr) {
                    stats->n_cmps += (uint32_t) nnbrs;
                    stats->cpu_us += (float) cpu_timer.elapsed();
                }

                cpu_timer.reset();
                // process prefetch-ed nhood
                for (uint64_t m = 0; m < nnbrs; ++m) {
                    uint32_t id = node_nbrs[m];
                    if (visited.insert(id).second) {
                        cmps++;
                        float dist = dist_scratch[m];
                        if (stats != nullptr) {
                            stats->n_cmps++;
                        }

                        Neighbor nn(id, dist);
                        retset.insert(nn);
                    }
                }

                if (stats != nullptr) {
                    stats->cpu_us += (float) cpu_timer.elapsed();
                }
            }

            hops++;
        }

        // re-sort by distance
        std::sort(full_retset.begin(), full_retset.end());

        if (use_reorder_data) {
            if (!(this->_reorder_data_exists)) {
                throw PolarisException("Requested use of reordering data which does "
                                       "not exist in index "
                                       "file",
                                       -1, __PRETTY_FUNCTION__, __FILE__, __LINE__);
            }

            std::vector<AlignedRead> vec_read_reqs;

            if (full_retset.size() > k_search * FULL_PRECISION_REORDER_MULTIPLIER)
                full_retset.erase(full_retset.begin() + k_search * FULL_PRECISION_REORDER_MULTIPLIER,
                                  full_retset.end());

            for (size_t i = 0; i < full_retset.size(); ++i) {
                // MULTISECTORFIX
                vec_read_reqs.emplace_back(VECTOR_SECTOR_NO(((size_t) full_retset[i].id)) * defaults::SECTOR_LEN,
                                           defaults::SECTOR_LEN, sector_scratch + i * defaults::SECTOR_LEN);

                if (stats != nullptr) {
                    stats->n_4k++;
                    stats->n_ios++;
                }
            }

            io_timer.reset();
            reader->read(vec_read_reqs, ctx); // synchronous IO linux
            if (stats != nullptr) {
                stats->io_us += io_timer.elapsed();
            }

            for (size_t i = 0; i < full_retset.size(); ++i) {
                auto id = full_retset[i].id;
                // MULTISECTORFIX
                auto location = (sector_scratch + i * defaults::SECTOR_LEN) + VECTOR_SECTOR_OFFSET(id);
                full_retset[i].distance = _dist_cmp->compare(aligned_query_T, (T *) location,
                                                             (uint32_t) this->_data_dim);
            }

            std::sort(full_retset.begin(), full_retset.end());
        }

        // copy k_search values
        for (uint64_t i = 0; i < k_search; i++) {
            indices[i] = full_retset[i].id;
            if (distances != nullptr) {
                distances[i] = full_retset[i].distance;
                if (metric == polaris::MetricType::METRIC_INNER_PRODUCT) {
                    // flip the sign to convert min to max
                    distances[i] = (-distances[i]);
                    // rescale to revert back to original norms (cancelling the
                    // effect of base and query pre-processing)
                    if (_max_base_norm != 0)
                        distances[i] *= (_max_base_norm * query_norm);
                }
            }
        }

        if (stats != nullptr) {
            stats->total_us = (float) query_timer.elapsed();
        }
    }


    // range search returns results of all neighbors within distance of range.
    // indices and distances need to be pre-allocated of size l_search and the
    // return value is the number of matching hits.
    template<typename T>
    uint32_t PQFlashIndex<T>::range_search(const T *query1, const double range, const uint64_t min_l_search,
                                           const uint64_t max_l_search, std::vector<uint64_t> &indices,
                                           std::vector<float> &distances, const uint64_t min_beam_width,
                                           QueryStats *stats) {
        uint32_t res_count = 0;

        bool stop_flag = false;

        uint32_t l_search = (uint32_t) min_l_search; // starting size of the candidate list
        while (!stop_flag) {
            indices.resize(l_search);
            distances.resize(l_search);
            uint64_t cur_bw = min_beam_width > (l_search / 5) ? min_beam_width : l_search / 5;
            cur_bw = (cur_bw > 100) ? 100 : cur_bw;
            for (auto &x: distances)
                x = std::numeric_limits<float>::max();
            this->cached_beam_search(query1, l_search, l_search, indices.data(), distances.data(), cur_bw, false,
                                     stats);
            for (uint32_t i = 0; i < l_search; i++) {
                if (distances[i] > (float) range) {
                    res_count = i;
                    break;
                } else if (i == l_search - 1)
                    res_count = l_search;
            }
            if (res_count < (uint32_t) (l_search / 2.0))
                stop_flag = true;
            l_search = l_search * 2;
            if (l_search > max_l_search)
                stop_flag = true;
        }
        indices.resize(res_count);
        distances.resize(res_count);
        return res_count;
    }

    template<typename T>
    uint64_t PQFlashIndex<T>::get_data_dim() {
        return _data_dim;
    }

    template<typename T>
    uint32_t PQFlashIndex<T>::optimize_beamwidth(T *tuning_sample,
                                                 uint64_t tuning_sample_num, uint64_t tuning_sample_aligned_dim,
                                                 uint32_t L,
                                                 uint32_t nthreads, uint32_t start_bw) {
        uint32_t cur_bw = start_bw;
        double max_qps = 0;
        uint32_t best_bw = start_bw;
        bool stop_flag = false;

        while (!stop_flag) {
            std::vector<uint64_t> tuning_sample_result_ids_64(tuning_sample_num, 0);
            std::vector<float> tuning_sample_result_dists(tuning_sample_num, 0);
            polaris::QueryStats *stats = new polaris::QueryStats[tuning_sample_num];

            auto s = std::chrono::high_resolution_clock::now();
#pragma omp parallel for schedule(dynamic, 1) num_threads(nthreads)
            for (int64_t i = 0; i < (int64_t) tuning_sample_num; i++) {
                cached_beam_search(tuning_sample + (i * tuning_sample_aligned_dim), 1, L,
                                   tuning_sample_result_ids_64.data() + (i * 1),
                                   tuning_sample_result_dists.data() + (i * 1), cur_bw, false, stats + i);
            }
            auto e = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> diff = e - s;
            double qps = (1.0f * (float) tuning_sample_num) / (1.0f * (float) diff.count());

            double lat_999 = polaris::get_percentile_stats<float>(
                    stats, tuning_sample_num, 0.999f, [](const polaris::QueryStats &stats) { return stats.total_us; });

            double mean_latency = polaris::get_mean_stats<float>(
                    stats, tuning_sample_num, [](const polaris::QueryStats &stats) { return stats.total_us; });

            if (qps > max_qps && lat_999 < (15000) + mean_latency * 2) {
                max_qps = qps;
                best_bw = cur_bw;
                cur_bw = (uint32_t) (std::ceil)((float) cur_bw * 1.1f);
            } else {
                stop_flag = true;
            }
            if (cur_bw > 64)
                stop_flag = true;

            delete[] stats;
        }
        return best_bw;
    }

    template<typename T>
    polaris::MetricType PQFlashIndex<T>::get_metric() {
        return this->metric;
    }


    template<typename T>
    std::vector<std::uint8_t> PQFlashIndex<T>::get_pq_vector(std::uint64_t vid) {
        std::uint8_t *pqVec = &this->data[vid * this->_n_chunks];
        return std::vector<std::uint8_t>(pqVec, pqVec + this->_n_chunks);
    }

    template<typename T>
    std::uint64_t PQFlashIndex<T>::get_num_points() {
        return _num_points;
    }

    // instantiations
    template
    class PQFlashIndex<uint8_t>;

    template
    class PQFlashIndex<int8_t>;

    template
    class PQFlashIndex<float>;

} // namespace polaris
