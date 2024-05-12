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

#if defined(DISKANN_RELEASE_UNUSED_TCMALLOC_MEMORY_AT_CHECKPOINTS) && defined(DISKANN_BUILD)
#include <gperftools/malloc_extension.h>
#endif

#include <polaris/graph/vamana/logger.h>
#include <polaris/graph/vamana/disk_utils.h>
#include <polaris/io/cached_io.h>
#include <polaris/graph/vamana/index.h>
#include <polaris/core/log.h>
#include <mkl.h>
#include <omp.h>
#include <polaris/core/percentile_stats.h>
#include <polaris/graph/vamana/partition.h>
#include <polaris/graph/vamana/pq_flash_index.h>
#include <polaris/graph/vamana/timer.h>
#include <polaris/datasets/bin.h>
#include <turbo/container/flat_hash_set.h>

namespace polaris {

    void add_new_file_to_single_index(std::string index_file, std::string new_file) {
        std::unique_ptr<uint64_t[]> metadata;
        uint64_t nr, nc;
        polaris::load_bin<uint64_t>(index_file, metadata, nr, nc);
        if (nc != 1) {
            std::stringstream stream;
            stream << "Error, index file specified does not have correct metadata. " << std::endl;
            throw polaris::PolarisException(stream.str(), -1);
        }
        size_t index_ending_offset = metadata[nr - 1];
        size_t read_blk_size = 64 * 1024 * 1024;
        cached_ofstream writer(index_file, read_blk_size);
        size_t check_file_size = collie::filesystem::file_size(index_file);
        if (check_file_size != index_ending_offset) {
            std::stringstream stream;
            stream << "Error, index file specified does not have correct metadata "
                      "(last entry must match the filesize). "
                   << std::endl;
            throw polaris::PolarisException(stream.str(), -1);
        }

        cached_ifstream reader(new_file, read_blk_size);
        size_t fsize = reader.get_file_size();
        if (fsize == 0) {
            std::stringstream stream;
            stream << "Error, new file specified is empty. Not appending. " << std::endl;
            throw polaris::PolarisException(stream.str(), -1);
        }

        size_t num_blocks = DIV_ROUND_UP(fsize, read_blk_size);
        char *dump = new char[read_blk_size];
        for (uint64_t i = 0; i < num_blocks; i++) {
            size_t cur_block_size =
                    read_blk_size > fsize - (i * read_blk_size) ? fsize - (i * read_blk_size) : read_blk_size;
            reader.read(dump, cur_block_size);
            writer.write(dump, cur_block_size);
        }
        //    reader.close();
        //    writer.close();

        delete[] dump;
        std::vector<uint64_t> new_meta;
        for (uint64_t i = 0; i < nr; i++)
            new_meta.push_back(metadata[i]);
        new_meta.push_back(metadata[nr - 1] + fsize);

        polaris::save_bin<uint64_t>(index_file, new_meta.data(), new_meta.size(), 1);
    }

    double get_memory_budget(double search_ram_budget) {
        double final_index_ram_limit = search_ram_budget;
        if (search_ram_budget - SPACE_FOR_CACHED_NODES_IN_GB >
            THRESHOLD_FOR_CACHING_IN_GB) { // slack for space used by cached
            // nodes
            final_index_ram_limit = search_ram_budget - SPACE_FOR_CACHED_NODES_IN_GB;
        }
        return final_index_ram_limit * 1024 * 1024 * 1024;
    }

    double get_memory_budget(const std::string &mem_budget_str) {
        double search_ram_budget = atof(mem_budget_str.c_str());
        return get_memory_budget(search_ram_budget);
    }

    size_t calculate_num_pq_chunks(double final_index_ram_limit, size_t points_num, uint32_t dim,
                                   const std::vector<std::string> &param_list) {
        size_t num_pq_chunks = (size_t) (std::floor)(uint64_t(final_index_ram_limit / (double) points_num));
        polaris::cout << "Calculated num_pq_chunks :" << num_pq_chunks << std::endl;
        if (param_list.size() >= 6) {
            float compress_ratio = (float) atof(param_list[5].c_str());
            if (compress_ratio > 0 && compress_ratio <= 1) {
                size_t chunks_by_cr = (size_t) (std::floor)(compress_ratio * dim);

                if (chunks_by_cr > 0 && chunks_by_cr < num_pq_chunks) {
                    polaris::cout << "Compress ratio:" << compress_ratio << " new #pq_chunks:" << chunks_by_cr
                                  << std::endl;
                    num_pq_chunks = chunks_by_cr;
                } else {
                    polaris::cout << "Compress ratio: " << compress_ratio << " #new pq_chunks: " << chunks_by_cr
                                  << " is either zero or greater than num_pq_chunks: " << num_pq_chunks
                                  << ". num_pq_chunks is unchanged. " << std::endl;
                }
            } else {
                polaris::cerr << "Compression ratio: " << compress_ratio << " should be in (0,1]" << std::endl;
            }
        }

        num_pq_chunks = num_pq_chunks <= 0 ? 1 : num_pq_chunks;
        num_pq_chunks = num_pq_chunks > dim ? dim : num_pq_chunks;
        num_pq_chunks = num_pq_chunks > MAX_PQ_CHUNKS ? MAX_PQ_CHUNKS : num_pq_chunks;

        polaris::cout << "Compressing " << dim << "-dimensional data into " << num_pq_chunks << " bytes per vector."
                      << std::endl;
        return num_pq_chunks;
    }

    template<typename T>
    T *generateRandomWarmup(uint64_t warmup_num, uint64_t warmup_dim, uint64_t warmup_aligned_dim) {
        T *warmup = nullptr;
        warmup_num = 100000;
        polaris::cout << "Generating random warmup file with dim " << warmup_dim << " and aligned dim "
                      << warmup_aligned_dim << std::flush;
        polaris::alloc_aligned(((void **) &warmup), warmup_num * warmup_aligned_dim * sizeof(T), 8 * sizeof(T));
        std::memset(warmup, 0, warmup_num * warmup_aligned_dim * sizeof(T));
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> dis(-128, 127);
        for (uint32_t i = 0; i < warmup_num; i++) {
            for (uint32_t d = 0; d < warmup_dim; d++) {
                warmup[i * warmup_aligned_dim + d] = (T) dis(gen);
            }
        }
        polaris::cout << "..done" << std::endl;
        return warmup;
    }


    template<typename T>
    T *load_warmup(const std::string &cache_warmup_file, uint64_t &warmup_num, uint64_t warmup_dim,
                   uint64_t warmup_aligned_dim) {
        T *warmup = nullptr;
        uint64_t file_dim, file_aligned_dim;

        std::error_code ec;
        if (collie::filesystem::exists(cache_warmup_file, ec)) {
            polaris::load_aligned_bin<T>(cache_warmup_file, warmup, warmup_num, file_dim, file_aligned_dim);
            if (file_dim != warmup_dim || file_aligned_dim != warmup_aligned_dim) {
                std::stringstream stream;
                stream << "Mismatched dimensions in sample file. file_dim = " << file_dim
                       << " file_aligned_dim: " << file_aligned_dim << " index_dim: " << warmup_dim
                       << " index_aligned_dim: " << warmup_aligned_dim << std::endl;
                throw polaris::PolarisException(stream.str(), -1);
            }
        } else {
            warmup = generateRandomWarmup<T>(warmup_num, warmup_dim, warmup_aligned_dim);
        }
        return warmup;
    }

/***************************************************
    Support for Merging Many Vamana Indices
 ***************************************************/

    void read_idmap(const std::string &fname, std::vector<uint32_t> &ivecs) {
        uint32_t npts32, dim;
        size_t actual_file_size = collie::filesystem::file_size(fname);
        std::ifstream reader(fname.c_str(), std::ios::binary);
        reader.read((char *) &npts32, sizeof(uint32_t));
        reader.read((char *) &dim, sizeof(uint32_t));
        if (dim != 1 || actual_file_size != ((size_t) npts32) * sizeof(uint32_t) + 2 * sizeof(uint32_t)) {
            std::stringstream stream;
            stream << "Error reading idmap file. Check if the file is bin file with "
                      "1 dimensional data. Actual: "
                   << actual_file_size << ", expected: " << (size_t) npts32 + 2 * sizeof(uint32_t) << std::endl;

            throw polaris::PolarisException(stream.str(), -1, __PRETTY_FUNCTION__, __FILE__, __LINE__);
        }
        ivecs.resize(npts32);
        reader.read((char *) ivecs.data(), ((size_t) npts32) * sizeof(uint32_t));
        reader.close();
    }

    int
    merge_shards(const std::string &vamana_prefix, const std::string &vamana_suffix, const std::string &idmaps_prefix,
                 const std::string &idmaps_suffix, const uint64_t nshards, uint32_t max_degree,
                 const std::string &output_vamana, const std::string &medoids_file) {
        // Read ID maps
        std::vector<std::string> vamana_names(nshards);
        std::vector<std::vector<uint32_t>> idmaps(nshards);
        for (uint64_t shard = 0; shard < nshards; shard++) {
            vamana_names[shard] = vamana_prefix + std::to_string(shard) + vamana_suffix;
            read_idmap(idmaps_prefix + std::to_string(shard) + idmaps_suffix, idmaps[shard]);
        }

        // find max node id
        size_t nnodes = 0;
        size_t nelems = 0;
        for (auto &idmap: idmaps) {
            for (auto &id: idmap) {
                nnodes = std::max(nnodes, (size_t) id);
            }
            nelems += idmap.size();
        }
        nnodes++;
        polaris::cout << "# nodes: " << nnodes << ", max. degree: " << max_degree << std::endl;

        // compute inverse map: node -> shards
        std::vector<std::pair<uint32_t, uint32_t>> node_shard;
        node_shard.reserve(nelems);
        for (size_t shard = 0; shard < nshards; shard++) {
            polaris::cout << "Creating inverse map -- shard #" << shard << std::endl;
            for (size_t idx = 0; idx < idmaps[shard].size(); idx++) {
                size_t node_id = idmaps[shard][idx];
                node_shard.push_back(std::make_pair((uint32_t) node_id, (uint32_t) shard));
            }
        }
        std::sort(node_shard.begin(), node_shard.end(), [](const auto &left, const auto &right) {
            return left.first < right.first || (left.first == right.first && left.second < right.second);
        });
        polaris::cout << "Finished computing node -> shards map" << std::endl;

        // create cached vamana readers
        std::vector<cached_ifstream> vamana_readers(nshards);
        for (size_t i = 0; i < nshards; i++) {
            vamana_readers[i].open(vamana_names[i], BUFFER_SIZE_FOR_CACHED_IO);
            size_t expected_file_size;
            vamana_readers[i].read((char *) &expected_file_size, sizeof(uint64_t));
        }

        size_t vamana_metadata_size =
                sizeof(uint64_t) + sizeof(uint32_t) + sizeof(uint32_t) +
                sizeof(uint64_t); // expected file size + max degree +
        // medoid_id + frozen_point info

        // create cached vamana writers
        cached_ofstream merged_vamana_writer(output_vamana, BUFFER_SIZE_FOR_CACHED_IO);

        size_t merged_index_size = vamana_metadata_size; // we initialize the size of the merged index to
        // the metadata size
        size_t merged_index_frozen = 0;
        merged_vamana_writer.write((char *) &merged_index_size,
                                   sizeof(uint64_t)); // we will overwrite the index size at the end

        uint32_t output_width = max_degree;
        uint32_t max_input_width = 0;
        // read width from each vamana to advance buffer by sizeof(uint32_t) bytes
        for (auto &reader: vamana_readers) {
            uint32_t input_width;
            reader.read((char *) &input_width, sizeof(uint32_t));
            max_input_width = input_width > max_input_width ? input_width : max_input_width;
        }

        polaris::cout << "Max input width: " << max_input_width << ", output width: " << output_width << std::endl;

        merged_vamana_writer.write((char *) &output_width, sizeof(uint32_t));
        std::ofstream medoid_writer(medoids_file.c_str(), std::ios::binary);
        uint32_t nshards_u32 = (uint32_t) nshards;
        uint32_t one_val = 1;
        medoid_writer.write((char *) &nshards_u32, sizeof(uint32_t));
        medoid_writer.write((char *) &one_val, sizeof(uint32_t));

        uint64_t vamana_index_frozen = 0; // as of now the functionality to merge many overlapping vamana
        // indices is supported only for bulk indices without frozen point.
        // Hence the final index will also not have any frozen points.
        for (uint64_t shard = 0; shard < nshards; shard++) {
            uint32_t medoid;
            // read medoid
            vamana_readers[shard].read((char *) &medoid, sizeof(uint32_t));
            vamana_readers[shard].read((char *) &vamana_index_frozen, sizeof(uint64_t));
            assert(vamana_index_frozen == false);
            // rename medoid
            medoid = idmaps[shard][medoid];

            medoid_writer.write((char *) &medoid, sizeof(uint32_t));
            // write renamed medoid
            if (shard == (nshards - 1)) //--> uncomment if running hierarchical
                merged_vamana_writer.write((char *) &medoid, sizeof(uint32_t));
        }
        merged_vamana_writer.write((char *) &merged_index_frozen, sizeof(uint64_t));
        medoid_writer.close();

        polaris::cout << "Starting merge" << std::endl;

        // Gopal. random_shuffle() is deprecated.
        std::random_device rng;
        std::mt19937 urng(rng());

        std::vector<bool> nhood_set(nnodes, 0);
        std::vector<uint32_t> final_nhood;

        uint32_t nnbrs = 0, shard_nnbrs = 0;
        uint32_t cur_id = 0;
        for (const auto &id_shard: node_shard) {
            uint32_t node_id = id_shard.first;
            uint32_t shard_id = id_shard.second;
            if (cur_id < node_id) {
                // Gopal. random_shuffle() is deprecated.
                std::shuffle(final_nhood.begin(), final_nhood.end(), urng);
                nnbrs = (uint32_t) (std::min)(final_nhood.size(), (uint64_t) max_degree);
                // write into merged ofstream
                merged_vamana_writer.write((char *) &nnbrs, sizeof(uint32_t));
                merged_vamana_writer.write((char *) final_nhood.data(), nnbrs * sizeof(uint32_t));
                merged_index_size += (sizeof(uint32_t) + nnbrs * sizeof(uint32_t));
                if (cur_id % 499999 == 1) {
                    polaris::cout << "." << std::flush;
                }
                cur_id = node_id;
                nnbrs = 0;
                for (auto &p: final_nhood)
                    nhood_set[p] = 0;
                final_nhood.clear();
            }
            // read from shard_id ifstream
            vamana_readers[shard_id].read((char *) &shard_nnbrs, sizeof(uint32_t));

            if (shard_nnbrs == 0) {
                polaris::cout << "WARNING: shard #" << shard_id << ", node_id " << node_id << " has 0 nbrs"
                              << std::endl;
            }

            std::vector<uint32_t> shard_nhood(shard_nnbrs);
            if (shard_nnbrs > 0)
                vamana_readers[shard_id].read((char *) shard_nhood.data(), shard_nnbrs * sizeof(uint32_t));
            // rename nodes
            for (uint64_t j = 0; j < shard_nnbrs; j++) {
                if (nhood_set[idmaps[shard_id][shard_nhood[j]]] == 0) {
                    nhood_set[idmaps[shard_id][shard_nhood[j]]] = 1;
                    final_nhood.emplace_back(idmaps[shard_id][shard_nhood[j]]);
                }
            }
        }

        // Gopal. random_shuffle() is deprecated.
        std::shuffle(final_nhood.begin(), final_nhood.end(), urng);
        nnbrs = (uint32_t) (std::min)(final_nhood.size(), (uint64_t) max_degree);
        // write into merged ofstream
        merged_vamana_writer.write((char *) &nnbrs, sizeof(uint32_t));
        if (nnbrs > 0) {
            merged_vamana_writer.write((char *) final_nhood.data(), nnbrs * sizeof(uint32_t));
        }
        merged_index_size += (sizeof(uint32_t) + nnbrs * sizeof(uint32_t));
        for (auto &p: final_nhood)
            nhood_set[p] = 0;
        final_nhood.clear();

        polaris::cout << "Expected size: " << merged_index_size << std::endl;

        merged_vamana_writer.reset();
        merged_vamana_writer.write((char *) &merged_index_size, sizeof(uint64_t));

        polaris::cout << "Finished merge" << std::endl;
        return 0;
    }

    template<typename T>
    int build_merged_vamana_index(std::string base_file, polaris::MetricType compareMetric, uint32_t L, uint32_t R,
                                  double sampling_rate, double ram_budget, std::string mem_index_path,
                                  std::string medoids_file, std::string centroids_file, size_t build_pq_bytes,
                                  bool use_opq,
                                  uint32_t num_threads) {
        size_t base_num, base_dim;
        polaris::get_bin_metadata(base_file, base_num, base_dim);

        double full_index_ram = estimate_ram_usage(base_num, (uint32_t) base_dim, sizeof(T), R);

        // TODO: Make this honest when there is filter support
        if (full_index_ram < ram_budget * 1024 * 1024 * 1024) {
            polaris::cout << "Full index fits in RAM budget, should consume at most "
                          << full_index_ram / (1024 * 1024 * 1024) << "GiBs, so building in one shot" << std::endl;

            polaris::IndexWriteParameters paras = polaris::IndexWriteParametersBuilder(L, R)
                    .with_saturate_graph(true)
                    .with_num_threads(num_threads)
                    .build();
            polaris::VamanaIndex<T> _index(compareMetric, base_dim, base_num,
                                           std::make_shared<polaris::IndexWriteParameters>(paras), nullptr,
                                           defaults::NUM_FROZEN_POINTS_STATIC, false, false,
                                           build_pq_bytes > 0, build_pq_bytes, use_opq);
            _index.build(base_file.c_str(), base_num);
            _index.save(mem_index_path.c_str());
            std::remove(medoids_file.c_str());
            std::remove(centroids_file.c_str());
            return 0;
        }

        std::string merged_index_prefix = mem_index_path + "_tempFiles";

        Timer timer;
        int num_parts =
                partition_with_ram_budget<T>(base_file, sampling_rate, ram_budget, 2 * R / 3, merged_index_prefix, 2);
        polaris::cout << timer.elapsed_seconds_for_step("partitioning data ") << std::endl;

        std::string cur_centroid_filepath = merged_index_prefix + "_centroids.bin";
        std::rename(cur_centroid_filepath.c_str(), centroids_file.c_str());

        timer.reset();
        for (int p = 0; p < num_parts; p++) {
#if defined(DISKANN_RELEASE_UNUSED_TCMALLOC_MEMORY_AT_CHECKPOINTS) && defined(DISKANN_BUILD)
            MallocExtension::instance()->ReleaseFreeMemory();
#endif

            std::string shard_base_file = merged_index_prefix + "_subshard-" + std::to_string(p) + ".bin";

            std::string shard_ids_file = merged_index_prefix + "_subshard-" + std::to_string(p) + "_ids_uint32.bin";

            retrieve_shard_data_from_ids<T>(base_file, shard_ids_file, shard_base_file);

            std::string shard_index_file = merged_index_prefix + "_subshard-" + std::to_string(p) + "_mem.index";

            polaris::IndexWriteParameters low_degree_params = polaris::IndexWriteParametersBuilder(L, 2 * R / 3)
                    .with_saturate_graph(false)
                    .with_num_threads(num_threads)
                    .build();

            uint64_t shard_base_dim, shard_base_pts;
            get_bin_metadata(shard_base_file, shard_base_pts, shard_base_dim);

            polaris::VamanaIndex<T> _index(compareMetric, shard_base_dim, shard_base_pts,
                                           std::make_shared<polaris::IndexWriteParameters>(low_degree_params), nullptr,
                                           defaults::NUM_FROZEN_POINTS_STATIC, false, false, build_pq_bytes > 0,
                                           build_pq_bytes, use_opq);
            _index.build(shard_base_file.c_str(), shard_base_pts);
            _index.save(shard_index_file.c_str());
            std::remove(shard_base_file.c_str());
        }
        polaris::cout << timer.elapsed_seconds_for_step("building indices on shards") << std::endl;

        timer.reset();
        polaris::merge_shards(merged_index_prefix + "_subshard-", "_mem.index", merged_index_prefix + "_subshard-",
                              "_ids_uint32.bin", num_parts, R, mem_index_path, medoids_file);
        polaris::cout << timer.elapsed_seconds_for_step("merging indices") << std::endl;

        // delete tempFiles
        for (int p = 0; p < num_parts; p++) {
            std::string shard_base_file = merged_index_prefix + "_subshard-" + std::to_string(p) + ".bin";
            std::string shard_id_file = merged_index_prefix + "_subshard-" + std::to_string(p) + "_ids_uint32.bin";
            std::string shard_index_file = merged_index_prefix + "_subshard-" + std::to_string(p) + "_mem.index";
            std::string shard_index_file_data = shard_index_file + ".data";

            std::remove(shard_base_file.c_str());
            std::remove(shard_id_file.c_str());
            std::remove(shard_index_file.c_str());
            std::remove(shard_index_file_data.c_str());
        }
        return 0;
    }

    template<typename T>
    void  create_disk_layout(const std::string base_file, const std::string mem_index_file, const std::string output_file,
                       const std::string reorder_data_file) {
        uint32_t npts, ndims;

        // amount to read or write in one shot
        size_t read_blk_size = 64 * 1024 * 1024;
        size_t write_blk_size = read_blk_size;
        cached_ifstream base_reader(base_file, read_blk_size);
        base_reader.read((char *) &npts, sizeof(uint32_t));
        base_reader.read((char *) &ndims, sizeof(uint32_t));

        size_t npts_64, ndims_64;
        npts_64 = npts;
        ndims_64 = ndims;

        // Check if we need to append data for re-ordering
        bool append_reorder_data = false;
        std::ifstream reorder_data_reader;

        uint32_t npts_reorder_file = 0, ndims_reorder_file = 0;
        if (reorder_data_file != std::string("")) {
            append_reorder_data = true;
            size_t reorder_data_file_size = collie::filesystem::file_size(reorder_data_file);
            reorder_data_reader.exceptions(std::ofstream::failbit | std::ofstream::badbit);

            try {
                reorder_data_reader.open(reorder_data_file, std::ios::binary);
                reorder_data_reader.read((char *) &npts_reorder_file, sizeof(uint32_t));
                reorder_data_reader.read((char *) &ndims_reorder_file, sizeof(uint32_t));
                if (npts_reorder_file != npts)
                    throw PolarisException("Mismatch in num_points between reorder "
                                           "data file and base file",
                                           -1, __PRETTY_FUNCTION__, __FILE__, __LINE__);
                if (reorder_data_file_size !=
                    8 + sizeof(float) * (size_t) npts_reorder_file * (size_t) ndims_reorder_file)
                    throw PolarisException("Discrepancy in reorder data file size ", -1, __PRETTY_FUNCTION__, __FILE__,
                                           __LINE__);
            }
            catch (std::system_error &e) {
                throw FileException(reorder_data_file, e, __PRETTY_FUNCTION__, __FILE__, __LINE__);
            }
        }

        // create cached reader + writer
        size_t actual_file_size = collie::filesystem::file_size(mem_index_file);
        polaris::cout << "Vamana index file size=" << actual_file_size << std::endl;
        std::ifstream vamana_reader(mem_index_file, std::ios::binary);
        cached_ofstream diskann_writer(output_file, write_blk_size);

        // metadata: width, medoid
        uint32_t width_u32, medoid_u32;
        size_t index_file_size;

        vamana_reader.read((char *) &index_file_size, sizeof(uint64_t));
        if (index_file_size != actual_file_size) {
            std::stringstream stream;
            stream << "VamanaIndex file size does not match expected size per "
                      "meta-data."
                   << " file size from file: " << index_file_size << " actual file size: " << actual_file_size
                   << std::endl;

            throw polaris::PolarisException(stream.str(), -1, __PRETTY_FUNCTION__, __FILE__, __LINE__);
        }
        uint64_t vamana_frozen_num = false, vamana_frozen_loc = 0;

        vamana_reader.read((char *) &width_u32, sizeof(uint32_t));
        vamana_reader.read((char *) &medoid_u32, sizeof(uint32_t));
        vamana_reader.read((char *) &vamana_frozen_num, sizeof(uint64_t));
        // compute
        uint64_t medoid, max_node_len, nnodes_per_sector;
        npts_64 = (uint64_t) npts;
        medoid = (uint64_t) medoid_u32;
        if (vamana_frozen_num == 1)
            vamana_frozen_loc = medoid;
        max_node_len = (((uint64_t) width_u32 + 1) * sizeof(uint32_t)) + (ndims_64 * sizeof(T));
        nnodes_per_sector = defaults::SECTOR_LEN / max_node_len; // 0 if max_node_len > SECTOR_LEN

        polaris::cout << "medoid: " << medoid << "B" << std::endl;
        polaris::cout << "max_node_len: " << max_node_len << "B" << std::endl;
        polaris::cout << "nnodes_per_sector: " << nnodes_per_sector << "B" << std::endl;

        // defaults::SECTOR_LEN buffer for each sector
        std::unique_ptr<char[]> sector_buf = std::make_unique<char[]>(defaults::SECTOR_LEN);
        std::unique_ptr<char[]> multisector_buf = std::make_unique<char[]>(
                ROUND_UP(max_node_len, defaults::SECTOR_LEN));
        std::unique_ptr<char[]> node_buf = std::make_unique<char[]>(max_node_len);
        uint32_t &nnbrs = *(uint32_t *) (node_buf.get() + ndims_64 * sizeof(T));
        uint32_t *nhood_buf = (uint32_t *) (node_buf.get() + (ndims_64 * sizeof(T)) + sizeof(uint32_t));

        // number of sectors (1 for meta data)
        uint64_t n_sectors = nnodes_per_sector > 0 ? ROUND_UP(npts_64, nnodes_per_sector) / nnodes_per_sector
                                                   : npts_64 * DIV_ROUND_UP(max_node_len, defaults::SECTOR_LEN);
        uint64_t n_reorder_sectors = 0;
        uint64_t n_data_nodes_per_sector = 0;

        if (append_reorder_data) {
            n_data_nodes_per_sector = defaults::SECTOR_LEN / (ndims_reorder_file * sizeof(float));
            n_reorder_sectors = ROUND_UP(npts_64, n_data_nodes_per_sector) / n_data_nodes_per_sector;
        }
        uint64_t disk_index_file_size = (n_sectors + n_reorder_sectors + 1) * defaults::SECTOR_LEN;

        std::vector<uint64_t> output_file_meta;
        output_file_meta.push_back(npts_64);
        output_file_meta.push_back(ndims_64);
        output_file_meta.push_back(medoid);
        output_file_meta.push_back(max_node_len);
        output_file_meta.push_back(nnodes_per_sector);
        output_file_meta.push_back(vamana_frozen_num);
        output_file_meta.push_back(vamana_frozen_loc);
        output_file_meta.push_back((uint64_t) append_reorder_data);
        if (append_reorder_data) {
            output_file_meta.push_back(n_sectors + 1);
            output_file_meta.push_back(ndims_reorder_file);
            output_file_meta.push_back(n_data_nodes_per_sector);
        }
        output_file_meta.push_back(disk_index_file_size);

        diskann_writer.write(sector_buf.get(), defaults::SECTOR_LEN);

        std::unique_ptr<T[]> cur_node_coords = std::make_unique<T[]>(ndims_64);
        polaris::cout << "# sectors: " << n_sectors << std::endl;
        uint64_t cur_node_id = 0;

        if (nnodes_per_sector > 0) { // Write multiple nodes per sector
            for (uint64_t sector = 0; sector < n_sectors; sector++) {
                if (sector % 100000 == 0) {
                    polaris::cout << "Sector #" << sector << "written" << std::endl;
                }
                memset(sector_buf.get(), 0, defaults::SECTOR_LEN);
                for (uint64_t sector_node_id = 0; sector_node_id < nnodes_per_sector && cur_node_id < npts_64;
                     sector_node_id++) {
                    memset(node_buf.get(), 0, max_node_len);
                    // read cur node's nnbrs
                    vamana_reader.read((char *) &nnbrs, sizeof(uint32_t));

                    // sanity checks on nnbrs
                    assert(nnbrs > 0);
                    assert(nnbrs <= width_u32);

                    // read node's nhood
                    vamana_reader.read((char *) nhood_buf, (std::min)(nnbrs, width_u32) * sizeof(uint32_t));
                    if (nnbrs > width_u32) {
                        vamana_reader.seekg((nnbrs - width_u32) * sizeof(uint32_t), vamana_reader.cur);
                    }

                    // write coords of node first
                    //  T *node_coords = data + ((uint64_t) ndims_64 * cur_node_id);
                    base_reader.read((char *) cur_node_coords.get(), sizeof(T) * ndims_64);
                    memcpy(node_buf.get(), cur_node_coords.get(), ndims_64 * sizeof(T));

                    // write nnbrs
                    *(uint32_t *) (node_buf.get() + ndims_64 * sizeof(T)) = (std::min)(nnbrs, width_u32);

                    // write nhood next
                    memcpy(node_buf.get() + ndims_64 * sizeof(T) + sizeof(uint32_t), nhood_buf,
                           (std::min)(nnbrs, width_u32) * sizeof(uint32_t));

                    // get offset into sector_buf
                    char *sector_node_buf = sector_buf.get() + (sector_node_id * max_node_len);

                    // copy node buf into sector_node_buf
                    memcpy(sector_node_buf, node_buf.get(), max_node_len);
                    cur_node_id++;
                }
                // flush sector to disk
                diskann_writer.write(sector_buf.get(), defaults::SECTOR_LEN);
            }
        } else { // Write multi-sector nodes
            uint64_t nsectors_per_node = DIV_ROUND_UP(max_node_len, defaults::SECTOR_LEN);
            for (uint64_t i = 0; i < npts_64; i++) {
                if ((i * nsectors_per_node) % 100000 == 0) {
                    polaris::cout << "Sector #" << i * nsectors_per_node << "written" << std::endl;
                }
                memset(multisector_buf.get(), 0, nsectors_per_node * defaults::SECTOR_LEN);

                memset(node_buf.get(), 0, max_node_len);
                // read cur node's nnbrs
                vamana_reader.read((char *) &nnbrs, sizeof(uint32_t));

                // sanity checks on nnbrs
                assert(nnbrs > 0);
                assert(nnbrs <= width_u32);

                // read node's nhood
                vamana_reader.read((char *) nhood_buf, (std::min)(nnbrs, width_u32) * sizeof(uint32_t));
                if (nnbrs > width_u32) {
                    vamana_reader.seekg((nnbrs - width_u32) * sizeof(uint32_t), vamana_reader.cur);
                }

                // write coords of node first
                //  T *node_coords = data + ((uint64_t) ndims_64 * cur_node_id);
                base_reader.read((char *) cur_node_coords.get(), sizeof(T) * ndims_64);
                memcpy(multisector_buf.get(), cur_node_coords.get(), ndims_64 * sizeof(T));

                // write nnbrs
                *(uint32_t *) (multisector_buf.get() + ndims_64 * sizeof(T)) = (std::min)(nnbrs, width_u32);

                // write nhood next
                memcpy(multisector_buf.get() + ndims_64 * sizeof(T) + sizeof(uint32_t), nhood_buf,
                       (std::min)(nnbrs, width_u32) * sizeof(uint32_t));

                // flush sector to disk
                diskann_writer.write(multisector_buf.get(), nsectors_per_node * defaults::SECTOR_LEN);
            }
        }

        if (append_reorder_data) {
            polaris::cout << "VamanaIndex written. Appending reorder data..." << std::endl;

            auto vec_len = ndims_reorder_file * sizeof(float);
            std::unique_ptr<char[]> vec_buf = std::make_unique<char[]>(vec_len);

            for (uint64_t sector = 0; sector < n_reorder_sectors; sector++) {
                if (sector % 100000 == 0) {
                    polaris::cout << "Reorder data Sector #" << sector << "written" << std::endl;
                }

                memset(sector_buf.get(), 0, defaults::SECTOR_LEN);

                for (uint64_t sector_node_id = 0; sector_node_id < n_data_nodes_per_sector && sector_node_id < npts_64;
                     sector_node_id++) {
                    memset(vec_buf.get(), 0, vec_len);
                    reorder_data_reader.read(vec_buf.get(), vec_len);

                    // copy node buf into sector_node_buf
                    memcpy(sector_buf.get() + (sector_node_id * vec_len), vec_buf.get(), vec_len);
                }
                // flush sector to disk
                diskann_writer.write(sector_buf.get(), defaults::SECTOR_LEN);
            }
        }
        diskann_writer.close();
        polaris::save_bin<uint64_t>(output_file, output_file_meta.data(), output_file_meta.size(), 1, 0);
        polaris::cout << "Output disk index file written to " << output_file << std::endl;
    }

    template POLARIS_API void create_disk_layout<int8_t>(const std::string base_file,
                                                         const std::string mem_index_file,
                                                         const std::string output_file,
                                                         const std::string reorder_data_file);

    template POLARIS_API void create_disk_layout<uint8_t>(const std::string base_file,
                                                          const std::string mem_index_file,
                                                          const std::string output_file,
                                                          const std::string reorder_data_file);

    template POLARIS_API void create_disk_layout<float>(const std::string base_file, const std::string mem_index_file,
                                                        const std::string output_file,
                                                        const std::string reorder_data_file);

    template POLARIS_API int8_t *load_warmup<int8_t>(const std::string &cache_warmup_file, uint64_t &warmup_num,
                                                     uint64_t warmup_dim, uint64_t warmup_aligned_dim);

    template POLARIS_API uint8_t *load_warmup<uint8_t>(const std::string &cache_warmup_file, uint64_t &warmup_num,
                                                       uint64_t warmup_dim, uint64_t warmup_aligned_dim);

    template POLARIS_API float *load_warmup<float>(const std::string &cache_warmup_file, uint64_t &warmup_num,
                                                   uint64_t warmup_dim, uint64_t warmup_aligned_dim);

    template POLARIS_API int build_merged_vamana_index<int8_t>(
            std::string base_file, polaris::MetricType compareMetric, uint32_t L, uint32_t R, double sampling_rate,
            double ram_budget, std::string mem_index_path, std::string medoids_path, std::string centroids_file,
            size_t build_pq_bytes, bool use_opq, uint32_t num_threads);

    template POLARIS_API int build_merged_vamana_index<float>(
            std::string base_file, polaris::MetricType compareMetric, uint32_t L, uint32_t R, double sampling_rate,
            double ram_budget, std::string mem_index_path, std::string medoids_path, std::string centroids_file,
            size_t build_pq_bytes, bool use_opq, uint32_t num_threads);

    template POLARIS_API int build_merged_vamana_index<uint8_t>(
            std::string base_file, polaris::MetricType compareMetric, uint32_t L, uint32_t R, double sampling_rate,
            double ram_budget, std::string mem_index_path, std::string medoids_path, std::string centroids_file,
            size_t build_pq_bytes, bool use_opq, uint32_t num_threads);
}; // namespace polaris
