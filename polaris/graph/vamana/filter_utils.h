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

#include <algorithm>
#include <fcntl.h>
#include <cassert>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>
#include <random>
#include <set>
#include <tuple>
#include <string>
#include <turbo/container/flat_hash_map.h>
#include <turbo/container/flat_hash_set.h>

#ifdef __APPLE__
#else

#include <malloc.h>

#endif

#include <sys/uio.h>

#include <polaris/io/cached_io.h>
#include <polaris/core/common_includes.h>
#include <polaris/utility/memory_mapper.h>
#include <polaris/graph/vamana/utils.h>
#include <polaris/utility/platform_macros.h>

// custom types (for readability)
typedef turbo::flat_hash_set<std::string> label_set;
typedef std::string path;

// structs for returning multiple items from a function
typedef std::tuple<std::vector<label_set>, turbo::flat_hash_map<std::string, uint32_t>, turbo::flat_hash_set<std::string>>
        parse_label_file_return_values;
typedef std::tuple<std::vector<std::vector<uint32_t>>, uint64_t> load_label_index_return_values;

namespace polaris {
    template<typename T>
    POLARIS_API void generate_label_indices(path input_data_path, path final_index_path_prefix, label_set all_labels,
                                            unsigned R, unsigned L, float alpha, unsigned num_threads);

    POLARIS_API load_label_index_return_values load_label_index(path label_index_path,
                                                                uint32_t label_number_of_points);

    template<typename LabelT>
    POLARIS_API std::tuple<std::vector<std::vector<LabelT>>, turbo::flat_hash_set<LabelT>> parse_formatted_label_file(
            path label_file);

    POLARIS_API parse_label_file_return_values parse_label_file(path label_data_path, std::string universal_label);

    template<typename T>
    POLARIS_API turbo::flat_hash_map<std::string, std::vector<uint32_t>> generate_label_specific_vector_files_compat(
            path input_data_path, turbo::flat_hash_map<std::string, uint32_t> labels_to_number_of_points,
            std::vector<label_set> point_ids_to_labels, label_set all_labels);

    /*
     * For each label, generates a file containing all vectors that have said label.
     * Also copies data from original bin file to new dimension-aligned file.
     *
     * Utilizes POSIX functions mmap and writev in order to minimize memory
     * overhead, so we include an STL version as well.
     *
     * Each data file is saved under the following format:
     *    input_data_path + "_" + label
     */

    template<typename T>
    inline turbo::flat_hash_map<std::string, std::vector<uint32_t>> generate_label_specific_vector_files(
            path input_data_path, turbo::flat_hash_map<std::string, uint32_t> labels_to_number_of_points,
            std::vector<label_set> point_ids_to_labels, label_set all_labels) {
        auto file_writing_timer = std::chrono::high_resolution_clock::now();
        polaris::MemoryMapper input_data(input_data_path);
        char *input_start = input_data.getBuf();

        uint32_t number_of_points, dimension;
        std::memcpy(&number_of_points, input_start, sizeof(uint32_t));
        std::memcpy(&dimension, input_start + sizeof(uint32_t), sizeof(uint32_t));
        const uint32_t VECTOR_SIZE = dimension * sizeof(T);
        const size_t METADATA = 2 * sizeof(uint32_t);
        if (number_of_points != point_ids_to_labels.size()) {
            std::cerr << "Error: number of points in labels file and data file differ." << std::endl;
            throw;
        }

        turbo::flat_hash_map<std::string, iovec *> label_to_iovec_map;
        turbo::flat_hash_map<std::string, uint32_t> label_to_curr_iovec;
        turbo::flat_hash_map<std::string, std::vector<uint32_t>> label_id_to_orig_id;

        // setup iovec list for each label
        for (const auto &lbl: all_labels) {
            iovec *label_iovecs = (iovec *) malloc(labels_to_number_of_points[lbl] * sizeof(iovec));
            if (label_iovecs == nullptr) {
                throw;
            }
            label_to_iovec_map[lbl] = label_iovecs;
            label_to_curr_iovec[lbl] = 0;
            label_id_to_orig_id[lbl].reserve(labels_to_number_of_points[lbl]);
        }

        // each point added to corresponding per-label iovec list
        for (uint32_t point_id = 0; point_id < number_of_points; point_id++) {
            char *curr_point = input_start + METADATA + (VECTOR_SIZE * point_id);
            iovec curr_iovec;

            curr_iovec.iov_base = curr_point;
            curr_iovec.iov_len = VECTOR_SIZE;
            for (const auto &lbl: point_ids_to_labels[point_id]) {
                *(label_to_iovec_map[lbl] + label_to_curr_iovec[lbl]) = curr_iovec;
                label_to_curr_iovec[lbl]++;
                label_id_to_orig_id[lbl].push_back(point_id);
            }
        }

        // write each label iovec to resp. file
        for (const auto &lbl: all_labels) {
            int label_input_data_fd;
            path curr_label_input_data_path(input_data_path + "_" + lbl);
            uint32_t curr_num_pts = labels_to_number_of_points[lbl];

            label_input_data_fd =
                    open(curr_label_input_data_path.c_str(), O_CREAT | O_WRONLY | O_TRUNC | O_APPEND, (mode_t) 0644);
            if (label_input_data_fd == -1)
                throw;

            // write metadata
            uint32_t metadata[2] = {curr_num_pts, dimension};
            int return_value = write(label_input_data_fd, metadata, sizeof(uint32_t) * 2);
            if (return_value == -1) {
                throw;
            }

            // limits on number of iovec structs per writev means we need to perform
            // multiple writevs
            size_t i = 0;
            while (curr_num_pts > IOV_MAX) {
                return_value = writev(label_input_data_fd, (label_to_iovec_map[lbl] + (IOV_MAX * i)), IOV_MAX);
                if (return_value == -1) {
                    close(label_input_data_fd);
                    throw;
                }
                curr_num_pts -= IOV_MAX;
                i += 1;
            }
            return_value = writev(label_input_data_fd, (label_to_iovec_map[lbl] + (IOV_MAX * i)), curr_num_pts);
            if (return_value == -1) {
                close(label_input_data_fd);
                throw;
            }

            free(label_to_iovec_map[lbl]);
            close(label_input_data_fd);
        }

        std::chrono::duration<double> file_writing_time =
                std::chrono::high_resolution_clock::now() - file_writing_timer;
        std::cout << "generated " << all_labels.size() << " label-specific vector files for index building in time "
                  << file_writing_time.count() << "\n"
                  << std::endl;

        return label_id_to_orig_id;
    }


    inline std::vector<uint32_t> loadTags(const std::string &tags_file, const std::string &base_file) {
        const bool tags_enabled = tags_file.empty() ? false : true;
        std::vector<uint32_t> location_to_tag;
        if (tags_enabled) {
            size_t tag_file_ndims, tag_file_npts;
            std::uint32_t *tag_data;
            polaris::load_bin<std::uint32_t>(tags_file, tag_data, tag_file_npts, tag_file_ndims);
            if (tag_file_ndims != 1) {
                polaris::cerr << "tags file error" << std::endl;
                throw polaris::PolarisException("tag file error", -1, __PRETTY_FUNCTION__, __FILE__, __LINE__);
            }

            // check if the point count match
            size_t base_file_npts, base_file_ndims;
            polaris::get_bin_metadata(base_file, base_file_npts, base_file_ndims);
            if (base_file_npts != tag_file_npts) {
                polaris::cerr << "point num in tags file mismatch" << std::endl;
                throw polaris::PolarisException("point num in tags file mismatch", -1, __PRETTY_FUNCTION__, __FILE__,
                                                __LINE__);
            }

            location_to_tag.assign(tag_data, tag_data + tag_file_npts);
            delete[] tag_data;
        }
        return location_to_tag;
    }

} // namespace polaris
