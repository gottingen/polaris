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

#include <errno.h>
#include <polaris/utility/platform_macros.h>
#include <polaris/utility/common_includes.h>
#include <collie/filesystem/fs.h>

#ifdef __APPLE__
#else

#include <malloc.h>

#endif

#include <polaris/distance/distance.h>
#include <polaris/graph/vamana/logger.h>
#include <polaris/io/cached_io.h>
#include <polaris/io/utils.h>
#include <polaris/utility/polaris_exception.h>
#include <polaris/utility/platform_macros.h>
#include <turbo/container/flat_hash_set.h>
#include <polaris/utility/types.h>
#include <any>



// generates formatted_label and _labels_map file.
inline void convert_labels_string_to_int(const std::string &inFileName, const std::string &outFileName,
                                         const std::string &mapFileName, const std::string &unv_label) {
    std::unordered_map<std::string, uint32_t> string_int_map;
    std::ofstream label_writer(outFileName);
    std::ifstream label_reader(inFileName);
    if (unv_label != "")
        string_int_map[unv_label] = 0; // if universal label is provided map it to 0 always
    std::string line, token;
    while (std::getline(label_reader, line)) {
        std::istringstream new_iss(line);
        std::vector<uint32_t> lbls;
        while (getline(new_iss, token, ',')) {
            token.erase(std::remove(token.begin(), token.end(), '\n'), token.end());
            token.erase(std::remove(token.begin(), token.end(), '\r'), token.end());
            if (string_int_map.find(token) == string_int_map.end()) {
                uint32_t nextId = (uint32_t) string_int_map.size() + 1;
                string_int_map[token] = nextId; // nextId can never be 0
            }
            lbls.push_back(string_int_map[token]);
        }
        if (lbls.size() <= 0) {
            std::cout << "No label found";
            exit(-1);
        }
        for (size_t j = 0; j < lbls.size(); j++) {
            if (j != lbls.size() - 1)
                label_writer << lbls[j] << ",";
            else
                label_writer << lbls[j] << std::endl;
        }
    }
    label_writer.close();

    std::ofstream map_writer(mapFileName);
    for (auto mp: string_int_map) {
        map_writer << mp.first << "\t" << mp.second << std::endl;
    }
    map_writer.close();
}


namespace polaris {
    static const size_t MAX_SIZE_OF_STREAMBUF = 2LL * 1024 * 1024 * 1024;

    inline void print_error_and_terminate(std::stringstream &error_stream) {
        polaris::cerr << error_stream.str() << std::endl;
        throw polaris::PolarisException(error_stream.str(), -1, __PRETTY_FUNCTION__, __FILE__, __LINE__);
    }

    inline void report_memory_allocation_failure() {
        std::stringstream stream;
        stream << "Memory Allocation Failed.";
        print_error_and_terminate(stream);
    }

    inline void report_misalignment_of_requested_size(size_t align) {
        std::stringstream stream;
        stream << "Requested memory size is not a multiple of " << align << ". Can not be allocated.";
        print_error_and_terminate(stream);
    }

    inline void alloc_aligned(void **ptr, size_t size, size_t align) {
        *ptr = nullptr;
        if (IS_ALIGNED(size, align) == 0)
            report_misalignment_of_requested_size(align);
#ifndef _WINDOWS
        *ptr = ::aligned_alloc(align, size);
#else
        *ptr = ::_aligned_malloc(size, align); // note the swapped arguments!
#endif
        if (*ptr == nullptr)
            report_memory_allocation_failure();
    }

    inline void realloc_aligned(void **ptr, size_t size, size_t align) {
        if (IS_ALIGNED(size, align) == 0)
            report_misalignment_of_requested_size(align);
#ifdef _WINDOWS
        *ptr = ::_aligned_realloc(*ptr, size, align);
#else
        polaris::cerr << "No aligned realloc on GCC. Must malloc and mem_align, "
                         "left it out for now."
                      << std::endl;
#endif
        if (*ptr == nullptr)
            report_memory_allocation_failure();
    }

    inline void check_stop(std::string arnd) {
        int brnd;
        polaris::cout << arnd << std::endl;
        std::cin >> brnd;
    }

    inline void aligned_free(void *ptr) {
        // Gopal. Must have a check here if the pointer was actually allocated by
        // _alloc_aligned
        if (ptr == nullptr) {
            return;
        }
#ifndef _WINDOWS
        free(ptr);
#else
        ::_aligned_free(ptr);
#endif
    }

    inline void GenRandom(std::mt19937 &rng, unsigned *addr, unsigned size, unsigned N) {
        for (unsigned i = 0; i < size; ++i) {
            addr[i] = rng() % (N - size);
        }

        std::sort(addr, addr + size);
        for (unsigned i = 1; i < size; ++i) {
            if (addr[i] <= addr[i - 1]) {
                addr[i] = addr[i - 1] + 1;
            }
        }
        unsigned off = rng() % N;
        for (unsigned i = 0; i < size; ++i) {
            addr[i] = (addr[i] + off) % N;
        }
    }

    // get_bin_metadata functions END

    inline size_t get_graph_num_frozen_points(const std::string &graph_file) {
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
    inline std::string getValues(T *data, size_t num) {
        std::stringstream stream;
        stream << "[";
        for (size_t i = 0; i < num; i++) {
            stream << std::to_string(data[i]) << ",";
        }
        stream << "]" << std::endl;

        return stream.str();
    }


    inline void wait_for_keystroke() {
        int a;
        std::cout << "Press any number to continue.." << std::endl;
        std::cin >> a;
    }
    // load_bin functions END

    inline void load_truthset(const std::string &bin_file, uint32_t *&ids, float *&dists, size_t &npts, size_t &dim) {
        size_t read_blk_size = 64 * 1024 * 1024;
        cached_ifstream reader(bin_file, read_blk_size);
        polaris::cout << "Reading truthset file " << bin_file.c_str() << " ..." << std::endl;
        size_t actual_file_size = reader.get_file_size();

        int npts_i32, dim_i32;
        reader.read((char *) &npts_i32, sizeof(int));
        reader.read((char *) &dim_i32, sizeof(int));
        npts = (unsigned) npts_i32;
        dim = (unsigned) dim_i32;

        polaris::cout << "Metadata: #pts = " << npts << ", #dims = " << dim << "... " << std::endl;

        int truthset_type = -1; // 1 means truthset has ids and distances, 2 means
        // only ids, -1 is error
        size_t expected_file_size_with_dists = 2 * npts * dim * sizeof(uint32_t) + 2 * sizeof(uint32_t);

        if (actual_file_size == expected_file_size_with_dists)
            truthset_type = 1;

        size_t expected_file_size_just_ids = npts * dim * sizeof(uint32_t) + 2 * sizeof(uint32_t);

        if (actual_file_size == expected_file_size_just_ids)
            truthset_type = 2;

        if (truthset_type == -1) {
            std::stringstream stream;
            stream << "Error. File size mismatch. File should have bin format, with "
                      "npts followed by ngt followed by npts*ngt ids and optionally "
                      "followed by npts*ngt distance values; actual size: "
                   << actual_file_size << ", expected: " << expected_file_size_with_dists << " or "
                   << expected_file_size_just_ids;
            polaris::cout << stream.str();
            throw polaris::PolarisException(stream.str(), -1, __PRETTY_FUNCTION__, __FILE__, __LINE__);
        }

        ids = new uint32_t[npts * dim];
        reader.read((char *) ids, npts * dim * sizeof(uint32_t));

        if (truthset_type == 1) {
            dists = new float[npts * dim];
            reader.read((char *) dists, npts * dim * sizeof(float));
        }
    }

    inline void prune_truthset_for_range(const std::string &bin_file, float range,
                                         std::vector<std::vector<uint32_t>> &groundtruth, size_t &npts) {
        size_t read_blk_size = 64 * 1024 * 1024;
        cached_ifstream reader(bin_file, read_blk_size);
        polaris::cout << "Reading truthset file " << bin_file.c_str() << "... " << std::endl;
        size_t actual_file_size = reader.get_file_size();

        int npts_i32, dim_i32;
        reader.read((char *) &npts_i32, sizeof(int));
        reader.read((char *) &dim_i32, sizeof(int));
        npts = (unsigned) npts_i32;
        uint64_t dim = (unsigned) dim_i32;
        uint32_t *ids;
        float *dists;

        polaris::cout << "Metadata: #pts = " << npts << ", #dims = " << dim << "... " << std::endl;

        int truthset_type = -1; // 1 means truthset has ids and distances, 2 means
        // only ids, -1 is error
        size_t expected_file_size_with_dists = 2 * npts * dim * sizeof(uint32_t) + 2 * sizeof(uint32_t);

        if (actual_file_size == expected_file_size_with_dists)
            truthset_type = 1;

        if (truthset_type == -1) {
            std::stringstream stream;
            stream << "Error. File size mismatch. File should have bin format, with "
                      "npts followed by ngt followed by npts*ngt ids and optionally "
                      "followed by npts*ngt distance values; actual size: "
                   << actual_file_size << ", expected: " << expected_file_size_with_dists;
            polaris::cout << stream.str();
            throw polaris::PolarisException(stream.str(), -1, __PRETTY_FUNCTION__, __FILE__, __LINE__);
        }

        ids = new uint32_t[npts * dim];
        reader.read((char *) ids, npts * dim * sizeof(uint32_t));

        if (truthset_type == 1) {
            dists = new float[npts * dim];
            reader.read((char *) dists, npts * dim * sizeof(float));
        }
        float min_dist = std::numeric_limits<float>::max();
        float max_dist = 0;
        groundtruth.resize(npts);
        for (uint32_t i = 0; i < npts; i++) {
            groundtruth[i].clear();
            for (uint32_t j = 0; j < dim; j++) {
                if (dists[i * dim + j] <= range) {
                    groundtruth[i].emplace_back(ids[i * dim + j]);
                }
                min_dist = min_dist > dists[i * dim + j] ? dists[i * dim + j] : min_dist;
                max_dist = max_dist < dists[i * dim + j] ? dists[i * dim + j] : max_dist;
            }
            // std::cout<<groundtruth[i].size() << " " ;
        }
        std::cout << "Min dist: " << min_dist << ", Max dist: " << max_dist << std::endl;
        delete[] ids;
        delete[] dists;
    }

    inline void load_range_truthset(const std::string &bin_file, std::vector<std::vector<uint32_t>> &groundtruth,
                                    uint64_t &gt_num) {
        size_t read_blk_size = 64 * 1024 * 1024;
        cached_ifstream reader(bin_file, read_blk_size);
        polaris::cout << "Reading truthset file " << bin_file.c_str() << "... " << std::flush;
        size_t actual_file_size = reader.get_file_size();

        int nptsuint32_t, totaluint32_t;
        reader.read((char *) &nptsuint32_t, sizeof(int));
        reader.read((char *) &totaluint32_t, sizeof(int));

        gt_num = (uint64_t) nptsuint32_t;
        uint64_t total_res = (uint64_t) totaluint32_t;

        polaris::cout << "Metadata: #pts = " << gt_num << ", #total_results = " << total_res << "..." << std::endl;

        size_t expected_file_size = 2 * sizeof(uint32_t) + gt_num * sizeof(uint32_t) + total_res * sizeof(uint32_t);

        if (actual_file_size != expected_file_size) {
            std::stringstream stream;
            stream << "Error. File size mismatch in range truthset. actual size: " << actual_file_size
                   << ", expected: " << expected_file_size;
            polaris::cout << stream.str();
            throw polaris::PolarisException(stream.str(), -1, __PRETTY_FUNCTION__, __FILE__, __LINE__);
        }
        groundtruth.clear();
        groundtruth.resize(gt_num);
        std::vector<uint32_t> gt_count(gt_num);

        reader.read((char *) gt_count.data(), sizeof(uint32_t) * gt_num);

        std::vector<uint32_t> gt_stats(gt_count);
        std::sort(gt_stats.begin(), gt_stats.end());

        std::cout << "GT count percentiles:" << std::endl;
        for (uint32_t p = 0; p < 100; p += 5)
            std::cout << "percentile " << p << ": " << gt_stats[static_cast<size_t>(std::floor((p / 100.0) * gt_num))]
                      << std::endl;
        std::cout << "percentile 100"
                  << ": " << gt_stats[gt_num - 1] << std::endl;

        for (uint32_t i = 0; i < gt_num; i++) {
            groundtruth[i].clear();
            groundtruth[i].resize(gt_count[i]);
            if (gt_count[i] != 0)
                reader.read((char *) groundtruth[i].data(), sizeof(uint32_t) * gt_count[i]);
        }
    }

    POLARIS_API double calculate_range_search_recall(unsigned num_queries,
                                                     std::vector<std::vector<uint32_t>> &groundtruth,
                                                     std::vector<std::vector<uint32_t>> &our_results);

    inline void print_progress(double percentage) {
        int val = (int) (percentage * 100);
        int lpad = (int) (percentage * PBWIDTH);
        int rpad = PBWIDTH - lpad;
        printf("\r%3d%% [%.*s%*s]", val, lpad, PBSTR, rpad, "");
        fflush(stdout);
    }

    // this function will take in_file of n*d dimensions and save the output as a
    // floating point matrix
    // with n*(d+1) dimensions. All vectors are scaled by a large value M so that
    // the norms are <=1 and the final coordinate is set so that the resulting
    // norm (in d+1 coordinates) is equal to 1 this is a classical transformation
    // from MIPS to L2 search from "On Symmetric and Asymmetric LSHs for Inner
    // Product Search" by Neyshabur and Srebro

    template<typename T>
    float prepare_base_for_inner_products(const std::string in_file, const std::string out_file) {
        std::cout << "Pre-processing base file by adding extra coordinate" << std::endl;
        std::ifstream in_reader(in_file.c_str(), std::ios::binary);
        std::ofstream out_writer(out_file.c_str(), std::ios::binary);
        uint64_t npts, in_dims, out_dims;
        float max_norm = 0;

        uint32_t npts32, dims32;
        in_reader.read((char *) &npts32, sizeof(uint32_t));
        in_reader.read((char *) &dims32, sizeof(uint32_t));

        npts = npts32;
        in_dims = dims32;
        out_dims = in_dims + 1;
        uint32_t outdims32 = (uint32_t) out_dims;

        out_writer.write((char *) &npts32, sizeof(uint32_t));
        out_writer.write((char *) &outdims32, sizeof(uint32_t));

        size_t BLOCK_SIZE = 100000;
        size_t block_size = npts <= BLOCK_SIZE ? npts : BLOCK_SIZE;
        std::unique_ptr<T[]> in_block_data = std::make_unique<T[]>(block_size * in_dims);
        std::unique_ptr<float[]> out_block_data = std::make_unique<float[]>(block_size * out_dims);

        std::memset(out_block_data.get(), 0, sizeof(float) * block_size * out_dims);
        uint64_t num_blocks = DIV_ROUND_UP(npts, block_size);

        std::vector<float> norms(npts, 0);

        for (uint64_t b = 0; b < num_blocks; b++) {
            uint64_t start_id = b * block_size;
            uint64_t end_id = (b + 1) * block_size < npts ? (b + 1) * block_size : npts;
            uint64_t block_pts = end_id - start_id;
            in_reader.read((char *) in_block_data.get(), block_pts * in_dims * sizeof(T));
            for (uint64_t p = 0; p < block_pts; p++) {
                for (uint64_t j = 0; j < in_dims; j++) {
                    norms[start_id + p] += in_block_data[p * in_dims + j] * in_block_data[p * in_dims + j];
                }
                max_norm = max_norm > norms[start_id + p] ? max_norm : norms[start_id + p];
            }
        }

        max_norm = std::sqrt(max_norm);

        in_reader.seekg(2 * sizeof(uint32_t), std::ios::beg);
        for (uint64_t b = 0; b < num_blocks; b++) {
            uint64_t start_id = b * block_size;
            uint64_t end_id = (b + 1) * block_size < npts ? (b + 1) * block_size : npts;
            uint64_t block_pts = end_id - start_id;
            in_reader.read((char *) in_block_data.get(), block_pts * in_dims * sizeof(T));
            for (uint64_t p = 0; p < block_pts; p++) {
                for (uint64_t j = 0; j < in_dims; j++) {
                    out_block_data[p * out_dims + j] = in_block_data[p * in_dims + j] / max_norm;
                }
                float res = 1 - (norms[start_id + p] / (max_norm * max_norm));
                res = res <= 0 ? 0 : std::sqrt(res);
                out_block_data[p * out_dims + out_dims - 1] = res;
            }
            out_writer.write((char *) out_block_data.get(), block_pts * out_dims * sizeof(float));
        }
        out_writer.close();
        return max_norm;
    }

    // plain saves data as npts X ndims array into filename
    template<typename T>
    void save_Tvecs(const char *filename, T *data, size_t npts, size_t ndims) {
        std::string fname(filename);

        // create cached ofstream with 64MB cache
        cached_ofstream writer(fname, 64 * 1048576);

        unsigned dims_u32 = (unsigned) ndims;

        // start writing
        for (size_t i = 0; i < npts; i++) {
            // write dims in u32
            writer.write((char *) &dims_u32, sizeof(unsigned));

            // get cur point in data
            T *cur_pt = data + i * ndims;
            writer.write((char *) cur_pt, ndims * sizeof(T));
        }
    }

    template<typename T>
    inline size_t save_data_in_base_dimensions(const std::string &filename, T *data, size_t npts, size_t ndims,
                                               size_t aligned_dim, size_t offset = 0) {
        std::ofstream writer; //(filename, std::ios::binary | std::ios::out);
        open_file_to_write(writer, filename);
        int npts_i32 = (int) npts, ndims_i32 = (int) ndims;
        size_t bytes_written = 2 * sizeof(uint32_t) + npts * ndims * sizeof(T);
        writer.seekp(offset, writer.beg);
        writer.write((char *) &npts_i32, sizeof(int));
        writer.write((char *) &ndims_i32, sizeof(int));
        for (size_t i = 0; i < npts; i++) {
            writer.write((char *) (data + i * aligned_dim), ndims * sizeof(T));
        }
        writer.close();
        return bytes_written;
    }

    template<typename T>
    inline void copy_aligned_data_from_file(const char *bin_file, T *&data, size_t &npts, size_t &dim,
                                            const size_t &rounded_dim, size_t offset = 0) {
        if (data == nullptr) {
            polaris::cerr << "Memory was not allocated for " << data << " before calling the load function. Exiting..."
                          << std::endl;
            throw polaris::PolarisException("Null pointer passed to copy_aligned_data_from_file function", -1,
                                            __PRETTY_FUNCTION__,
                                            __FILE__, __LINE__);
        }
        std::ifstream reader;
        reader.exceptions(std::ios::badbit | std::ios::failbit);
        reader.open(bin_file, std::ios::binary);
        reader.seekg(offset, reader.beg);

        int npts_i32, dim_i32;
        reader.read((char *) &npts_i32, sizeof(int));
        reader.read((char *) &dim_i32, sizeof(int));
        npts = (unsigned) npts_i32;
        dim = (unsigned) dim_i32;

        for (size_t i = 0; i < npts; i++) {
            reader.read((char *) (data + i * rounded_dim), dim * sizeof(T));
            memset(data + i * rounded_dim + dim, 0, (rounded_dim - dim) * sizeof(T));
        }
    }

// NOTE :: good efficiency when total_vec_size is integral multiple of 64
    inline void prefetch_vector(const char *vec, size_t vecsize) {
        size_t max_prefetch_size = (vecsize / 64) * 64;
        for (size_t d = 0; d < max_prefetch_size; d += 64)
            _mm_prefetch((const char *) vec + d, _MM_HINT_T0);
    }

// NOTE :: good efficiency when total_vec_size is integral multiple of 64
    inline void prefetch_vector_l2(const char *vec, size_t vecsize) {
        size_t max_prefetch_size = (vecsize / 64) * 64;
        for (size_t d = 0; d < max_prefetch_size; d += 64)
            _mm_prefetch((const char *) vec + d, _MM_HINT_T1);
    }

    // NOTE: Implementation in utils.cpp.
    void block_convert(std::ofstream &writr, std::ifstream &readr, float *read_buf, uint64_t npts, uint64_t ndims);

    POLARIS_API void normalize_data_file(const std::string &inFileName, const std::string &outFileName);

}; // namespace polaris

struct PivotContainer {
    PivotContainer() = default;

    PivotContainer(size_t pivo_id, float pivo_dist) : piv_id{pivo_id}, piv_dist{pivo_dist} {
    }

    bool operator<(const PivotContainer &p) const {
        return p.piv_dist < piv_dist;
    }

    bool operator>(const PivotContainer &p) const {
        return p.piv_dist > piv_dist;
    }

    size_t piv_id;
    float piv_dist;
};

inline bool validate_index_file_size(std::ifstream &in) {
    if (!in.is_open())
        throw polaris::PolarisException("VamanaIndex file size check called on unopened file stream", -1, __PRETTY_FUNCTION__,
                                        __FILE__,
                                        __LINE__);
    in.seekg(0, in.end);
    size_t actual_file_size = in.tellg();
    in.seekg(0, in.beg);
    size_t expected_file_size;
    in.read((char *) &expected_file_size, sizeof(uint64_t));
    in.seekg(0, in.beg);
    if (actual_file_size != expected_file_size) {
        polaris::cerr << "VamanaIndex file size error. Expected size (metadata): " << expected_file_size
                      << ", actual file size : " << actual_file_size << "." << std::endl;
        return false;
    }
    return true;
}

template<typename T>
inline float get_norm(T *arr, const size_t dim) {
    float sum = 0.0f;
    for (uint32_t i = 0; i < dim; i++) {
        sum += arr[i] * arr[i];
    }
    return sqrt(sum);
}

// This function is valid only for float data type.
template<typename T = float>
inline void normalize(T *arr, const size_t dim) {
    float norm = get_norm(arr, dim);
    for (uint32_t i = 0; i < dim; i++) {
        arr[i] = (T) (arr[i] / norm);
    }
}

inline std::vector<std::string> read_file_to_vector_of_strings(const std::string &filename, bool unique = false) {
    std::vector<std::string> result;
    std::set<std::string> elementSet;
    if (filename != "") {
        std::ifstream file(filename);
        if (file.fail()) {
            throw polaris::PolarisException(std::string("Failed to open file ") + filename, -1);
        }
        std::string line;
        while (std::getline(file, line)) {
            if (line.empty()) {
                break;
            }
            if (line.find(',') != std::string::npos) {
                std::cerr << "Every query must have exactly one filter" << std::endl;
                exit(-1);
            }
            if (!line.empty() && (line.back() == '\r' || line.back() == '\n')) {
                line.erase(line.size() - 1);
            }
            if (!elementSet.count(line)) {
                result.push_back(line);
            }
            if (unique) {
                elementSet.insert(line);
            }
        }
        file.close();
    } else {
        throw polaris::PolarisException(std::string("Failed to open file. filename can not be blank"), -1);
    }
    return result;
}

inline void
clean_up_artifacts(turbo::flat_hash_set<std::string> paths_to_clean, turbo::flat_hash_set<std::string> path_suffixes) {
    try {
        for (const auto &path: paths_to_clean) {
            for (const auto &suffix: path_suffixes) {
                std::string curr_path_to_clean(path + "_" + suffix);
                if (std::remove(curr_path_to_clean.c_str()) != 0)
                    polaris::cout << "Warning: Unable to remove file :" << curr_path_to_clean << std::endl;
            }
        }
        polaris::cout << "Cleaned all artifacts" << std::endl;
    }
    catch (const std::exception &e) {
        polaris::cout << "Warning: Unable to clean all artifacts " << e.what() << std::endl;
    }
}

#ifdef _WINDOWS
#include <intrin.h>
#include <Psapi.h>

extern bool AvxSupportedCPU;
extern bool Avx2SupportedCPU;

inline size_t getMemoryUsage()
{
    PROCESS_MEMORY_COUNTERS_EX pmc;
    GetProcessMemoryInfo(GetCurrentProcess(), (PROCESS_MEMORY_COUNTERS *)&pmc, sizeof(pmc));
    return pmc.PrivateUsage;
}

inline std::string getWindowsErrorMessage(DWORD lastError)
{
    char *errorText;
    FormatMessageA(
        // use system message tables to retrieve error text
        FORMAT_MESSAGE_FROM_SYSTEM
            // allocate buffer on local heap for error text
            | FORMAT_MESSAGE_ALLOCATE_BUFFER
            // Important! will fail otherwise, since we're not
            // (and CANNOT) pass insertion parameters
            | FORMAT_MESSAGE_IGNORE_INSERTS,
        NULL, // unused with FORMAT_MESSAGE_FROM_SYSTEM
        lastError, MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT),
        (LPSTR)&errorText, // output
        0,                 // minimum size for output buffer
        NULL);             // arguments - see note

    return errorText != nullptr ? std::string(errorText) : std::string();
}

inline void printProcessMemory(const char *message)
{
    PROCESS_MEMORY_COUNTERS counters;
    HANDLE h = GetCurrentProcess();
    GetProcessMemoryInfo(h, &counters, sizeof(counters));
    polaris::cout << message
                  << " [Peaking Working Set size: " << counters.PeakWorkingSetSize * 1.0 / (1024.0 * 1024 * 1024)
                  << "GB Working set size: " << counters.WorkingSetSize * 1.0 / (1024.0 * 1024 * 1024)
                  << "GB Private bytes " << counters.PagefileUsage * 1.0 / (1024 * 1024 * 1024) << "GB]" << std::endl;
}
#else

// need to check and change this
inline bool avx2Supported() {
    return true;
}

inline void printProcessMemory(const char *) {
}

inline size_t getMemoryUsage() { // for non-windows, we have not implemented this function
    return 0;
}

#endif

extern bool AvxSupportedCPU;
extern bool Avx2SupportedCPU;
