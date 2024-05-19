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

#include <fstream>
#include <string>
#include <iostream>
#include <cerrno>
#include <polaris/core/log.h>
#include <polaris/core/memory.h>
#include <polaris/io/utils.h>
#include <polaris/io/cached_io.h>
#include <collie/utility/status.h>
#include <collie/utility/result.h>

namespace polaris {

    inline void
    get_bin_metadata_impl(std::basic_istream<char> &reader, size_t &nrows, size_t &ncols, size_t offset = 0) {
        int nrows_32, ncols_32;
        reader.seekg(offset, reader.beg);
        reader.read((char *) &nrows_32, sizeof(int));
        reader.read((char *) &ncols_32, sizeof(int));
        nrows = nrows_32;
        ncols = ncols_32;
    }


    inline void get_bin_metadata(const std::string &bin_file, size_t &nrows, size_t &ncols, size_t offset = 0) {
        std::ifstream reader(bin_file.c_str(), std::ios::binary);
        get_bin_metadata_impl(reader, nrows, ncols, offset);
    }

    template<typename T>
    inline void
    load_bin_impl(std::basic_istream<char> &reader, T *&data, size_t &npts, size_t &dim, size_t file_offset = 0) {
        int npts_i32, dim_i32;

        reader.seekg(file_offset, reader.beg);
        reader.read((char *) &npts_i32, sizeof(int));
        reader.read((char *) &dim_i32, sizeof(int));
        npts = (unsigned) npts_i32;
        dim = (unsigned) dim_i32;

        POLARIS_LOG(INFO) << "Metadata: #pts = " << npts << ", #dims = " << dim << "...";

        data = new T[npts * dim];
        reader.read((char *) data, npts * dim * sizeof(T));
    }

    template<typename T>
    inline collie::Status load_bin(const std::string &bin_file, T *&data, size_t &npts, size_t &dim, size_t offset = 0) {
        POLARIS_LOG(INFO) << "Reading bin file " << bin_file.c_str() << " ...";
        std::ifstream reader;
        reader.exceptions(std::ifstream::failbit | std::ifstream::badbit);

        try {
            POLARIS_LOG(INFO) << "Opening bin file " << bin_file.c_str() << "... ";
            reader.open(bin_file, std::ios::binary | std::ios::ate);
            reader.seekg(0);
            load_bin_impl<T>(reader, data, npts, dim, offset);
        }
        catch (std::system_error &e) {
            return collie::Status::from_errno(errno, e.what());
        }
        POLARIS_LOG(INFO)<< "done.";
        return collie::Status::ok_status();
    }

    template<typename T>
    inline collie::Status load_bin(const std::string &bin_file, std::unique_ptr<T[]> &data, size_t &npts, size_t &dim,
                         size_t offset = 0) {
        T *ptr;
        auto r = load_bin<T>(bin_file, ptr, npts, dim, offset);
        data.reset(ptr);
        return r;
    }

    template<typename T>
    inline collie::Result<size_t> save_bin(const std::string &filename, T *data, size_t npts, size_t ndims, size_t offset = 0) {
        std::ofstream writer;
        COLLIE_RETURN_NOT_OK(open_file_to_write(writer, filename));
        POLARIS_LOG(INFO) << "Writing bin: " << filename.c_str();
        writer.seekp(offset, writer.beg);
        int npts_i32 = (int) npts, ndims_i32 = (int) ndims;
        size_t bytes_written = npts * ndims * sizeof(T) + 2 * sizeof(uint32_t);
        writer.write((char *) &npts_i32, sizeof(int));
        writer.write((char *) &ndims_i32, sizeof(int));
        POLARIS_LOG(INFO) << "bin: #pts = " << npts << ", #dims = " << ndims << ", size = " << bytes_written << "B";

        writer.write((char *) data, npts * ndims * sizeof(T));
        writer.close();
        POLARIS_LOG(INFO) << "Finished writing bin.";
        return bytes_written;
    }

    inline collie::Status load_tags(const std::string &tags_file, const std::string &base_file,std::vector<uint32_t> &location_to_tag) {
        const bool tags_enabled = tags_file.empty() ? false : true;
        if (tags_enabled) {
            size_t tag_file_ndims, tag_file_npts;
            std::uint32_t *tag_data;
            COLLIE_RETURN_NOT_OK(polaris::load_bin<std::uint32_t>(tags_file, tag_data, tag_file_npts, tag_file_ndims));
            if (tag_file_ndims != 1) {
                POLARIS_LOG(ERROR) << "tags file error";
                return collie::Status::invalid_argument("tags file error");
            }

            // check if the point count match
            size_t base_file_npts, base_file_ndims;
            polaris::get_bin_metadata(base_file, base_file_npts, base_file_ndims);
            if (base_file_npts != tag_file_npts) {
                POLARIS_LOG(ERROR) << "point num in tags file mismatch";
                return collie::Status::invalid_argument("point num in tags file mismatch");
            }

            location_to_tag.assign(tag_data, tag_data + tag_file_npts);
            delete[] tag_data;
        }
        return collie::Status::ok_status();
    }

    inline collie::Status load_tags(const std::string &tags_file, size_t base_file_npts,std::vector<uint32_t> &location_to_tag) {
        const bool tags_enabled = tags_file.empty() ? false : true;
        if (tags_enabled) {
            size_t tag_file_ndims, tag_file_npts;
            std::uint32_t *tag_data;
            COLLIE_RETURN_NOT_OK(polaris::load_bin<std::uint32_t>(tags_file, tag_data, tag_file_npts, tag_file_ndims));
            if (tag_file_ndims != 1) {
                POLARIS_LOG(ERROR) << "tags file error";
                return collie::Status::invalid_argument("tags file error");
            }

            if (base_file_npts != tag_file_npts) {
                POLARIS_LOG(ERROR) << "point num in tags file mismatch";
                return collie::Status::invalid_argument("point num in tags file mismatch");
            }

            location_to_tag.assign(tag_data, tag_data + tag_file_npts);
            delete[] tag_data;
        }
        return collie::Status::ok_status();
    }

    template<typename InType, typename OutType>
    void convert_types(const InType *srcmat, OutType *destmat, size_t npts, size_t dim) {
        for (int64_t i = 0; i < (int64_t) npts; i++) {
            for (uint64_t j = 0; j < dim; j++) {
                destmat[i * dim + j] = (OutType) srcmat[i * dim + j];
            }
        }
    }

    template<typename T>
    inline collie::Status load_aligned_bin_impl(std::basic_istream<char> &reader, size_t actual_file_size, T *&data, size_t &npts,
                                      size_t &dim, size_t &rounded_dim) {
        int npts_i32, dim_i32;
        reader.read((char *) &npts_i32, sizeof(int));
        reader.read((char *) &dim_i32, sizeof(int));
        npts = (unsigned) npts_i32;
        dim = (unsigned) dim_i32;

        size_t expected_actual_file_size = npts * dim * sizeof(T) + 2 * sizeof(uint32_t);
        if (actual_file_size != expected_actual_file_size) {
            POLARIS_LOG(ERROR) << "Error. File size mismatch. Actual size is " << actual_file_size << " while expected size is  "
                   << expected_actual_file_size << " npts = " << npts << " dim = " << dim << " size of <T>= "
                   << sizeof(T);
            return collie::Status::invalid_argument("File size mismatch");
        }
        rounded_dim = ROUND_UP(dim, 8);
        POLARIS_LOG(INFO)<< "Metadata: #pts = " << npts << ", #dims = " << dim << ", aligned_dim = " << rounded_dim
                      << "... ";
        size_t allocSize = npts * rounded_dim * sizeof(T);
        POLARIS_LOG(INFO) << "allocating aligned memory of " << allocSize << " bytes... ";
        alloc_aligned(((void **) &data), allocSize, 8 * sizeof(T));
        POLARIS_LOG(INFO) << "done. Copying data to mem_aligned buffer...";

        for (size_t i = 0; i < npts; i++) {
            reader.read((char *) (data + i * rounded_dim), dim * sizeof(T));
            memset(data + i * rounded_dim + dim, 0, (rounded_dim - dim) * sizeof(T));
        }
        POLARIS_LOG(INFO) << " done.";
        return collie::Status::ok_status();
    }

    template<typename T>
    inline collie::Status load_aligned_bin(const std::string &bin_file, T *&data, size_t &npts, size_t &dim, size_t &rounded_dim) {
        std::ifstream reader;
        reader.exceptions(std::ifstream::failbit | std::ifstream::badbit);
        POLARIS_LOG(INFO) << "Reading (with alignment) bin file " << bin_file << " ...";
        reader.open(bin_file, std::ios::binary | std::ios::ate);

        uint64_t fsize = reader.tellg();
        reader.seekg(0);
        return load_aligned_bin_impl(reader, fsize, data, npts, dim, rounded_dim);
    }

    inline collie::Status load_truthset(const std::string &bin_file, uint32_t *&ids, float *&dists, size_t &npts, size_t &dim) {
        size_t read_blk_size = 64 * 1024 * 1024;
        cached_ifstream reader(bin_file, read_blk_size);
        POLARIS_LOG(INFO) << "Reading truthset file " << bin_file.c_str() << " ...";
        size_t actual_file_size = reader.get_file_size();

        int npts_i32, dim_i32;
        reader.read((char *) &npts_i32, sizeof(int));
        reader.read((char *) &dim_i32, sizeof(int));
        npts = (unsigned) npts_i32;
        dim = (unsigned) dim_i32;

        POLARIS_LOG(INFO) << "Metadata: #pts = " << npts << ", #dims = " << dim << "... ";

        int truthset_type = -1; // 1 means truthset has ids and distances, 2 means
        // only ids, -1 is error
        size_t expected_file_size_with_dists = 2 * npts * dim * sizeof(uint32_t) + 2 * sizeof(uint32_t);

        if (actual_file_size == expected_file_size_with_dists)
            truthset_type = 1;

        size_t expected_file_size_just_ids = npts * dim * sizeof(uint32_t) + 2 * sizeof(uint32_t);

        if (actual_file_size == expected_file_size_just_ids)
            truthset_type = 2;

        if (truthset_type == -1) {
            POLARIS_LOG(INFO) << "Error. File size mismatch. File should have bin format, with "
                      "npts followed by ngt followed by npts*ngt ids and optionally "
                      "followed by npts*ngt distance values; actual size: "
                   << actual_file_size << ", expected: " << expected_file_size_with_dists << " or "
                   << expected_file_size_just_ids;
            return collie::Status::invalid_argument("File size mismatch");
        }

        ids = new uint32_t[npts * dim];
        reader.read((char *) ids, npts * dim * sizeof(uint32_t));

        if (truthset_type == 1) {
            dists = new float[npts * dim];
            reader.read((char *) dists, npts * dim * sizeof(float));
        }
        return collie::Status::ok_status();
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
    [[nodiscard]] inline collie::Status copy_aligned_data_from_file(const char *bin_file, T *&data, size_t &npts, size_t &dim,
                                            const size_t &rounded_dim, size_t offset = 0) {
        if (data == nullptr) {
            return collie::Status::invalid_argument("Null pointer passed to copy_aligned_data_from_file function");
        }
        try {
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
        } catch (std::system_error &e) {
            return collie::Status::from_errno(errno,e.what());
        }
        return collie::Status::ok_status();
    }

}  // namespace polaris
