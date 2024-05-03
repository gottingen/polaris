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
#include <string>
#include <polaris/datasets/tsv.h>
#include <polaris/utility/platform_macros.h>
#include <polaris/graph/vamana/utils.h>
#include <iostream>

namespace polaris {

    static void block_convert_float(std::ifstream &reader, std::ofstream &writer, size_t npts, size_t ndims) {
        auto read_buf = new float[npts * (ndims + 1)];

        auto cursor = read_buf;
        float val;

        for (size_t i = 0; i < npts; i++) {
            for (size_t d = 0; d < ndims; ++d) {
                reader >> val;
                *cursor = val;
                cursor++;
            }
        }
        writer.write((char *) read_buf, npts * ndims * sizeof(float));
        delete[] read_buf;
    }

    static void block_convert_int8(std::ifstream &reader, std::ofstream &writer, size_t npts, size_t ndims) {
        auto read_buf = new int8_t[npts * (ndims + 1)];

        auto cursor = read_buf;
        int val;

        for (size_t i = 0; i < npts; i++) {
            for (size_t d = 0; d < ndims; ++d) {
                reader >> val;
                *cursor = (int8_t) val;
                cursor++;
            }
        }
        writer.write((char *) read_buf, npts * ndims * sizeof(uint8_t));
        delete[] read_buf;
    }

    static void block_convert_uint8(std::ifstream &reader, std::ofstream &writer, size_t npts, size_t ndims) {
        auto read_buf = new uint8_t[npts * (ndims + 1)];

        auto cursor = read_buf;
        int val;

        for (size_t i = 0; i < npts; i++) {
            for (size_t d = 0; d < ndims; ++d) {
                reader >> val;
                *cursor = (uint8_t) val;
                cursor++;
            }
        }
        writer.write((char *) read_buf, npts * ndims * sizeof(uint8_t));
        delete[] read_buf;
    }

    turbo::Status
    tsv_to_float_bin(const std::string &tsv_file, const std::string &bin_file, uint32_t ndims, uint32_t npts) {

        std::ifstream reader(tsv_file, std::ios::binary | std::ios::ate);
        //  size_t          fsize = reader.tellg();
        reader.seekg(0, std::ios::beg);
        reader.seekg(0, std::ios::beg);

        size_t blk_size = 131072;
        size_t nblks = ROUND_UP(npts, blk_size) / blk_size;
        std::cout << "# blks: " << nblks << std::endl;
        std::ofstream writer(bin_file, std::ios::binary);
        auto npts_u32 = (uint32_t) npts;
        auto ndims_u32 = (uint32_t) ndims;
        writer.write((char *) &npts_u32, sizeof(uint32_t));
        writer.write((char *) &ndims_u32, sizeof(uint32_t));

        for (size_t i = 0; i < nblks; i++) {
            size_t cblk_size = std::min(npts - i * blk_size, blk_size);
            block_convert_float(reader, writer, cblk_size, ndims);
            std::cout << "Block #" << i << " written" << std::endl;
        }

        reader.close();
        writer.close();
        return turbo::ok_status();
    }

    turbo::Status
    tsv_to_uint8_bin(const std::string &tsv_file, const std::string &bin_file, uint32_t ndims, uint32_t npts) {
        std::ifstream reader(tsv_file, std::ios::binary | std::ios::ate);
        //  size_t          fsize = reader.tellg();
        reader.seekg(0, std::ios::beg);
        reader.seekg(0, std::ios::beg);

        size_t blk_size = 131072;
        size_t nblks = ROUND_UP(npts, blk_size) / blk_size;
        std::cout << "# blks: " << nblks << std::endl;
        std::ofstream writer(bin_file, std::ios::binary);
        auto npts_u32 = (uint32_t) npts;
        auto ndims_u32 = (uint32_t) ndims;
        writer.write((char *) &npts_u32, sizeof(uint32_t));
        writer.write((char *) &ndims_u32, sizeof(uint32_t));

        for (size_t i = 0; i < nblks; i++) {
            size_t cblk_size = std::min(npts - i * blk_size, blk_size);
            block_convert_uint8(reader, writer, cblk_size, ndims);
            std::cout << "Block #" << i << " written" << std::endl;
        }

        reader.close();
        writer.close();
        return turbo::ok_status();
    }

    turbo::Status
    tsv_to_int8_bin(const std::string &tsv_file, const std::string &bin_file, uint32_t ndims, uint32_t npts) {
        std::ifstream reader(tsv_file, std::ios::binary | std::ios::ate);
        //  size_t          fsize = reader.tellg();
        reader.seekg(0, std::ios::beg);
        reader.seekg(0, std::ios::beg);

        size_t blk_size = 131072;
        size_t nblks = ROUND_UP(npts, blk_size) / blk_size;
        std::cout << "# blks: " << nblks << std::endl;
        std::ofstream writer(bin_file, std::ios::binary);
        auto npts_u32 = (uint32_t) npts;
        auto ndims_u32 = (uint32_t) ndims;
        writer.write((char *) &npts_u32, sizeof(uint32_t));
        writer.write((char *) &ndims_u32, sizeof(uint32_t));

        for (size_t i = 0; i < nblks; i++) {
            size_t cblk_size = std::min(npts - i * blk_size, blk_size);
            block_convert_int8(reader, writer, cblk_size, ndims);
            std::cout << "Block #" << i << " written" << std::endl;
        }

        reader.close();
        writer.close();
        return turbo::ok_status();
    }


    template<class T>
    static void block_convert(std::ofstream &writer, std::ifstream &reader, T *read_buf, size_t npts, size_t ndims) {
        reader.read((char *) read_buf, npts * ndims * sizeof(float));

        for (size_t i = 0; i < npts; i++) {
            for (size_t d = 0; d < ndims; d++) {
                writer << read_buf[d + i * ndims];
                if (d < ndims - 1)
                    writer << "\t";
                else
                    writer << "\n";
            }
        }
    }

    turbo::Status
    int8_bin_to_tsv(const std::string &tsv_file, const std::string &bin_file) {
        std::ifstream reader(tsv_file, std::ios::binary);
        uint32_t npts_u32;
        uint32_t ndims_u32;
        reader.read((char *) &npts_u32, sizeof(uint32_t));
        reader.read((char *) &ndims_u32, sizeof(uint32_t));
        size_t npts = npts_u32;
        size_t ndims = ndims_u32;
        std::cout << "Dataset: #pts = " << npts << ", # dims = " << ndims << std::endl;

        size_t blk_size = 131072;
        size_t nblks = ROUND_UP(npts, blk_size) / blk_size;

        std::ofstream writer(bin_file);
        char *read_buf = new char[blk_size * ndims * 4];
        for (size_t i = 0; i < nblks; i++) {
            size_t cblk_size = std::min(npts - i * blk_size, blk_size);
            block_convert<int8_t>(writer, reader, (int8_t *) read_buf, cblk_size, ndims);
            std::cout << "Block #" << i << " written" << std::endl;
        }

        delete[] read_buf;

        writer.close();
        reader.close();
        return turbo::ok_status();
    }

    turbo::Status
    uint8_bin_to_tsv(const std::string &tsv_file, const std::string &bin_file) {
        std::ifstream reader(tsv_file, std::ios::binary);
        uint32_t npts_u32;
        uint32_t ndims_u32;
        reader.read((char *) &npts_u32, sizeof(uint32_t));
        reader.read((char *) &ndims_u32, sizeof(uint32_t));
        size_t npts = npts_u32;
        size_t ndims = ndims_u32;
        std::cout << "Dataset: #pts = " << npts << ", # dims = " << ndims << std::endl;

        size_t blk_size = 131072;
        size_t nblks = ROUND_UP(npts, blk_size) / blk_size;

        std::ofstream writer(bin_file);
        char *read_buf = new char[blk_size * ndims * 4];
        for (size_t i = 0; i < nblks; i++) {
            size_t cblk_size = std::min(npts - i * blk_size, blk_size);
            block_convert<uint8_t>(writer, reader, (uint8_t *) read_buf, cblk_size, ndims);
            std::cout << "Block #" << i << " written" << std::endl;
        }

        delete[] read_buf;

        writer.close();
        reader.close();
        return turbo::ok_status();
    }

    turbo::Status
    float_bin_to_tsv(const std::string &tsv_file, const std::string &bin_file) {
        std::ifstream reader(tsv_file, std::ios::binary);
        uint32_t npts_u32;
        uint32_t ndims_u32;
        reader.read((char *) &npts_u32, sizeof(uint32_t));
        reader.read((char *) &ndims_u32, sizeof(uint32_t));
        size_t npts = npts_u32;
        size_t ndims = ndims_u32;
        std::cout << "Dataset: #pts = " << npts << ", # dims = " << ndims << std::endl;

        size_t blk_size = 131072;
        size_t nblks = ROUND_UP(npts, blk_size) / blk_size;

        std::ofstream writer(bin_file);
        char *read_buf = new char[blk_size * ndims * 4];
        for (size_t i = 0; i < nblks; i++) {
            size_t cblk_size = std::min(npts - i * blk_size, blk_size);
            block_convert<float>(writer, reader, (float *) read_buf, cblk_size, ndims);
            std::cout << "Block #" << i << " written" << std::endl;
        }

        delete[] read_buf;

        writer.close();
        reader.close();
        return turbo::ok_status();
    }

}  // namespace polaris
