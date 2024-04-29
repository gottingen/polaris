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

#include <polaris/datasets/fvecs.h>
#include <polaris/internal/platform_macros.h>
#include <polaris/graph/utils.h>
#include <iostream>

namespace polaris {

    void
    block_convert_float(std::ifstream &reader, std::ofstream &writer, float *read_buf, float *write_buf, size_t npts,
                        size_t ndims) {
        reader.read((char *) read_buf, npts * (ndims * sizeof(float) + sizeof(uint32_t)));
        for (size_t i = 0; i < npts; i++) {
            memcpy(write_buf + i * ndims, (read_buf + i * (ndims + 1)) + 1, ndims * sizeof(float));
        }
        writer.write((char *) write_buf, npts * ndims * sizeof(float));
    }

    // Convert byte types
    void block_convert_byte(std::ifstream &reader, std::ofstream &writer, uint8_t *read_buf, uint8_t *write_buf,
                            size_t npts, size_t ndims) {
        reader.read((char *) read_buf, npts * (ndims * sizeof(uint8_t) + sizeof(uint32_t)));
        for (size_t i = 0; i < npts; i++) {
            memcpy(write_buf + i * ndims, (read_buf + i * (ndims + sizeof(uint32_t))) + sizeof(uint32_t),
                   ndims * sizeof(uint8_t));
        }
        writer.write((char *) write_buf, npts * ndims * sizeof(uint8_t));
    }

    turbo::Status float_vecs_to_bin(const std::string &vecs_file, const std::string &bin_file) {

        int datasize = sizeof(float);

        std::ifstream reader(vecs_file, std::ios::binary | std::ios::ate);
        size_t fsize = reader.tellg();
        reader.seekg(0, std::ios::beg);

        uint32_t ndims_u32;
        reader.read((char *) &ndims_u32, sizeof(uint32_t));
        reader.seekg(0, std::ios::beg);
        size_t ndims = (size_t) ndims_u32;
        size_t npts = fsize / ((ndims * datasize) + sizeof(uint32_t));
        std::cout << "Dataset: #pts = " << npts << ", # dims = " << ndims << std::endl;

        size_t blk_size = 131072;
        size_t nblks = ROUND_UP(npts, blk_size) / blk_size;
        std::cout << "# blks: " << nblks << std::endl;
        std::ofstream writer(bin_file, std::ios::binary);
        int32_t npts_s32 = (int32_t) npts;
        int32_t ndims_s32 = (int32_t) ndims;
        writer.write((char *) &npts_s32, sizeof(int32_t));
        writer.write((char *) &ndims_s32, sizeof(int32_t));

        size_t chunknpts = std::min(npts, blk_size);
        uint8_t *read_buf = new uint8_t[chunknpts * ((ndims * datasize) + sizeof(uint32_t))];
        uint8_t *write_buf = new uint8_t[chunknpts * ndims * datasize];

        for (size_t i = 0; i < nblks; i++) {
            size_t cblk_size = std::min(npts - i * blk_size, blk_size);
            block_convert_float(reader, writer, (float *) read_buf, (float *) write_buf, cblk_size, ndims);
            std::cout << "Block #" << i << " written" << std::endl;
        }

        delete[] read_buf;
        delete[] write_buf;

        reader.close();
        writer.close();
        return turbo::ok_status();
    }

    turbo::Status uint8_vecs_to_bin(const std::string &vecs_file, const std::string &bin_file) {

        int datasize = sizeof(uint8_t);

        std::ifstream reader(vecs_file, std::ios::binary | std::ios::ate);
        size_t fsize = reader.tellg();
        reader.seekg(0, std::ios::beg);

        uint32_t ndims_u32;
        reader.read((char *) &ndims_u32, sizeof(uint32_t));
        reader.seekg(0, std::ios::beg);
        size_t ndims = (size_t) ndims_u32;
        size_t npts = fsize / ((ndims * datasize) + sizeof(uint32_t));
        std::cout << "Dataset: #pts = " << npts << ", # dims = " << ndims << std::endl;

        size_t blk_size = 131072;
        size_t nblks = ROUND_UP(npts, blk_size) / blk_size;
        std::cout << "# blks: " << nblks << std::endl;
        std::ofstream writer(bin_file, std::ios::binary);
        int32_t npts_s32 = (int32_t) npts;
        int32_t ndims_s32 = (int32_t) ndims;
        writer.write((char *) &npts_s32, sizeof(int32_t));
        writer.write((char *) &ndims_s32, sizeof(int32_t));

        size_t chunknpts = std::min(npts, blk_size);
        uint8_t *read_buf = new uint8_t[chunknpts * ((ndims * datasize) + sizeof(uint32_t))];
        uint8_t *write_buf = new uint8_t[chunknpts * ndims * datasize];

        for (size_t i = 0; i < nblks; i++) {
            size_t cblk_size = std::min(npts - i * blk_size, blk_size);
            block_convert_byte(reader, writer, read_buf, write_buf, cblk_size, ndims);
            std::cout << "Block #" << i << " written" << std::endl;
        }

        delete[] read_buf;
        delete[] write_buf;

        reader.close();
        writer.close();
        return turbo::ok_status();
    }

    void block_convert(std::ifstream &reader, std::ofstream &writer, float *read_buf, uint8_t *write_buf, size_t npts,
                       size_t ndims) {
        reader.read((char *) read_buf, npts * (ndims * sizeof(float) + sizeof(uint32_t)));
        for (size_t i = 0; i < npts; i++) {
            memcpy(write_buf + i * (ndims + 4), read_buf + i * (ndims + 1), sizeof(uint32_t));
            for (size_t d = 0; d < ndims; d++)
                write_buf[i * (ndims + 4) + 4 + d] = (uint8_t) read_buf[i * (ndims + 1) + 1 + d];
        }
        writer.write((char *) write_buf, npts * (ndims * 1 + 4));
    }

    turbo::Status float_vecs_to_uint8_vecs(const std::string &vecs_file, const std::string &bin_file) {
        std::ifstream reader(vecs_file, std::ios::binary | std::ios::ate);
        size_t fsize = reader.tellg();
        reader.seekg(0, std::ios::beg);

        uint32_t ndims_u32;
        reader.read((char *) &ndims_u32, sizeof(uint32_t));
        reader.seekg(0, std::ios::beg);
        size_t ndims = (size_t) ndims_u32;
        size_t npts = fsize / ((ndims + 1) * sizeof(float));
        std::cout << "Dataset: #pts = " << npts << ", # dims = " << ndims << std::endl;

        size_t blk_size = 131072;
        size_t nblks = ROUND_UP(npts, blk_size) / blk_size;
        std::cout << "# blks: " << nblks << std::endl;
        std::ofstream writer(bin_file, std::ios::binary);
        auto read_buf = new float[npts * (ndims + 1)];
        auto write_buf = new uint8_t[npts * (ndims + 4)];
        for (size_t i = 0; i < nblks; i++) {
            size_t cblk_size = std::min(npts - i * blk_size, blk_size);
            block_convert(reader, writer, read_buf, write_buf, cblk_size, ndims);
            std::cout << "Block #" << i << " written" << std::endl;
        }

        delete[] read_buf;
        delete[] write_buf;

        reader.close();
        writer.close();
    }

    static void block_convert(std::ofstream &writer, int8_t *write_buf, std::ifstream &reader, float *read_buf, size_t npts,
                       size_t ndims, float bias, float scale) {
        reader.read((char *)read_buf, npts * ndims * sizeof(float));

        for (size_t i = 0; i < npts; i++)
        {
            for (size_t d = 0; d < ndims; d++)
            {
                write_buf[d + i * ndims] = (int8_t)((read_buf[d + i * ndims] - bias) * (254.0 / scale));
            }
        }
        writer.write((char *)write_buf, npts * ndims);
    }

    turbo::Status float_bin_to_int8(const std::string &src, const std::string &dst, float bias, float scale) {

        std::ifstream reader(src, std::ios::binary);
        uint32_t npts_u32;
        uint32_t ndims_u32;
        reader.read((char *)&npts_u32, sizeof(uint32_t));
        reader.read((char *)&ndims_u32, sizeof(uint32_t));
        size_t npts = npts_u32;
        size_t ndims = ndims_u32;
        std::cout << "Dataset: #pts = " << npts << ", # dims = " << ndims << std::endl;

        size_t blk_size = 131072;
        size_t nblks = ROUND_UP(npts, blk_size) / blk_size;

        std::ofstream writer(dst, std::ios::binary);
        auto read_buf = new float[blk_size * ndims];
        auto write_buf = new int8_t[blk_size * ndims];
        writer.write((char *)(&npts_u32), sizeof(uint32_t));
        writer.write((char *)(&ndims_u32), sizeof(uint32_t));

        for (size_t i = 0; i < nblks; i++) {
            size_t cblk_size = std::min(npts - i * blk_size, blk_size);
            block_convert(writer, write_buf, reader, read_buf, cblk_size, ndims, bias, scale);
            std::cout << "Block #" << i << " written" << std::endl;
        }

        delete[] read_buf;
        delete[] write_buf;

        writer.close();
        reader.close();
    }
} // namespace polaris