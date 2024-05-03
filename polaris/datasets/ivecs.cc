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

#include <polaris/datasets/ivecs.h>
#include <polaris/utility/platform_macros.h>
#include <polaris/graph/vamana/utils.h>
#include <iostream>

namespace polaris {

    void
    block_convert(std::ifstream &reader, std::ofstream &writer, uint32_t *read_buf, uint32_t *write_buf, size_t npts,
                  size_t ndims) {
        reader.read((char *) read_buf, npts * (ndims * sizeof(uint32_t) + sizeof(uint32_t)));
        for (size_t i = 0; i < npts; i++) {
            memcpy(write_buf + i * ndims, (read_buf + i * (ndims + 1)) + 1, ndims * sizeof(uint32_t));
        }
        writer.write((char *) write_buf, npts * ndims * sizeof(uint32_t));
    }

    turbo::Status ivecs_to_bin(const std::string &vecs_file, const std::string &bin_file) {
        std::ifstream reader(vecs_file, std::ios::binary | std::ios::ate);
        size_t fsize = reader.tellg();
        reader.seekg(0, std::ios::beg);

        uint32_t ndims_u32;
        reader.read((char *) &ndims_u32, sizeof(uint32_t));
        reader.seekg(0, std::ios::beg);
        size_t ndims = (size_t) ndims_u32;
        size_t npts = fsize / ((ndims + 1) * sizeof(uint32_t));
        std::cout << "Dataset: #pts = " << npts << ", # dims = " << ndims << std::endl;

        size_t blk_size = 131072;
        size_t nblks = ROUND_UP(npts, blk_size) / blk_size;
        std::cout << "# blks: " << nblks << std::endl;
        std::ofstream writer(bin_file, std::ios::binary);
        int npts_s32 = (int) npts;
        int ndims_s32 = (int) ndims;
        writer.write((char *) &npts_s32, sizeof(int));
        writer.write((char *) &ndims_s32, sizeof(int));
        uint32_t *read_buf = new uint32_t[npts * (ndims + 1)];
        uint32_t *write_buf = new uint32_t[npts * ndims];
        for (size_t i = 0; i < nblks; i++) {
            size_t cblk_size = std::min(npts - i * blk_size, blk_size);
            block_convert(reader, writer, read_buf, write_buf, cblk_size, ndims);
            std::cout << "Block #" << i << " written" << std::endl;
        }

        delete[] read_buf;
        delete[] write_buf;

        reader.close();
        writer.close();
        return turbo::ok_status();
    }

    turbo::Status uint32_to_uint8_bin(const std::string &vecs_file, const std::string &bin_file) {
        uint32_t *input;
        size_t npts, nd;
        polaris::load_bin<uint32_t>(vecs_file, input, npts, nd);
        uint8_t *output = new uint8_t[npts * nd];
        polaris::convert_types<uint32_t, uint8_t>(input, output, npts, nd);
        polaris::save_bin<uint8_t>(bin_file, output, npts, nd);
        delete[] output;
        delete[] input;
        return turbo::ok_status();
    }
}  // namespace polaris
