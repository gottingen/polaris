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

#include <polaris/datasets/to_float.h>
#include <polaris/internal/platform_macros.h>
#include <polaris/graph/utils.h>
#include <iostream>

namespace polaris {

    turbo::Status int8_to_float(const std::string &input_file, const std::string &output_file) {
        int8_t *input;
        size_t npts, nd;
        polaris::load_bin<int8_t>(input_file, input, npts, nd);
        float *output = new float[npts * nd];
        polaris::convert_types<int8_t, float>(input, output, npts, nd);
        polaris::save_bin<float>(output_file, output, npts, nd);
        delete[] output;
        delete[] input;
        return turbo::ok_status();
    }

    static void
    int8_to_float_scalar_block_convert(std::ofstream &writer, float *write_buf, std::ifstream &reader, int8_t *read_buf,
                                       size_t npts,
                                       size_t ndims, float bias, float scale) {
        reader.read((char *) read_buf, npts * ndims * sizeof(int8_t));

        for (size_t i = 0; i < npts; i++) {
            for (size_t d = 0; d < ndims; d++) {
                write_buf[d + i * ndims] = (((float) read_buf[d + i * ndims] - bias) * scale);
            }
        }
        writer.write((char *) write_buf, npts * ndims * sizeof(float));
    }

    turbo::Status
    int8_to_float_scalar(const std::string &input_file, const std::string &output_file, float bias, float scale) {
        std::ifstream reader(input_file, std::ios::binary);
        uint32_t npts_u32;
        uint32_t ndims_u32;
        reader.read((char *) &npts_u32, sizeof(uint32_t));
        reader.read((char *) &ndims_u32, sizeof(uint32_t));
        size_t npts = npts_u32;
        size_t ndims = ndims_u32;
        std::cout << "Dataset: #pts = " << npts << ", # dims = " << ndims << std::endl;

        size_t blk_size = 131072;
        size_t nblks = ROUND_UP(npts, blk_size) / blk_size;

        std::ofstream writer(output_file, std::ios::binary);
        auto read_buf = new int8_t[blk_size * ndims];
        auto write_buf = new float[blk_size * ndims];

        writer.write((char *) (&npts_u32), sizeof(uint32_t));
        writer.write((char *) (&ndims_u32), sizeof(uint32_t));

        for (size_t i = 0; i < nblks; i++) {
            size_t cblk_size = std::min(npts - i * blk_size, blk_size);
            int8_to_float_scalar_block_convert(writer, write_buf, reader, read_buf, cblk_size, ndims, bias, scale);
            std::cout << "Block #" << i << " written" << std::endl;
        }

        delete[] read_buf;
        delete[] write_buf;

        writer.close();
        reader.close();
        return turbo::ok_status();
    }

    turbo::Status uint8_to_float(const std::string &input_file, const std::string &output_file) {
        uint8_t *input;
        size_t npts, nd;
        polaris::load_bin<uint8_t>(input_file, input, npts, nd);
        float *output = new float[npts * nd];
        polaris::convert_types<uint8_t, float>(input, output, npts, nd);
        polaris::save_bin<float>(output_file, output, npts, nd);
        delete[] output;
        delete[] input;
        return turbo::ok_status();
    }
}  // namespace polaris
