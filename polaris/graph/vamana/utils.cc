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

#include <polaris/graph/vamana/utils.h>

#include <stdio.h>


const uint32_t MAX_REQUEST_SIZE = 1024 * 1024 * 1024; // 64MB
const uint32_t MAX_SIMULTANEOUS_READ_REQUESTS = 128;


bool Avx2SupportedCPU = true;
bool AvxSupportedCPU = false;

namespace polaris {

    void block_convert(std::ofstream &writr, std::ifstream &readr, float *read_buf, size_t npts, size_t ndims) {
        readr.read((char *) read_buf, npts * ndims * sizeof(float));
        uint32_t ndims_u32 = (uint32_t) ndims;
#pragma omp parallel for
        for (int64_t i = 0; i < (int64_t) npts; i++) {
            float norm_pt = std::numeric_limits<float>::epsilon();
            for (uint32_t dim = 0; dim < ndims_u32; dim++) {
                norm_pt += *(read_buf + i * ndims + dim) * *(read_buf + i * ndims + dim);
            }
            norm_pt = std::sqrt(norm_pt);
            for (uint32_t dim = 0; dim < ndims_u32; dim++) {
                *(read_buf + i * ndims + dim) = *(read_buf + i * ndims + dim) / norm_pt;
            }
        }
        writr.write((char *) read_buf, npts * ndims * sizeof(float));
    }

    void normalize_data_file(const std::string &inFileName, const std::string &outFileName) {
        std::ifstream readr(inFileName, std::ios::binary);
        std::ofstream writr(outFileName, std::ios::binary);

        int npts_s32, ndims_s32;
        readr.read((char *) &npts_s32, sizeof(int32_t));
        readr.read((char *) &ndims_s32, sizeof(int32_t));

        writr.write((char *) &npts_s32, sizeof(int32_t));
        writr.write((char *) &ndims_s32, sizeof(int32_t));

        size_t npts = (size_t) npts_s32;
        size_t ndims = (size_t) ndims_s32;
        polaris::cout << "Normalizing FLOAT vectors in file: " << inFileName << std::endl;
        polaris::cout << "Dataset: #pts = " << npts << ", # dims = " << ndims << std::endl;

        size_t blk_size = 131072;
        size_t nblks = ROUND_UP(npts, blk_size) / blk_size;
        polaris::cout << "# blks: " << nblks << std::endl;

        float *read_buf = new float[npts * ndims];
        for (size_t i = 0; i < nblks; i++) {
            size_t cblk_size = std::min(npts - i * blk_size, blk_size);
            block_convert(writr, readr, read_buf, cblk_size, ndims);
        }
        delete[] read_buf;

        polaris::cout << "Wrote normalized points to file: " << outFileName << std::endl;
    }

    double calculate_range_search_recall(uint32_t num_queries, std::vector<std::vector<uint32_t>> &groundtruth,
                                         std::vector<std::vector<uint32_t>> &our_results) {
        double total_recall = 0;
        std::set<uint32_t> gt, res;

        for (size_t i = 0; i < num_queries; i++) {
            gt.clear();
            res.clear();

            gt.insert(groundtruth[i].begin(), groundtruth[i].end());
            res.insert(our_results[i].begin(), our_results[i].end());
            uint32_t cur_recall = 0;
            for (auto &v: gt) {
                if (res.find(v) != res.end()) {
                    cur_recall++;
                }
            }
            if (gt.size() != 0)
                total_recall += ((100.0 * cur_recall) / gt.size());
            else
                total_recall += 100;
        }
        return total_recall / (num_queries);
    }


} // namespace polaris
