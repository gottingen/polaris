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
#include <polaris/datasets/recall.h>
#include <polaris/utility/platform_macros.h>
#include <polaris/graph/vamana/utils.h>
#include <iostream>

namespace polaris {

    turbo::Status calculate_recall(const std::string &vecs_file, const std::string &bin_file, uint32_t r) {
        uint32_t *gold_std = NULL;
        float *gs_dist = nullptr;
        uint32_t *our_results = NULL;
        float *or_dist = nullptr;
        size_t points_num, points_num_gs, points_num_or;
        size_t dim_gs;
        size_t dim_or;
        polaris::load_truthset(vecs_file, gold_std, gs_dist, points_num_gs, dim_gs);
        polaris::load_truthset(bin_file, our_results, or_dist, points_num_or, dim_or);

        if (points_num_gs != points_num_or) {
            std::cout << "Error. Number of queries mismatch in ground truth and "
                         "our results"
                      << std::endl;
            return turbo::internal_error("Error. Number of queries mismatch in ground truth and our results");
        }
        points_num = points_num_gs;

        uint32_t recall_at = r;

        if ((dim_or < recall_at) || (recall_at > dim_gs)) {
            std::cout << "ground truth has size " << dim_gs << "; our set has " << dim_or
                      << " points. Asking for recall "
                      << recall_at << std::endl;
            return turbo::internal_error(
                    "Error. Recall at is greater than the number of points in the ground truth set");
        }
        std::cout << "Calculating recall@" << recall_at << std::endl;
        double recall_val = polaris::calculate_recall((uint32_t) points_num, gold_std, gs_dist, (uint32_t) dim_gs,
                                                      our_results, (uint32_t) dim_or, (uint32_t) recall_at);

        //  double avg_recall = (recall*1.0)/(points_num*1.0);
        std::cout << "Avg. recall@" << recall_at << " is " << recall_val << "\n";
        return turbo::ok_status();
    }
}  // namespace polaris

