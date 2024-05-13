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


#include <algorithm>
#include <atomic>
#include <cassert>
#include <fstream>
#include <iostream>
#include <set>
#include <string>
#include <vector>
#include <collie/cli/cli.h>
#include <polaris/graph/vamana/disk_utils.h>

namespace polaris {
    void setup_merge_shards_cli(collie::App *app) {
        std::string vamana_prefix;
        std::string vamana_suffix;
        std::string idmaps_prefix;
        std::string idmaps_suffix;
        uint64_t nshards;
        uint32_t max_degree;
        std::string output_index;
        std::string output_medoids;

        app->add_option("--vamana_index_prefix", vamana_prefix, "vamana_index_prefix")->required();
        app->add_option("--vamana_index_suffix", vamana_suffix, "vamana_index_suffix")->required();
        app->add_option("--idmaps_prefix", idmaps_prefix, "idmaps_prefix")->required();
        app->add_option("--idmaps_suffix", idmaps_suffix, "idmaps_suffix")->required();
        app->add_option("--n_shards", nshards, "n_shards")->required();
        app->add_option("--max_degree", max_degree, "max_degree")->required();
        app->add_option("--output_vamana_path", output_index, "output_vamana_path")->required();
        app->add_option("--output_medoids_path", output_medoids, "output_medoids_path")->required();
        app->callback(
                [&vamana_prefix, &vamana_suffix, &idmaps_prefix, &idmaps_suffix, &nshards, &max_degree, &output_index, &output_medoids]() {
                    return polaris::merge_shards(vamana_prefix, vamana_suffix, idmaps_prefix, idmaps_suffix, nshards,
                                                 max_degree,
                                                 output_index, output_medoids);
                });

    }
}  // namespace polaris