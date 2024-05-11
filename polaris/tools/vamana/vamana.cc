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

#include <polaris/tools/vamana/vamana.h>

namespace polaris {

    void setup_vamana_cli(collie::App *app) {
        auto build_disk_index_cmd = app->add_subcommand("build_disk_index", "Build a disk-based index.");
        setup_build_disk_index_cli(build_disk_index_cmd);
        auto build_memory_index_cmd = app->add_subcommand("build_memory_index", "Build a memory-based index.");
        setup_build_memory_index_cli(build_memory_index_cmd);
        auto search_disk_index_cmd = app->add_subcommand("search_disk_index", "Search a disk-based index.");
        setup_search_disk_index_cli(search_disk_index_cmd);
        auto search_memory_index_cmd = app->add_subcommand("search_memory_index", "Search a memory-based index.");
        setup_search_memory_index_cli(search_memory_index_cmd);
        auto merge_shards_cmd = app->add_subcommand("merge_shards", "Merge shards.");
        setup_merge_shards_cli(merge_shards_cmd);
        app->require_subcommand(1);
    }
}  // namespace polaris
