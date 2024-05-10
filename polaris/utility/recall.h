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

#include <polaris/utility/platform_macros.h>
#include <polaris/utility/common_includes.h>
#include <turbo/container/flat_hash_set.h>
#include <polaris/core/search_context.h>
#include <memory>

namespace polaris {

    POLARIS_API double calculate_recall(unsigned num_queries, unsigned *gold_std, float *gs_dist, unsigned dim_gs,
                                        unsigned *our_results, unsigned dim_or, unsigned recall_at);

    POLARIS_API double calculate_recall(unsigned num_queries, unsigned *gold_std, float *gs_dist, unsigned dim_gs,
                                        const std::vector<std::unique_ptr<SearchContext >> &ctx_list, unsigned dim_or, unsigned recall_at);

    POLARIS_API double calculate_recall(unsigned num_queries, unsigned *gold_std, float *gs_dist, unsigned dim_gs,
                                        unsigned *our_results, unsigned dim_or, unsigned recall_at,
                                        const turbo::flat_hash_set<unsigned> &active_tags);
}  // namespace polaris
