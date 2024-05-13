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
#pragma once

#include <cstdint>
#include <vector>
#include <turbo/container/flat_hash_set.h>
#include <polaris/core/lock.h>
#include <polaris/core/search_context.h>
#include <turbo/status/status.h>
#include <polaris/core/index_config.h>
#include <polaris/core/percentile_stats.h>
#include <polaris/core/report.h>

namespace polaris {

    enum class IndexType {
        IT_NONE,
        IT_FLAT,
        IT_FLATIP,
        IT_FLATL2,
        IT_LSH,
        IT_IVFFLAT,
        IT_VAMANA_DISK,
        IT_VAMANA
    };


    class UnifiedIndex {
    public:
        virtual ~UnifiedIndex() = default;

        virtual turbo::Status initialize(const IndexConfig &config) = 0;

        virtual turbo::Status load(const std::string &index_path) = 0;

        virtual turbo::Status save(const std::string &index_path) = 0;

        virtual turbo::Status add(vid_t vid, const std::vector<uint8_t> &vec) = 0;

        virtual turbo::Status lazy_remove(vid_t vid) = 0;

        virtual turbo::ResultStatus<consolidation_report> consolidate_deletes(const IndexWriteParameters &parameters) = 0;

        virtual turbo::Status search(SearchContext &context) = 0;

        [[nodiscard]] virtual bool supports_dynamic() const  = 0;

        /// @brief Get the snapshot of the index
        /// @return the snapshot of the index 0 if not supported
        [[nodiscard]] virtual uint64_t snapshot() const = 0;

        /// @brief Optimize the beam width for the index
        virtual turbo::ResultStatus<uint32_t> optimize_beam_width(void *tuning_sample, uint64_t tuning_sample_num,
                                                uint64_t tuning_sample_aligned_dim, uint32_t L, uint32_t nthreads,
                                                uint32_t start_bw = 2) = 0;

        static turbo::ResultStatus<size_t> get_frozen_points(IndexType it, const std::string &index_path);

        static UnifiedIndex *create_index(IndexType type);
    };
}  // namespace polaris
