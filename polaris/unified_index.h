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
#include <collie/utility/status.h>
#include <polaris/core/index_config.h>
#include <polaris/core/percentile_stats.h>
#include <polaris/core/report.h>

namespace polaris {

    struct UnifiedBuildParameters {
        std::string data_file;
        std::string tags_file;
        size_t num_points_to_load;
        void *data{nullptr};
        std::vector<vid_t> tags;
        std::string output_path;
        // for disk index
        std::string codebook_prefix;
    };


    class UnifiedIndex {
    public:
        virtual ~UnifiedIndex() = default;

        virtual collie::Status initialize(const IndexConfig &config) = 0;

        [[nodiscard]] virtual collie::Status build(const UnifiedBuildParameters &parameters) = 0;

        [[nodiscard]] virtual collie::Status load(const std::string &index_path) = 0;

        virtual collie::Status save(const std::string &index_path) = 0;

        virtual collie::Status add(vid_t vid, const void *vec) = 0;

        virtual collie::Status lazy_remove(vid_t vid) = 0;

        virtual collie::Status get_vector(vid_t vid, void *vec) const = 0;

        virtual size_t size() const = 0;

        virtual collie::Result<consolidation_report>
        consolidate_deletes(const IndexWriteParameters &parameters) = 0;

        virtual collie::Status search(SearchContext &context) = 0;

        [[nodiscard]] virtual bool supports_dynamic() const = 0;

        /// @brief Get the snapshot of the index
        /// @return the snapshot of the index 0 if not supported
        [[nodiscard]] virtual uint64_t snapshot() const = 0;

        /// @brief Optimize the beam width for the index
        virtual collie::Result<uint32_t> optimize_beam_width(void *tuning_sample, uint64_t tuning_sample_num,
                                                                  uint64_t tuning_sample_aligned_dim, uint32_t L,
                                                                  uint32_t nthreads,
                                                                  uint32_t start_bw = 2) = 0;

        static collie::Result<size_t> get_frozen_points(IndexType it, const std::string &index_path);

        static UnifiedIndex *create_index(IndexType type);
    };
}  // namespace polaris
