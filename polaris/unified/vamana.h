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

#include <polaris/unified_index.h>
#include <polaris/graph/vamana/index_factory.h>

namespace polaris {

    class Vamana : public UnifiedIndex {
    public:
        Vamana() = default;

        ~Vamana() override = default;

        turbo::Status initialize(const IndexConfig &config) override;

        turbo::Status load(const std::string &index_path) override;

        turbo::Status save(const std::string &index_path) override;

        turbo::Status add(vid_t vid, const std::vector<uint8_t> &vec) override;

        turbo::Status lazy_remove(vid_t vid) override;

        turbo::ResultStatus<consolidation_report> consolidate_deletes(const IndexWriteParameters &parameters) override;

        turbo::Status search(SearchContext &context) override;

        [[nodiscard]] bool supports_dynamic() const override;

        [[nodiscard]] uint64_t snapshot() const override { return 0; }

        turbo::ResultStatus<uint32_t> optimize_beam_width(void *tuning_sample, uint64_t tuning_sample_num,
                                                          uint64_t tuning_sample_aligned_dim, uint32_t L,
                                                          uint32_t nthreads,
                                                          uint32_t start_bw = 2) override {
            return turbo::make_status(turbo::kUnimplemented, "Not implemented");
        }

    private:
        std::unique_ptr<AbstractIndex> index_;
        IndexConfig config_;
    };
}  // namespace polaris
