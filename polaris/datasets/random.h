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

#include <string>
#include <turbo/status/status.h>
#include <polaris/graph/partition.h>

namespace polaris {

    template <typename T> int aux_gen(const std::string &base_file, const std::string &output_prefix, float sampling_rate) {
        gen_random_slice<T>(base_file, output_prefix, sampling_rate);
        return 0;
    }

    void gen_random_float(const std::string &base_file, const std::string &output_prefix, float sampling_rate) {
        aux_gen<float>(base_file, output_prefix, sampling_rate);
    }

    void gen_random_uint8(const std::string &base_file, const std::string &output_prefix, float sampling_rate) {
        aux_gen<uint8_t>(base_file, output_prefix, sampling_rate);
    }

    void gen_random_int8(const std::string &base_file, const std::string &output_prefix, float sampling_rate) {
        aux_gen<int8_t>(base_file, output_prefix, sampling_rate);
    }

    turbo::Status generate_pq_float(const std::string &base_file, const std::string &output_prefix, uint32_t num_pq_chunks, float sampling_rate, bool opq);
    turbo::Status generate_pq_int8(const std::string &base_file, const std::string &output_prefix, uint32_t num_pq_chunks, float sampling_rate, bool opq);
    turbo::Status generate_pq_uint8(const std::string &base_file, const std::string &output_prefix, uint32_t num_pq_chunks, float sampling_rate, bool opq);

}  // namespace polaris
