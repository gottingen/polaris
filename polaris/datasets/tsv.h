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

#include <turbo/status/status.h>
#include <string>

namespace polaris {
    turbo::Status
    tsv_to_float_bin(const std::string &tsv_file, const std::string &bin_file, uint32_t ndims, uint32_t nvec);

    turbo::Status
    tsv_to_uint8_bin(const std::string &tsv_file, const std::string &bin_file, uint32_t ndims, uint32_t nvec);

    turbo::Status
    tsv_to_int8_bin(const std::string &tsv_file, const std::string &bin_file, uint32_t ndims, uint32_t nvec);

    turbo::Status
    int8_bin_to_tsv(const std::string &tsv_file, const std::string &bin_file);

    turbo::Status
    uint8_bin_to_tsv(const std::string &tsv_file, const std::string &bin_file);

    turbo::Status
    float_bin_to_tsv(const std::string &tsv_file, const std::string &bin_file);
}  // namespace polaris