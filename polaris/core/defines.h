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

#include <collie/simd/simd.h>

#if !COLLIE_SIMD_WITH_AVX2 && !COLLIE_SIMD_WITH_NEON64
#error "No AVX2 support or NEON64 support"
#endif

#include <vector>

namespace polaris {

    static constexpr size_t kAlignment = collie::simd::best_arch::alignment();

    template<typename T>
    using aligned_vector = std::vector<T, collie::aligned_allocator<T, kAlignment>>;

}  // namespace polaris