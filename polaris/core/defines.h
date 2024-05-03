// Copyright 2024 The EA Authors.
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

/*
#cmakedefine NGT_GRAPH_CHECK_VECTOR		// use vector to check whether accessed
#cmakedefine NGT_AVX2				// use avx2
#cmakedefine NGT_LARGE_DATASET			// more than 10M objects
#cmakedefine NGT_DISTANCE_COMPUTATION_COUNT	// count # of distance computations
#cmakedefine NGT_QBG_DISABLED
#cmakedefine NGTQG_ZERO_GLOBAL
#cmakedefine NGTQG_NO_ROTATION
// End of cmake defines
*/
//////////////////////////////////////////////////////////////////////////
// Release Definitions for OSS

//#define		NGT_DISTANCE_COMPUTATION_COUNT
//#define		NGT_SEARCH_TIMER

#define		NGT_CREATION_EDGE_SIZE			10
#define		NGT_EXPLORATION_COEFFICIENT		1.1
#define		NGT_INSERTION_EXPLORATION_COEFFICIENT	1.1
#define		NGT_SHARED_MEMORY_MAX_SIZE		1024	// MB
#define		NGT_FORCED_REMOVE		// When errors occur due to the index inconsistency, ignore them.

#define		NGT_COMPACT_VECTOR
#define		NGT_GRAPH_READ_ONLY_GRAPH

#ifdef	NGT_LARGE_DATASET
#define	NGT_GRAPH_CHECK_HASH_BASED_BOOLEAN_SET
#else
#define	NGT_GRAPH_CHECK_VECTOR
#endif


#if defined(NGT_AVX2)
#undef NGT_AVX512
#else
#if defined(__AVX512F__) && defined(__AVX512DQ__)
#define NGT_AVX512
#elif defined(__AVX2__)
#define NGT_AVX2
#else
#define NGT_NO_AVX
#endif
#endif


#include <vector>

namespace polaris {

    static constexpr size_t kAlignment = collie::simd::best_arch::alignment();

    template<typename T>
    using aligned_vector = std::vector<T, collie::aligned_allocator<T, kAlignment>>;

}  // namespace polaris