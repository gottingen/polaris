/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// -*- c++ -*-

#pragma once
#include <polaris/utility/platform_macros.h>

namespace polaris {

/// The metric space for vector comparison for Faiss indices and algorithms.
///
/// Most algorithms support both inner product and L2, with the flat
/// (brute-force) indices supporting additional metric types for vector
/// comparison.

enum MetricType {
    METRIC_NONE = -1,          ///< undefined
    METRIC_L1 = 0,                ///< L1 (aka cityblock)
    METRIC_L2 = 1,            ///< squared L2 search
    METRIC_HAMMING = 2,        ///< Hamming distance (binary vectors)
    METRIC_ANGLE = 3,          ///< angle distance
    METRIC_COSINE = 4,            ///< cosine similarity
    METRIC_NORMALIZED_ANGLE = 5,  ///< normalized angle distance
    METRIC_NORMALIZED_COSINE = 6, ///< normalized cosine distance
    METRIC_JACCARD = 7,            ///< Jaccard distance defined as: sum_i(min(a_i, b_i)) / sum_i(max(a_i, b_i))
    METRIC_SPARSE_JACCARD = 8,    ///< Jaccard distance for sparse binary vectors
    METRIC_NORMALIZED_L2 = 9,     ///< normalized L2 distance
    METRIC_INNER_PRODUCT = 10, ///< maximum inner product search
    METRIC_POINCARE = 11,     ///< Poincare distance
    METRIC_LORENTZ = 12,      ///< Lorentz distance

    /// some additional metrics defined in scipy.spatial.distance
    METRIC_Linf,              ///< infinity distance
    METRIC_Lp,                ///< L_p distance, p is given by a polaris::Index
    /// metric_arg
    METRIC_Canberra = 20,
    METRIC_BrayCurtis,
    METRIC_JensenShannon,
};

/// all vector indices are this type
using idx_t = int64_t;

/// this function is used to distinguish between min and max indexes since
/// we need to support similarity and dis-similarity metrics in a flexible way
constexpr bool is_similarity_metric(MetricType metric_type) {
    return ((metric_type == METRIC_INNER_PRODUCT) ||
            (metric_type == METRIC_JACCARD));
}

} // namespace polaris
