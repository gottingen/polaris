/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <polaris/index.h>
#include <vector>

namespace polaris {
namespace gpu {

/// Utility function to translate (list id, offset) to a user index on
/// the CPU. In a cpp in order to use OpenMP.
void ivfOffsetToUserIndex(
        idx_t* indices,
        idx_t numLists,
        idx_t queries,
        int k,
        const std::vector<std::vector<idx_t>>& listOffsetToUserIndex);

} // namespace gpu
} // namespace polaris
