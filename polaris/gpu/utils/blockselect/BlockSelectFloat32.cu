/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <polaris/gpu/utils/blockselect/BlockSelectImpl.cuh>

namespace polaris {
namespace gpu {

BLOCK_SELECT_IMPL(float, true, 32, 2);
BLOCK_SELECT_IMPL(float, false, 32, 2);

} // namespace gpu
} // namespace polaris
