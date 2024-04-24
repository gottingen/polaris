/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <polaris/gpu/utils/warpselect/WarpSelectImpl.cuh>

namespace polaris {
namespace gpu {

WARP_SELECT_IMPL(float, true, 1024, 8);

}
} // namespace polaris
