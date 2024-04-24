/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <polaris/gpu/utils/blockselect/BlockSelectImpl.cuh>

namespace faiss {
namespace gpu {

BLOCK_SELECT_IMPL(float, false, 512, 8);

}
} // namespace faiss
