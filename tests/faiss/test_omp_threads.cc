/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include <polaris/faiss/utils.h>

TEST(Threading, openmp) {
    EXPECT_TRUE(polaris::check_openmp());
}
