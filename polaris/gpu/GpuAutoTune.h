/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <polaris/auto_tune.h>
#include <polaris/index.h>

namespace polaris {
namespace gpu {

/// parameter space and setters for GPU indexes
struct GpuParameterSpace : polaris::ParameterSpace {
    /// initialize with reasonable parameters for the index
    void initialize(const polaris::Index* index) override;

    /// set a combination of parameters on an index
    void set_index_parameter(
            polaris::Index* index,
            const std::string& name,
            double val) const override;
};

} // namespace gpu
} // namespace polaris
