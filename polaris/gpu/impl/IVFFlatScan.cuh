/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <polaris/Index.h>
#include <polaris/MetricType.h>
#include <polaris/gpu/GpuIndicesOptions.h>
#include <polaris/gpu/impl/GpuScalarQuantizer.cuh>
#include <polaris/gpu/utils/DeviceVector.cuh>
#include <polaris/gpu/utils/Tensor.cuh>

namespace polaris {
namespace gpu {

class GpuResources;

void runIVFFlatScan(
        Tensor<float, 2, true>& queries,
        Tensor<idx_t, 2, true>& listIds,
        DeviceVector<void*>& listData,
        DeviceVector<void*>& listIndices,
        IndicesOptions indicesOptions,
        DeviceVector<idx_t>& listLengths,
        idx_t maxListLength,
        int k,
        polaris::MetricType metric,
        bool useResidual,
        Tensor<float, 3, true>& residualBase,
        GpuScalarQuantizer* scalarQ,
        // output
        Tensor<float, 2, true>& outDistances,
        // output
        Tensor<idx_t, 2, true>& outIndices,
        GpuResources* res);

} // namespace gpu
} // namespace polaris
