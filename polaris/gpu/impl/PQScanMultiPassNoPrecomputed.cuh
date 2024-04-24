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
#include <polaris/gpu/utils/DeviceVector.cuh>
#include <polaris/gpu/utils/Tensor.cuh>

namespace faiss {
namespace gpu {

class GpuResources;

template <typename CentroidT>
void runPQScanMultiPassNoPrecomputed(
        Tensor<float, 2, true>& queries,
        Tensor<CentroidT, 2, true>& centroids,
        Tensor<float, 3, true>& pqCentroidsInnermostCode,
        Tensor<float, 2, true>& coarseDistances,
        Tensor<idx_t, 2, true>& coarseIndices,
        bool useFloat16Lookup,
        bool useMMCodeDistance,
        bool interleavedCodeLayout,
        int bitsPerSubQuantizer,
        int numSubQuantizers,
        int numSubQuantizerCodes,
        DeviceVector<void*>& listCodes,
        DeviceVector<void*>& listIndices,
        IndicesOptions indicesOptions,
        DeviceVector<idx_t>& listLengths,
        idx_t maxListLength,
        int k,
        faiss::MetricType metric,
        // output
        Tensor<float, 2, true>& outDistances,
        // output
        Tensor<idx_t, 2, true>& outIndices,
        GpuResources* res);

} // namespace gpu
} // namespace faiss

#include <polaris/gpu/impl/PQScanMultiPassNoPrecomputed-inl.cuh>
