/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <polaris/IndexFlat.h>
#include <polaris/gpu/GpuIndexFlat.h>
#include <polaris/gpu/StandardGpuResources.h>
#include <polaris/gpu/test/TestUtils.h>
#include <polaris/gpu/utils/DeviceUtils.h>
#include <gtest/gtest.h>

// Test to see if we can recover after attempting to allocate too much GPU
// memory
TEST(TestGpuMemoryException, AddException) {
    size_t numBrokenAdd = std::numeric_limits<int>::max();
    size_t numRealAdd = 10000;
    size_t devFree = 0;
    size_t devTotal = 0;

    CUDA_VERIFY(cudaMemGetInfo(&devFree, &devTotal));

    // Figure out the dimensionality needed to get at least greater than
    // devTotal
    size_t brokenAddDims = ((devTotal / sizeof(float)) / numBrokenAdd) + 1;
    size_t realAddDims = 128;

    polaris::gpu::StandardGpuResources res;

    polaris::gpu::GpuIndexFlatConfig config;
    config.device = polaris::gpu::randVal(0, polaris::gpu::getNumDevices() - 1);
    config.use_raft = false;

    polaris::gpu::GpuIndexFlatL2 gpuIndexL2Broken(
            &res, (int)brokenAddDims, config);
    polaris::gpu::GpuIndexFlatL2 gpuIndexL2(&res, (int)realAddDims, config);
    polaris::IndexFlatL2 cpuIndex((int)realAddDims);

    // Should throw on attempting to allocate too much data
    {
        // allocate memory without initialization
        auto vecs = std::unique_ptr<float[]>(
                new float[numBrokenAdd * brokenAddDims]);
        EXPECT_THROW(
                gpuIndexL2Broken.add(numBrokenAdd, vecs.get()),
                polaris::FaissException);
    }

    // Should be able to add a smaller set of data now
    {
        auto vecs = polaris::gpu::randVecs(numRealAdd, realAddDims);
        EXPECT_NO_THROW(gpuIndexL2.add(numRealAdd, vecs.data()));
        cpuIndex.add(numRealAdd, vecs.data());
    }

    // Should throw on attempting to allocate too much data
    {
        // allocate memory without initialization
        auto vecs = std::unique_ptr<float[]>(
                new float[numBrokenAdd * brokenAddDims]);
        EXPECT_THROW(
                gpuIndexL2Broken.add(numBrokenAdd, vecs.get()),
                polaris::FaissException);
    }

    // Should be able to query results from what we had before
    {
        size_t numQuery = 10;
        auto vecs = polaris::gpu::randVecs(numQuery, realAddDims);
        EXPECT_NO_THROW(compareIndices(
                vecs,
                cpuIndex,
                gpuIndexL2,
                numQuery,
                realAddDims,
                50,
                "",
                6e-3f,
                0.1f,
                0.015f));
    }
}

int main(int argc, char** argv) {
    testing::InitGoogleTest(&argc, argv);

    // just run with a fixed test seed
    polaris::gpu::setTestSeed(100);

    return RUN_ALL_TESTS();
}
