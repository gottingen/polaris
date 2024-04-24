/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <polaris/index_binary_flat.h>
#include <polaris/gpu/GpuIndexBinaryFlat.h>
#include <polaris/gpu/StandardGpuResources.h>
#include <polaris/gpu/impl/IndexUtils.h>
#include <polaris/gpu/test/TestUtils.h>
#include <polaris/gpu/utils/DeviceUtils.h>
#include <polaris/utils/random.h>
#include <polaris/utils/utils.h>
#include <gtest/gtest.h>
#include <sstream>
#include <vector>

void compareBinaryDist(
        const std::vector<int>& cpuDist,
        const std::vector<polaris::idx_t>& cpuLabels,
        const std::vector<int>& gpuDist,
        const std::vector<polaris::idx_t>& gpuLabels,
        int numQuery,
        int k) {
    for (int i = 0; i < numQuery; ++i) {
        // The index order can be permuted within a group that has the same
        // distance, since this is based on the order in which the algorithm
        // encounters the values. The last set of equivalent distances seen in
        // the min-k might be truncated, so we can't check that set, but all
        // others we can check.
        std::set<polaris::idx_t> cpuLabelSet;
        std::set<polaris::idx_t> gpuLabelSet;

        int curDist = -1;

        for (int j = 0; j < k; ++j) {
            int idx = i * k + j;

            if (curDist == -1) {
                curDist = cpuDist[idx];
            }

            if (curDist != cpuDist[idx]) {
                // Distances must be monotonically increasing
                EXPECT_LT(curDist, cpuDist[idx]);

                // This is a new set of distances
                EXPECT_EQ(cpuLabelSet, gpuLabelSet);
                curDist = cpuDist[idx];
                cpuLabelSet.clear();
                gpuLabelSet.clear();
            }

            cpuLabelSet.insert(cpuLabels[idx]);
            gpuLabelSet.insert(gpuLabels[idx]);

            // Because the distances are reproducible, they must be exactly the
            // same
            EXPECT_EQ(cpuDist[idx], gpuDist[idx]);
        }
    }
}

template <int DimMultiple>
void testGpuIndexBinaryFlat(int kOverride = -1) {
    polaris::gpu::StandardGpuResources res;
    res.noTempMemory();

    polaris::gpu::GpuIndexBinaryFlatConfig config;
    config.device = polaris::gpu::randVal(0, polaris::gpu::getNumDevices() - 1);

    // multiples of 8 and multiples of 32 use different implementations
    int dims = polaris::gpu::randVal(1, 20) * DimMultiple;
    polaris::gpu::GpuIndexBinaryFlat gpuIndex(&res, dims, config);

    polaris::IndexBinaryFlat cpuIndex(dims);

    int k = kOverride > 0
            ? kOverride
            : polaris::gpu::randVal(1, polaris::gpu::getMaxKSelection());
    int numVecs = polaris::gpu::randVal(k + 1, 20000);
    int numQuery = polaris::gpu::randVal(1, 1000);

    auto data = polaris::gpu::randBinaryVecs(numVecs, dims);
    gpuIndex.add(numVecs, data.data());
    cpuIndex.add(numVecs, data.data());

    auto query = polaris::gpu::randBinaryVecs(numQuery, dims);

    std::vector<int> cpuDist(numQuery * k);
    std::vector<polaris::idx_t> cpuLabels(numQuery * k);

    cpuIndex.search(
            numQuery, query.data(), k, cpuDist.data(), cpuLabels.data());

    std::vector<int> gpuDist(numQuery * k);
    std::vector<polaris::idx_t> gpuLabels(numQuery * k);

    gpuIndex.search(
            numQuery, query.data(), k, gpuDist.data(), gpuLabels.data());

    compareBinaryDist(cpuDist, cpuLabels, gpuDist, gpuLabels, numQuery, k);
}

TEST(TestGpuIndexBinaryFlat, Test8) {
    for (int tries = 0; tries < 4; ++tries) {
        testGpuIndexBinaryFlat<8>();
    }
}

TEST(TestGpuIndexBinaryFlat, Test32) {
    for (int tries = 0; tries < 4; ++tries) {
        testGpuIndexBinaryFlat<32>();
    }
}

TEST(TestGpuIndexBinaryFlat, LargeIndex) {
    // Construct on a random device to test multi-device, if we have
    // multiple devices
    int device = polaris::gpu::randVal(0, polaris::gpu::getNumDevices() - 1);

    polaris::gpu::StandardGpuResources res;
    res.noTempMemory();

    // Skip this device if we do not have sufficient memory
    constexpr size_t kMem = size_t(8) * 1024 * 1024 * 1024;

    if (polaris::gpu::getFreeMemory(device) < kMem) {
        std::cerr << "TestGpuIndexFlat.LargeIndex: skipping due "
                     "to insufficient device memory\n";
        return;
    }

    std::cerr << "Running LargeIndex test\n";

    polaris::gpu::GpuIndexBinaryFlatConfig config;
    config.device = device;

    int dims = 1250 * 8;
    polaris::gpu::GpuIndexBinaryFlat gpuIndex(&res, dims, config);

    polaris::IndexBinaryFlat cpuIndex(dims);

    int k = 10;
    int nb = 4000000;
    int nq = 10;

    auto xb = polaris::gpu::randBinaryVecs(nb, dims);
    auto xq = polaris::gpu::randBinaryVecs(nq, dims);
    gpuIndex.add(nb, xb.data());
    cpuIndex.add(nb, xb.data());

    std::vector<int> cpuDist(nq * k);
    std::vector<polaris::idx_t> cpuLabels(nq * k);

    cpuIndex.search(nq, xq.data(), k, cpuDist.data(), cpuLabels.data());

    std::vector<int> gpuDist(nq * k);
    std::vector<polaris::idx_t> gpuLabels(nq * k);

    gpuIndex.search(nq, xq.data(), k, gpuDist.data(), gpuLabels.data());

    compareBinaryDist(cpuDist, cpuLabels, gpuDist, gpuLabels, nq, k);
}

TEST(TestGpuIndexBinaryFlat, Reconstruct) {
    int n = 1000;
    std::vector<uint8_t> xb(8 * n);
    polaris::byte_rand(xb.data(), xb.size(), 123);
    std::unique_ptr<polaris::IndexBinaryFlat> index(
            new polaris::IndexBinaryFlat(64));
    index->add(n, xb.data());

    std::vector<uint8_t> xb3(8 * n);
    index->reconstruct_n(0, index->ntotal, xb3.data());
    EXPECT_EQ(xb, xb3);

    polaris::gpu::StandardGpuResources res;
    res.noTempMemory();

    std::unique_ptr<polaris::gpu::GpuIndexBinaryFlat> index2(
            new polaris::gpu::GpuIndexBinaryFlat(&res, index.get()));

    std::vector<uint8_t> xb2(8 * n);

    index2->reconstruct_n(0, index->ntotal, xb2.data());
    EXPECT_EQ(xb2, xb3);
}

int main(int argc, char** argv) {
    testing::InitGoogleTest(&argc, argv);

    // just run with a fixed test seed
    polaris::gpu::setTestSeed(100);

    return RUN_ALL_TESTS();
}
