/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <cstdio>
#include <cstdlib>
#include <random>

#include <polaris/faiss/index_flat.h>
#include <polaris/gpu/GpuAutoTune.h>
#include <polaris/gpu/GpuCloner.h>
#include <polaris/gpu/GpuIndexFlat.h>
#include <polaris/gpu/StandardGpuResources.h>
#include <polaris/gpu/utils/DeviceUtils.h>

int main() {
    int d = 64;      // dimension
    int nb = 100000; // database size
    int nq = 10000;  // nb of queries

    std::mt19937 rng;
    std::uniform_real_distribution<> distrib;

    float* xb = new float[d * nb];
    float* xq = new float[d * nq];

    for (int i = 0; i < nb; i++) {
        for (int j = 0; j < d; j++)
            xb[d * i + j] = distrib(rng);
        xb[d * i] += i / 1000.;
    }

    for (int i = 0; i < nq; i++) {
        for (int j = 0; j < d; j++)
            xq[d * i + j] = distrib(rng);
        xq[d * i] += i / 1000.;
    }

    int ngpus = polaris::gpu::getNumDevices();

    printf("Number of GPUs: %d\n", ngpus);

    std::vector<polaris::gpu::GpuResourcesProvider*> res;
    std::vector<int> devs;
    for (int i = 0; i < ngpus; i++) {
        res.push_back(new polaris::gpu::StandardGpuResources);
        devs.push_back(i);
    }

    polaris::IndexFlatL2 cpu_index(d);

    polaris::Index* gpu_index =
            polaris::gpu::index_cpu_to_gpu_multiple(res, devs, &cpu_index);

    printf("is_trained = %s\n", gpu_index->is_trained ? "true" : "false");
    gpu_index->add(nb, xb); // add vectors to the index
    printf("ntotal = %ld\n", gpu_index->ntotal);

    int k = 4;

    { // search xq
        long* I = new long[k * nq];
        float* D = new float[k * nq];

        gpu_index->search(nq, xq, k, D, I);

        // print results
        printf("I (5 first results)=\n");
        for (int i = 0; i < 5; i++) {
            for (int j = 0; j < k; j++)
                printf("%5ld ", I[i * k + j]);
            printf("\n");
        }

        printf("I (5 last results)=\n");
        for (int i = nq - 5; i < nq; i++) {
            for (int j = 0; j < k; j++)
                printf("%5ld ", I[i * k + j]);
            printf("\n");
        }

        delete[] I;
        delete[] D;
    }

    delete gpu_index;

    for (int i = 0; i < ngpus; i++) {
        delete res[i];
    }

    delete[] xb;
    delete[] xq;

    return 0;
}
