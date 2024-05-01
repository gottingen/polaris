/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <omp.h>
#include <unistd.h>
#include <memory>

#include <polaris/index/ivf_lib.h>
#include <polaris/index/index_ivf.h>
#include <polaris/internal/id_selector.h>
#include <polaris/index/index_factory.h>
#include <polaris/core/index_io.h>
#include <polaris/core/random.h>
#include <polaris/core/utils.h>

/************************
 * This benchmark attempts to measure the runtime overhead to use an IDSelector
 * over doing an unconditional sequential scan. Unfortunately the results of the
 * benchmark also depend a lot on the parallel_mode and the way
 * search_with_parameters works.
 */

int main() {
    using idx_t = polaris::idx_t;
    int d = 64;
    size_t nb = 1024 * 1024;
    size_t nq = 512 * 16;
    size_t k = 10;
    std::vector<float> data((nb + nq) * d);
    float* xb = data.data();
    float* xq = data.data() + nb * d;
    polaris::rand_smooth_vectors(nb + nq, d, data.data(), 1234);

    std::unique_ptr<polaris::Index> index;
    // const char *index_key = "IVF1024,Flat";
    const char* index_key = "IVF1024,SQ8";
    printf("index_key=%s\n", index_key);
    std::string stored_name =
            std::string("/tmp/bench_ivf_selector_") + index_key + ".faissindex";

    if (access(stored_name.c_str(), F_OK) != 0) {
        printf("creating index\n");
        index.reset(polaris::index_factory(d, index_key));

        double t0 = polaris::getmillisecs();
        index->train(nb, xb);
        double t1 = polaris::getmillisecs();
        index->add(nb, xb);
        double t2 = polaris::getmillisecs();
        printf("Write %s\n", stored_name.c_str());
        polaris::write_index(index.get(), stored_name.c_str());
    } else {
        printf("Read %s\n", stored_name.c_str());
        index.reset(polaris::read_index(stored_name.c_str()));
    }
    polaris::IndexIVF* index_ivf = static_cast<polaris::IndexIVF*>(index.get());
    index->verbose = true;

    for (int tt = 0; tt < 3; tt++) {
        if (tt == 1) {
            index_ivf->parallel_mode = 3;
        } else {
            index_ivf->parallel_mode = 0;
        }

        if (tt == 2) {
            printf("set single thread\n");
            omp_set_num_threads(1);
        }
        printf("parallel_mode=%d\n", index_ivf->parallel_mode);

        std::vector<float> D1(nq * k);
        std::vector<idx_t> I1(nq * k);
        {
            double t2 = polaris::getmillisecs();
            index->search(nq, xq, k, D1.data(), I1.data());
            double t3 = polaris::getmillisecs();

            printf("search time, no selector: %.3f ms\n", t3 - t2);
        }

        std::vector<float> D2(nq * k);
        std::vector<idx_t> I2(nq * k);
        {
            double t2 = polaris::getmillisecs();
            polaris::IVFSearchParameters params;

            polaris::ivflib::search_with_parameters(
                    index.get(), nq, xq, k, D2.data(), I2.data(), &params);
            double t3 = polaris::getmillisecs();
            printf("search time with nullptr selector: %.3f ms\n", t3 - t2);
        }
        POLARIS_THROW_IF_NOT(I1 == I2);
        POLARIS_THROW_IF_NOT(D1 == D2);

        {
            double t2 = polaris::getmillisecs();
            polaris::IVFSearchParameters params;
            polaris::IDSelectorAll sel;
            params.sel = &sel;

            polaris::ivflib::search_with_parameters(
                    index.get(), nq, xq, k, D2.data(), I2.data(), &params);
            double t3 = polaris::getmillisecs();
            printf("search time with selector: %.3f ms\n", t3 - t2);
        }
        POLARIS_THROW_IF_NOT(I1 == I2);
        POLARIS_THROW_IF_NOT(D1 == D2);

        std::vector<float> D3(nq * k);
        std::vector<idx_t> I3(nq * k);
        {
            int nt = omp_get_max_threads();
            double t2 = polaris::getmillisecs();
            polaris::IVFSearchParameters params;

#pragma omp parallel for if (nt > 1)
            for (idx_t slice = 0; slice < nt; slice++) {
                idx_t i0 = nq * slice / nt;
                idx_t i1 = nq * (slice + 1) / nt;
                if (i1 > i0) {
                    polaris::ivflib::search_with_parameters(
                            index.get(),
                            i1 - i0,
                            xq + i0 * d,
                            k,
                            D3.data() + i0 * k,
                            I3.data() + i0 * k,
                            &params);
                }
            }
            double t3 = polaris::getmillisecs();
            printf("search time with null selector + manual parallel: %.3f ms\n",
                   t3 - t2);
        }
        POLARIS_THROW_IF_NOT(I1 == I3);
        POLARIS_THROW_IF_NOT(D1 == D3);
    }

    return 0;
}
