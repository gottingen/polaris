// Copyright 2023 The Elastic-AI Authors.
// part of Elastic AI Search
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
//

#include <polaris/internal/platform_macros.h>
#include <polaris/distance/distance.h>
#include <polaris/distance/norm.h>
#include <polaris/distance/primitive.h>
#include <polaris/internal/polaris_assert.h>
#include <polaris/core/executor.h>
#include <collie/taskflow/algorithm/for_each.h>

namespace polaris {

    float fvec_L2sqr(const float *x, const float *y, size_t d) {
        auto al = collie::simd::is_aligned<collie::simd::best_arch>(x) &&
                  collie::simd::is_aligned<collie::simd::best_arch>(y);
        if (al) {
            return PrimL2Sqr<float, collie::simd::best_arch, collie::simd::aligned_mode>(x, y, d);
        }
        return PrimL2Sqr<float, collie::simd::best_arch, collie::simd::unaligned_mode>(x, y, d);
    }

    void fvec_L2sqr_ny(float *dis, const float *x, const float *y, size_t d, size_t ny, void *executor) {
        collie::tf::Executor *exec = static_cast<collie::tf::Executor *>(executor);
        if (exec == nullptr) {
            exec = get_global_executor();
        }
        collie::tf::Taskflow taskflow;
        auto func = [&dis, x, y, d](int i) -> void {
            dis[i] = fvec_L2sqr(x, y + i * d, d);
        };
        taskflow.for_each_index(0, static_cast<int>(ny), 1, func);
        exec->run(taskflow).wait();
    }

    size_t fvec_L2sqr_ny_nearest(float *distances_tmp_buffer, const float *x, const float *y, size_t d, size_t ny,
                                 void *executor) {
        fvec_L2sqr_ny(distances_tmp_buffer, x, y, d, ny, executor);

        size_t nearest_idx = 0;
        float min_dis = HUGE_VALF;

        for (size_t i = 0; i < ny; i++) {
            if (distances_tmp_buffer[i] < min_dis) {
                min_dis = distances_tmp_buffer[i];
                nearest_idx = i;
            }
        }

        return nearest_idx;
    }


    void fvec_L2sqr_by_idx(float *__restrict dis, const float *x, const float *y, const int64_t *__restrict ids,
                           size_t d,
                           size_t nx,
                           size_t ny,
                           void *executor) {
        collie::tf::Executor *exec = static_cast<collie::tf::Executor *>(executor);
        if (exec == nullptr) {
            exec = get_global_executor();
        }
        collie::tf::Taskflow taskflow;
        auto func = [&dis, x, y, d, ids,ny](int i) {
            const int64_t *__restrict idsj = ids + i * ny;
            const float *xj = x + i * d;
            float *__restrict disj = dis + i * ny;
            for (size_t j = 0; j < ny; j++) {
                if (idsj[j] < 0) {
                    disj[j] = INFINITY;
                } else {
                    disj[j] = fvec_L2sqr(xj, y + d * idsj[j], d);
                }
            }
        };
        taskflow.for_each_index(0, static_cast<int>(nx), 1, func);
        exec->run(taskflow).wait();
    }

    /*********************************************************
     * Autovectorized implementations
     */
    float fvec_inner_product(const float *x, const float *y, size_t d) {
        auto al = collie::simd::is_aligned<collie::simd::best_arch>(x) &&
                  collie::simd::is_aligned<collie::simd::best_arch>(y);
        if (al) {
            return PrimIP<float, collie::simd::best_arch, collie::simd::aligned_mode>(x, y, d);
        }
        return PrimIP<float, collie::simd::best_arch, collie::simd::unaligned_mode>(x, y, d);
    }

    void fvec_inner_products_ny(float *ip, const float *x, const float *y, size_t d, size_t ny, void *executor) {
        auto *exec = static_cast<collie::tf::Executor *>(executor);
        if (exec == nullptr) {
            exec = get_global_executor();
        }
        collie::tf::Taskflow taskflow;
        auto func = [&ip, x, y, d](int i) {
            ip[i] = fvec_inner_product(x, y + i * d, d);
        };
        taskflow.for_each_index(0, static_cast<int>(ny), 1, func);
        exec->run(taskflow).wait();
    }

    void fvec_inner_products_by_idx(
            float* ip,
            const float* x,
            const float* y,
            const int64_t* ids,
            size_t d,
            size_t nx,
            size_t ny,
            void *executor) {
        collie::tf::Executor *exec = static_cast<collie::tf::Executor *>(executor);
        if (exec == nullptr) {
            exec = get_global_executor();
        }
        collie::tf::Taskflow taskflow;
        auto func = [&ip, x, y, d, ids,ny](int i) {
            const int64_t *__restrict idsj = ids + i * ny;
            const float *xj = x + i * d;
            float *__restrict ipj = ip + i * ny;
            for (size_t j = 0; j < ny; j++) {
                if (idsj[j] < 0) {
                    ipj[j] = -INFINITY;
                } else {
                    ipj[j] = fvec_inner_product(xj, y + d * idsj[j], d);
                }
            }
        };
        taskflow.for_each_index(0, static_cast<int>(nx), 1, func);
    }

    float fvec_L1(const float *x, const float *y, size_t d) {
        auto al = collie::simd::is_aligned<collie::simd::best_arch>(x) &&
                  collie::simd::is_aligned<collie::simd::best_arch>(y);
        if (al) {
            return PrimL1<float, collie::simd::best_arch, collie::simd::aligned_mode>(x, y, d);
        }
        return PrimL1<float, collie::simd::best_arch, collie::simd::unaligned_mode>(x, y, d);
    }

    float fvec_norm_L2sqr(const float *x, size_t d) {
        auto al = collie::simd::is_aligned<collie::simd::best_arch>(x);
        if (al) {
            return NormL2<float, collie::simd::best_arch, collie::simd::aligned_mode>::norm_l2_sqr(x, d);
        }
        return NormL2<float, collie::simd::best_arch, collie::simd::unaligned_mode>::norm_l2_sqr(x, d);
    }

    void fvec_norms_L2sqr(float *norms, const float *x, size_t d, size_t nx, void *executor) {
        auto *exec = static_cast<collie::tf::Executor *>(executor);
        if (exec == nullptr) {
            exec = get_global_executor();
        }
        if (nx > 10000) {
            collie::tf::Taskflow taskflow;
            auto func = [&norms, x, d](int i) {
                norms[i] = fvec_norm_L2sqr(x + i * d, d);
            };
            taskflow.for_each_index(0, static_cast<int>(nx), 1, func);
            exec->run(taskflow).wait();
        } else {
            for (size_t i = 0; i < nx; i++) {
                norms[i] = fvec_norm_L2sqr(x + i * d, d);
            }
        }
    }

    void fvec_norms_L2(float *norms, const float *x, size_t d, size_t nx, void *executor) {
        fvec_norms_L2sqr(norms, x, d, nx, executor);
        for (size_t i = 0; i < nx; i++) {
            norms[i] = sqrt(norms[i]);
        }
    }

    float fvec_Linf(const float *x, const float *y, size_t d) {
        auto al = collie::simd::is_aligned<collie::simd::best_arch>(x) &&
                  collie::simd::is_aligned<collie::simd::best_arch>(y);
        if (al) {
            return PrimLinf<float, collie::simd::best_arch, collie::simd::aligned_mode>(x, y, d);
        }
        return PrimLinf<float, collie::simd::best_arch, collie::simd::unaligned_mode>(x, y, d);
    }

    void fvec_add(size_t d, const float *a, const float *b, float *c) {
        auto al = collie::simd::is_aligned<collie::simd::best_arch>(a) &&
                  collie::simd::is_aligned<collie::simd::best_arch>(b) &&
                  collie::simd::is_aligned<collie::simd::best_arch>(c);
        if (al) {
            PrimAdd<float, collie::simd::best_arch, collie::simd::aligned_mode>(d, a, b, c);
        } else {
            PrimAdd<float, collie::simd::best_arch, collie::simd::unaligned_mode>(d, a, b, c);
        }
    }

    void fvec_add(size_t d, const float *a, float b, float *c) {
        auto al = collie::simd::is_aligned<collie::simd::best_arch>(a) &&
                  collie::simd::is_aligned<collie::simd::best_arch>(c);
        if (al) {
            PrimAdd<float, collie::simd::best_arch, collie::simd::aligned_mode>(d, a, b, c);
        } else {
            PrimAdd<float, collie::simd::best_arch, collie::simd::unaligned_mode>(d, a, b, c);
        }
    }

    void fvec_madd(size_t n, const float *a, float bf, const float *b, float *c) {
        auto al = collie::simd::is_aligned<collie::simd::best_arch>(a) &&
                  collie::simd::is_aligned<collie::simd::best_arch>(b) &&
                  collie::simd::is_aligned<collie::simd::best_arch>(c);
        if (al) {
            PrimMAdd<float, collie::simd::best_arch, collie::simd::aligned_mode>(n, a, bf, b, c);
        } else {
            PrimMAdd<float, collie::simd::best_arch, collie::simd::unaligned_mode>(n, a, bf, b, c);
        }
    }

    int fvec_madd_and_argmin(size_t n, const float *a, float bf, const float *b, float *c) {
        auto al = collie::simd::is_aligned<collie::simd::best_arch>(a) &&
                  collie::simd::is_aligned<collie::simd::best_arch>(b) &&
                  collie::simd::is_aligned<collie::simd::best_arch>(c);
        if (al) {
            PrimMAdd<float, collie::simd::best_arch, collie::simd::aligned_mode>(n, a, bf, b, c);
        } else {
            PrimMAdd<float, collie::simd::best_arch, collie::simd::unaligned_mode>(n, a, bf, b, c);
        }
        float min_val = c[0];
        int min_idx = 0;
        for (size_t i = 1; i < n; i++) {
            if (c[i] < min_val) {
                min_val = c[i];
                min_idx = i;
            }
        }
        return min_idx;
    }

    void fvec_sub(size_t d, const float *a, const float *b, float *c) {
        auto al = collie::simd::is_aligned<collie::simd::best_arch>(a) &&
                  collie::simd::is_aligned<collie::simd::best_arch>(b) &&
                  collie::simd::is_aligned<collie::simd::best_arch>(c);
        if (al) {
            PrimSub<float, collie::simd::best_arch, collie::simd::aligned_mode>(d, a, b, c);
        } else {
            PrimSub<float, collie::simd::best_arch, collie::simd::unaligned_mode>(d, a, b, c);
        }
    }

    void fvec_sub(size_t d, const float *a, float b, float *c) {
        auto al = collie::simd::is_aligned<collie::simd::best_arch>(a) &&
                  collie::simd::is_aligned<collie::simd::best_arch>(c);
        if (al) {
            PrimSub<float, collie::simd::best_arch, collie::simd::aligned_mode>(d, a, b, c);
        } else {
            PrimSub<float, collie::simd::best_arch, collie::simd::unaligned_mode>(d, a, b, c);
        }
    }

}  // namespace polaris

