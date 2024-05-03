/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Copyright 2004-present Facebook. All Rights Reserved.
// -*- c++ -*-

#include <polaris/capi/utils/distances_c.h>
#include <polaris/faiss/utils/distances.h>
#include <cstdio>

void faiss_pairwise_L2sqr(
        int64_t d,
        int64_t nq,
        const float* xq,
        int64_t nb,
        const float* xb,
        float* dis,
        int64_t ldq,
        int64_t ldb,
        int64_t ldd) {
    polaris::pairwise_L2sqr(d, nq, xq, nb, xb, dis, ldq, ldb, ldd);
}

void faiss_pairwise_L2sqr_with_defaults(
        int64_t d,
        int64_t nq,
        const float* xq,
        int64_t nb,
        const float* xb,
        float* dis) {
    polaris::pairwise_L2sqr(d, nq, xq, nb, xb, dis);
}

void faiss_fvec_inner_products_ny(
        float* ip,
        const float* x,
        const float* y,
        size_t d,
        size_t ny) {
    polaris::fvec_inner_products_ny(ip, x, y, d, ny);
}

void faiss_fvec_L2sqr_ny(
        float* dis,
        const float* x,
        const float* y,
        size_t d,
        size_t ny) {
    polaris::fvec_L2sqr_ny(dis, x, y, d, ny);
}

float faiss_fvec_norm_L2sqr(const float* x, size_t d) {
    return polaris::fvec_norm_L2sqr(x, d);
}

void faiss_fvec_norms_L2(float* norms, const float* x, size_t d, size_t nx) {
    polaris::fvec_norms_L2(norms, x, d, nx);
}

void faiss_fvec_norms_L2sqr(float* norms, const float* x, size_t d, size_t nx) {
    polaris::fvec_norms_L2sqr(norms, x, d, nx);
}

void faiss_fvec_renorm_L2(size_t d, size_t nx, float* x) {
    polaris::fvec_renorm_L2(d, nx, x);
}

void faiss_set_distance_compute_blas_threshold(int value) {
    polaris::distance_compute_blas_threshold = value;
}

int faiss_get_distance_compute_blas_threshold() {
    return polaris::distance_compute_blas_threshold;
}

void faiss_set_distance_compute_blas_query_bs(int value) {
    polaris::distance_compute_blas_query_bs = value;
}

int faiss_get_distance_compute_blas_query_bs() {
    return polaris::distance_compute_blas_query_bs;
}

void faiss_set_distance_compute_blas_database_bs(int value) {
    polaris::distance_compute_blas_database_bs = value;
}

int faiss_get_distance_compute_blas_database_bs() {
    return polaris::distance_compute_blas_database_bs;
}

void faiss_set_distance_compute_min_k_reservoir(int value) {
    polaris::distance_compute_min_k_reservoir = value;
}

int faiss_get_distance_compute_min_k_reservoir() {
    return polaris::distance_compute_min_k_reservoir;
}
