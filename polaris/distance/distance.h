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
#pragma once

#include <polaris/utility/platform_macros.h>
#include <cstring>
#include <polaris/core/defines.h>
#include <polaris/core/metric_type.h>

namespace polaris {

    template<typename T>
    class Distance {
    public:
        POLARIS_API Distance(polaris::MetricType dist_metric) : _distance_metric(dist_metric) {
        }

        // distance comparison function
        POLARIS_API virtual float compare(const T *a, const T *b, uint32_t length) const = 0;

        // Needed only for COSINE-BYTE and INNER_PRODUCT-BYTE
        POLARIS_API virtual float compare(const T *a, const T *b, const float normA, const float normB,
                                          uint32_t length) const;

        POLARIS_API virtual polaris::MetricType get_metric() const;


        // Invokes normalization for a single vector during search. The scratch space
        // has to be created by the caller keeping track of the fact that
        // normalization might change the dimension of the query vector.
        POLARIS_API virtual void preprocess_query(const T *query_vec, const size_t query_dim, T *scratch_query);

        // If an algorithm has a requirement that some data be aligned to a certain
        // boundary it can use this function to indicate that requirement. Currently,
        // we are setting it to 8 because that works well for AVX2. If we have AVX512
        // implementations of distance algos, they might have to set this to 16
        // (depending on how they are implemented)
        POLARIS_API virtual size_t get_required_alignment() const;

        // Providing a default implementation for the virtual destructor because we
        // don't expect most metric implementations to need it.
        POLARIS_API virtual ~Distance() = default;

    protected:
        polaris::MetricType _distance_metric;
        size_t _alignment_factor = 8;
    };

    template<typename T>
    Distance<T> *get_distance_function(MetricType m);

    float fvec_L2sqr(const float *x, const float *y, size_t d);

    /* compute ny square L2 distance between x and a set of contiguous y vectors */
    void fvec_L2sqr_ny(float *dis, const float *x, const float *y, size_t d, size_t ny, void *executor = nullptr);
    /* compute ny square L2 distance between x and a set of contiguous y vectors
       and return the index of the nearest vector.
       return 0 if ny == 0. */
    size_t fvec_L2sqr_ny_nearest(float* distances_tmp_buffer,const float* x,const float* y,size_t d,size_t ny, void *executor = nullptr);

    /** compute the squared L2 distances between x and a subset y of ny vectors
     * defined by ids
     *
     * dis(i, j) = inner_product(x(i, :), y(ids(i, j), :))
     *
     * @param dis   output array, size nx * ny
     * @param x     first-term vector, size nx * d
     * @param y     second-term vector, size (max(ids) + 1) * d
     * @param ids   ids to sample from y, size nx * ny
     */
    void fvec_L2sqr_by_idx(
            float* dis,
            const float* x,
            const float* y,
            const int64_t* ids, /* ids of y vecs */
            size_t d,
            size_t nx,
            size_t ny,
            void *executor = nullptr);

    /// inner product
    float fvec_inner_product(const float *x, const float *y, size_t d);

    /* compute the inner product between nx vectors x and one y */
    void fvec_inner_products_ny(float* ip, const float* x,const float* y,size_t d,size_t ny, void *executor = nullptr);


    /** compute the inner product between x and a subset y of ny vectors defined by
     * ids
     *
     * ip(i, j) = inner_product(x(i, :), y(ids(i, j), :))
     *
     * @param ip    output array, size nx * ny
     * @param x     first-term vector, size nx * d
     * @param y     second-term vector, size (max(ids) + 1) * d
     * @param ids   ids to sample from y, size nx * ny
     */
    void fvec_inner_products_by_idx(
            float* ip,
            const float* x,
            const float* y,
            const int64_t* ids,
            size_t d,
            size_t nx,
            size_t ny,
            void *executor = nullptr);


    /// L1 distance
    float fvec_L1(const float *x, const float *y, size_t d);

    /** squared norm of a vector */
    float fvec_norm_L2sqr(const float *x, size_t d);

    /** compute the L2 norms for a set of vectors
     *
     * @param  norms    output norms, size nx
     * @param  x        set of vectors, size nx * d
     */
    void fvec_norms_L2(float* norms, const float* x, size_t d, size_t nx, void *executor = nullptr);

    /// same as fvec_norms_L2, but computes squared norms
    void fvec_norms_L2sqr(float* norms, const float* x, size_t d, size_t nx, void *executor = nullptr);


    /// infinity distance
    float fvec_Linf(const float *x, const float *y, size_t d);

    /*********************************************************
     * Vector to vector functions
     *********************************************************/

    /** compute c := a + b for vectors
     *
     * c and a can overlap, c and b can overlap
     *
     * @param a size d
     * @param b size d
     * @param c size d
     */
    void fvec_add(size_t d, const float *a, const float *b, float *c);

    /** compute c := a + b for a, c vectors and b a scalar
     *
     * c and a can overlap
     *
     * @param a size d
     * @param c size d
     */
    void fvec_add(size_t d, const float *a, float b, float *c);

    /** compute c := a + bf * b for a, b and c tables
     *
     * @param n   size of the tables
     * @param a   size n
     * @param b   size n
     * @param c   restult table, size n
     */
    void fvec_madd(size_t n, const float* a, float bf, const float* b, float* c);
    /** same as fvec_madd, also return index of the min of the result table
     * @return    index of the min of table c
     */
    int fvec_madd_and_argmin(size_t n,const float* a,float bf,const float* b,float* c);

    /** compute c := a - b for vectors
     *
     * c and a can overlap, c and b can overlap
     *
     * @param a size d
     * @param b size d
     * @param c size d
     */
    void fvec_sub(size_t d, const float *a, const float *b, float *c);

    void fvec_sub(size_t d, const float *a, float b, float *c);
}  // namespace polaris
