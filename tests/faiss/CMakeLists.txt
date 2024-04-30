# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set(FAISS_TEST_SRC
  test_binary_flat.cc
  test_dealloc_invlists.cc
  test_ivfpq_codec.cc
  test_ivfpq_indexing.cc
  test_lowlevel_ivf.cc
  test_ivf_index.cc
  test_merge.cc
  test_omp_threads.cc
  test_ondisk_ivf.cc
  test_pairs_decoding.cc
  test_params_override.cc
  test_pq_encoding.cc
  test_sliding_ivf.cc
  test_threaded_index.cc
  test_transfer_invlists.cc
  test_mem_leak.cc
  test_cppcontrib_sa_decode.cc
  test_cppcontrib_uintreader.cc
  test_simdlib.cc
  test_approx_topk.cc
  test_RCQ_cropping.cc
  test_distances_simd.cc
  test_heap.cc
  test_code_distance.cc
  test_hnsw.cc
  test_partitioning.cc
  test_fastscan_perf.cc
)

carbin_cc_test(
        NAME polaris_test
        MODULE run
        SOURCES ${FAISS_TEST_SRC}
        DEFINES ${POLARIS_CXX_DEFINES}
        LINKS polaris::polaris_simd ${POLARIS_DEPS_LINKS} GTest::gtest GTest::gtest_main
        CXXOPTS ${POLARIS_CXX_OPTIONS}
)

carbin_cc_test(
        NAME doubly_buffered_data_test
        MODULE run
        SOURCES doubly_buffered_data_test.cc
        DEFINES ${POLARIS_CXX_DEFINES}
        LINKS ${POLARIS_DEPS_LINKS} GTest::gtest GTest::gtest_main
        CXXOPTS ${POLARIS_CXX_OPTIONS}
)


