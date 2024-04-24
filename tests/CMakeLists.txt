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

add_executable(polaris_test ${FAISS_TEST_SRC})

if(NOT FAISS_OPT_LEVEL STREQUAL "avx2" AND NOT FAISS_OPT_LEVEL STREQUAL "avx512")
  target_link_libraries(polaris_test PRIVATE polaris)
endif()

if(FAISS_OPT_LEVEL STREQUAL "avx2")
  if(NOT WIN32)
    target_compile_options(polaris_test PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-mavx2 -mfma>)
  else()
    target_compile_options(polaris_test PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/arch:AVX2>)
  endif()
  target_link_libraries(polaris_test PRIVATE polaris_avx2)
endif()

if(FAISS_OPT_LEVEL STREQUAL "avx512")
  if(NOT WIN32)
    target_compile_options(polaris_test PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-mavx2 -mfma -mavx512f -mavx512f -mavx512cd -mavx512vl -mavx512dq -mavx512bw>)
  else()
    target_compile_options(polaris_test PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/arch:AVX512>)
  endif()
  target_link_libraries(polaris_test PRIVATE polaris_avx2)
endif()

include(FetchContent)
FetchContent_Declare(googletest
  URL "https://github.com/google/googletest/archive/release-1.12.1.tar.gz")
set(BUILD_GMOCK CACHE BOOL OFF)
set(INSTALL_GTEST CACHE BOOL OFF)
FetchContent_MakeAvailable(googletest)

find_package(OpenMP REQUIRED)

target_link_libraries(polaris_test PRIVATE
  OpenMP::OpenMP_CXX
  gtest_main
  $<$<BOOL:${FAISS_ENABLE_RAFT}>:raft::raft>
)

# Defines `gtest_discover_tests()`.
include(GoogleTest)
gtest_discover_tests(polaris_test)
