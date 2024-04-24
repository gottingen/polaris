# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set(FAISS_SRC
        auto_tune.cc
        clustering.cc
        ivf_lib.cc
        index.cc
        index_2layer.cc
        index_additive_quantizer.cc
        index_binary.cc
        index_binary_flat.cc
        index_binary_from_float.cc
        index_binary_hnsw.cc
        index_binary_hash.cc
        index_binary_ivf.cc
        index_flat.cc
        index_flat_codes.cc
        index_hnsw.cc
        index_id_map.cc
        index_ivf.cc
        index_ivf_additive_quantizer.cc
        index_ivf_flat.cc
        index_ivf_pq.cc
        index_ivf_fast_scan.cc
        index_ivf_additive_quantizer_fast_scan.cc
        index_ivf_pq_fast_scan.cc
        index_ivf_pqr.cc
        index_ivf_spectral_hash.cc
        index_lsh.cc
        index_nn_descent.cc
        index_lattice.cc
        index_nsg.cc
        index_pq.cc
        index_fast_scan.cc
        index_additive_quantizer_fast_scan.cc
        index_ivf_independent_quantizer.cc
        index_pq_fast_scan.cc
        index_pre_transform.cc
        index_refine.cc
        index_replicas.cc
        index_rowwise_minmax.cc
        index_scalar_quantizer.cc
        index_shards.cc
        index_shards_ivf.cc
        matrix_stats.cc
        meta_indexes.cc
        vector_transform.cc
        clone_index.cc
        index_factory.cc
        impl/aux_index_structures.cc
        impl/code_packer.cc
        impl/id_selector.cc
        impl/faiss_exception.cc
        impl/hnsw.cc
        impl/nsg.cc
        impl/polysemous_training.cc
        impl/ProductQuantizer.cc
        impl/additive_quantizer.cc
        impl/residual_quantizer.cc
        impl/local_search_quantizer.cc
        impl/product_additive_quantizer.cc
        impl/scalar_quantizer.cc
        impl/index_read.cc
        impl/index_write.cc
        impl/io.cc
        impl/kmeans1d.cc
        impl/lattice_Zn.cc
        impl/pq4_fast_scan.cc
        impl/pq4_fast_scan_search_1.cc
        impl/pq4_fast_scan_search_qbs.cc
        impl/residual_quantizer_encode_steps.cc
        impl/io.cc
        impl/lattice_Zn.cc
        impl/nn_descent.cc
        invlists/BlockInvertedLists.cc
        invlists/DirectMap.cc
        invlists/InvertedLists.cc
        invlists/InvertedListsIOHook.cc
        utils/heap.cc
        utils/worker_thread.cc
        utils/distances.cc
        utils/distances_simd.cc
        utils/extra_distances.cc
        utils/hamming.cc
        utils/partitioning.cc
        utils/quantize_lut.cc
        utils/random.cc
        utils/sorting.cc
        utils/utils.cc
        utils/distances_fused/avx512.cc
        utils/distances_fused/distances_fused.cc
        utils/distances_fused/simdlib_based.cc
)

set(FAISS_HEADERS
        auto_tune.h
        clustering.h
        ivf_lib.h
        index.h
        index_2layer.h
        index_additive_quantizer.h
        index_binary.h
        index_binary_flat.h
        index_binary_from_float.h
        index_binary_hnsw.h
        index_binary_hash.h
        index_binary_ivf.h
        index_flat.h
        index_flat_codes.h
        index_hnsw.h
        index_id_map.h
        index_ivf.h
        index_ivf_additive_quantizer.h
        index_ivf_independent_quantizer.h
        index_ivf_flat.h
        index_ivf_pq.h
        index_ivf_fast_scan.h
        index_ivf_additive_quantizer_fast_scan.h
        index_ivf_pq_fast_scan.h
        index_ivf_pqr.h
        index_ivf_spectral_hash.h
        index_lsh.h
        index_lattice.h
        index_nn_descent.h
        index_nsg.h
        index_pq.h
        index_fast_scan.h
        index_additive_quantizer_fast_scan.h
        index_pq_fast_scan.h
        index_pre_transform.h
        index_refine.h
        index_replicas.h
        index_rowwise_minmax.h
        index_scalar_quantizer.h
        index_shards.h
        index_shards_ivf.h
        matrix_stats.h
        meta_indexes.h
        metric_type.h
        vector_transform.h
        clone_index.h
        index_factory.h
        index_io.h
        impl/additive_quantizer.h
        impl/aux_index_structures.h
        impl/code_packer.h
        impl/id_selector.h
        impl/distance_computer.h
        impl/faiss_assert.h
        impl/faiss_exception.h
        impl/hnsw.h
        impl/local_search_quantizer.h
        impl/product_additive_quantizer.h
        impl/lookup_table_scaler.h
        impl/nn_descent.h
        impl/nsg.h
        impl/polysemous_training.h
        impl/product_quantizer-inl.h
        impl/ProductQuantizer.h
        impl/quantizer.h
        impl/residual_quantizer.h
        impl/result_handler.h
        impl/scalar_quantizer.h
        impl/threaded_index-inl.h
        impl/threaded_index.h
        impl/io.h
        impl/io_macros.h
        impl/kmeans1d.h
        impl/lattice_Zn.h
        impl/platform_macros.h
        impl/pq4_fast_scan.h
        impl/residual_quantizer_encode_steps.h
        impl/simd_result_handlers.h
        impl/code_distance/code_distance.h
        impl/code_distance/code_distance-generic.h
        impl/code_distance/code_distance-avx2.h
        invlists/BlockInvertedLists.h
        invlists/DirectMap.h
        invlists/InvertedLists.h
        invlists/InvertedListsIOHook.h
        utils/aligned_table.h
        utils/heap.h
        utils/worker_thread.h
        utils/distances.h
        utils/extra_distances-inl.h
        utils/extra_distances.h
        utils/fp16-fp16c.h
        utils/fp16-inl.h
        utils/fp16-arm.h
        utils/fp16.h
        utils/hamming-inl.h
        utils/hamming.h
        utils/ordered_key_value.h
        utils/partitioning.h
        utils/prefetch.h
        utils/quantize_lut.h
        utils/random.h
        utils/sorting.h
        utils/simdlib.h
        utils/simdlib_avx2.h
        utils/simdlib_emulated.h
        utils/simdlib_neon.h
        utils/utils.h
        utils/distances_fused/avx512.h
        utils/distances_fused/distances_fused.h
        utils/distances_fused/simdlib_based.h
        utils/approx_topk/approx_topk.h
        utils/approx_topk/avx2-inl.h
        utils/approx_topk/generic.h
        utils/approx_topk/mode.h
        utils/approx_topk_hamming/approx_topk_hamming.h
        utils/transpose/transpose-avx2-inl.h
        utils/hamming_distance/common.h
        utils/hamming_distance/generic-inl.h
        utils/hamming_distance/hamdis-inl.h
        utils/hamming_distance/neon-inl.h
        utils/hamming_distance/avx2-inl.h
)

if (NOT WIN32)
    list(APPEND FAISS_SRC invlists/OnDiskInvertedLists.cc)
    list(APPEND FAISS_HEADERS invlists/OnDiskInvertedLists.h)
endif ()

# Export FAISS_HEADERS variable to parent scope.
set(FAISS_HEADERS ${FAISS_HEADERS} PARENT_SCOPE)

add_library(polaris ${FAISS_SRC})

add_library(polaris_avx2 ${FAISS_SRC})
if (NOT FAISS_OPT_LEVEL STREQUAL "avx2" AND NOT FAISS_OPT_LEVEL STREQUAL "avx512")
    set_target_properties(polaris_avx2 PROPERTIES EXCLUDE_FROM_ALL TRUE)
endif ()
if (NOT WIN32)
    target_compile_options(polaris_avx2 PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-mavx2 -mfma -mf16c -mpopcnt>)
else ()
    # MSVC enables FMA with /arch:AVX2; no separate flags for F16C, POPCNT
    # Ref. FMA (under /arch:AVX2): https://docs.microsoft.com/en-us/cpp/build/reference/arch-x64
    # Ref. F16C (2nd paragraph): https://walbourn.github.io/directxmath-avx2/
    # Ref. POPCNT: https://docs.microsoft.com/en-us/cpp/intrinsics/popcnt16-popcnt-popcnt64
    target_compile_options(polaris_avx2 PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/arch:AVX2>)
    # we need bigobj for the swig wrapper
    add_compile_options(/bigobj)
endif ()

add_library(faiss_avx512 ${FAISS_SRC})
if (NOT FAISS_OPT_LEVEL STREQUAL "avx512")
    set_target_properties(faiss_avx512 PROPERTIES EXCLUDE_FROM_ALL TRUE)
endif ()
if (NOT WIN32)
    # All modern CPUs support F, CD, VL, DQ, BW extensions.
    # Ref: https://en.wikipedia.org/wiki/AVX512
    target_compile_options(faiss_avx512 PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-mavx2 -mfma -mf16c -mavx512f -mavx512cd -mavx512vl -mavx512dq -mavx512bw -mpopcnt>)
else ()
    target_compile_options(faiss_avx512 PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/arch:AVX512>)
    # we need bigobj for the swig wrapper
    add_compile_options(/bigobj)
endif ()

# Handle `#include <polaris/foo.h>`.
target_include_directories(polaris PUBLIC
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}>)
# Handle `#include <polaris/foo.h>`.
target_include_directories(polaris_avx2 PUBLIC
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}>)
# Handle `#include <polaris/foo.h>`.
target_include_directories(faiss_avx512 PUBLIC
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}>)

set_target_properties(polaris PROPERTIES
        POSITION_INDEPENDENT_CODE ON
        WINDOWS_EXPORT_ALL_SYMBOLS ON
)
set_target_properties(polaris_avx2 PROPERTIES
        POSITION_INDEPENDENT_CODE ON
        WINDOWS_EXPORT_ALL_SYMBOLS ON
)
set_target_properties(faiss_avx512 PROPERTIES
        POSITION_INDEPENDENT_CODE ON
        WINDOWS_EXPORT_ALL_SYMBOLS ON
)

if (WIN32)
    target_compile_definitions(polaris PRIVATE FAISS_MAIN_LIB)
    target_compile_definitions(polaris_avx2 PRIVATE FAISS_MAIN_LIB)
    target_compile_definitions(faiss_avx512 PRIVATE FAISS_MAIN_LIB)
endif ()

target_compile_definitions(polaris PRIVATE FINTEGER=int)
target_compile_definitions(polaris_avx2 PRIVATE FINTEGER=int)
target_compile_definitions(faiss_avx512 PRIVATE FINTEGER=int)

find_package(OpenMP REQUIRED)
target_link_libraries(polaris PRIVATE OpenMP::OpenMP_CXX)
target_link_libraries(polaris_avx2 PRIVATE OpenMP::OpenMP_CXX)
target_link_libraries(faiss_avx512 PRIVATE OpenMP::OpenMP_CXX)

find_package(MKL)
if (MKL_FOUND)
    target_link_libraries(polaris PRIVATE ${MKL_LIBRARIES})
    target_link_libraries(polaris_avx2 PRIVATE ${MKL_LIBRARIES})
    target_link_libraries(faiss_avx512 PRIVATE ${MKL_LIBRARIES})
else ()
    find_package(BLAS REQUIRED)
    target_link_libraries(polaris PRIVATE ${BLAS_LIBRARIES})
    target_link_libraries(polaris_avx2 PRIVATE ${BLAS_LIBRARIES})
    target_link_libraries(faiss_avx512 PRIVATE ${BLAS_LIBRARIES})

    find_package(LAPACK REQUIRED)
    target_link_libraries(polaris PRIVATE ${LAPACK_LIBRARIES})
    target_link_libraries(polaris_avx2 PRIVATE ${LAPACK_LIBRARIES})
    target_link_libraries(faiss_avx512 PRIVATE ${LAPACK_LIBRARIES})
endif ()

install(TARGETS polaris
        EXPORT polaris-targets
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)
if (FAISS_OPT_LEVEL STREQUAL "avx2")
    install(TARGETS polaris_avx2
            EXPORT polaris-targets
            ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
            LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    )
endif ()
if (FAISS_OPT_LEVEL STREQUAL "avx512")
    install(TARGETS polaris_avx2 polaris_avx512
            EXPORT polaris-targets
            ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
            LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    )
endif ()

foreach (header ${FAISS_HEADERS})
    get_filename_component(dir ${header} DIRECTORY)
    install(FILES ${header}
            DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/polaris/${dir}
    )
endforeach ()

include(CMakePackageConfigHelpers)
write_basic_package_version_file(
        "${PROJECT_BINARY_DIR}/cmake/polaris-config-version.cmake"
        VERSION ${CMAKE_PROJECT_VERSION}
        COMPATIBILITY AnyNewerVersion
)

configure_file(${PROJECT_SOURCE_DIR}/cmake/polaris-config.cmake.in
        ${PROJECT_BINARY_DIR}/cmake/polaris-config.cmake
        COPYONLY
)
install(FILES ${PROJECT_BINARY_DIR}/cmake/polaris-config.cmake
        ${PROJECT_BINARY_DIR}/cmake/polaris-config-version.cmake
        DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/polaris
)

install(EXPORT polaris-targets
        DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/polaris
)
