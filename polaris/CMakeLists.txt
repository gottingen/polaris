#
# Copyright 2023 The titan-search Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
###########################################################################
# below are examples, after initialization, modify it to your own.
###########################################################################
###carbin_example
#[[
carbin_cc_library(
        NAMESPACE polaris
        NAME object
        SOURCES
        object.cc
        CXXOPTS
        ${CARBIN_CXX_OPTIONS}
        PLINKS
        ${CARBIN_DEPS_LINK}
        PUBLIC
)
]]

set(POLARIS_SRC
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


set(POLARIS_HEADERS
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
    list(APPEND POLARIS_SRC invlists/OnDiskInvertedLists.cc)
    list(APPEND POLARIS_HEADERS invlists/OnDiskInvertedLists.h)
endif ()


# Export POLARIS_HEADERScc variable to parent scope.
set(POLARIS_HEADERScc ${POLARIS_HEADERScc} PARENT_SCOPE)

carbin_cc_library(
        NAMESPACE polaris
        NAME polaris
        SOURCES
        ${POLARIS_SRC}
        DEFINES ${POLARIS_CXX_DEFINES}
        CXXOPTS
        ${POLARIS_CXX_NORMAL_OPTIONS}
        PLINKS
        ${POLARIS_DEPS_LINKS}
        PUBLIC
)

carbin_cc_library(
        NAMESPACE polaris
        NAME polaris_simd
        SOURCES
        ${POLARIS_SRC}
        DEFINES ${POLARIS_CXX_DEFINES}
        CXXOPTS
        ${POLARIS_CXX_OPTIONS}
        PLINKS
        ${POLARIS_DEPS_LINKS}
        PUBLIC
)

foreach (header ${POLARIS_HEADERS})
    get_filename_component(dir ${header} DIRECTORY)
    install(FILES ${header}
            DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/polaris/${dir}
    )
endforeach ()

set(POLARIS_C_SRC
        capi/auto_tune_c.cc
        capi/clustering_c.cc
        capi/index_flat_c.cc
        capi/index_ivf_flat_c.cc
        capi/index_ivf_c.cc
        capi/index_lsh_c.cc
        capi/index_pre_transform_c.cc
        capi/vector_transform_c.cc
        capi/index_shards_c.cc
        capi/index_replicas_c.cc
        capi/index_c.cc
        capi/index_binary_c.cc
        capi/index_scalar_quantizer_c.cc
        capi/meta_indexes_c.cc
        capi/clone_index_c.cc
        capi/error_impl.cc
        capi/index_factory_c.cc
        capi/index_io_c.cc
        capi/impl/auxIndex_structures_c.cc
        capi/utils/distances_c.cc
)

carbin_cc_library(
        NAMESPACE polaris
        NAME polaris_c
        SOURCES
        ${POLARIS_C_SRC}
        DEFINES ${POLARIS_CXX_DEFINES}
        CXXOPTS
        ${POLARIS_CXX_NORMAL_OPTIONS}
        PLINKS
        ${POLARIS_DEPS_LINKS} polaris::polaris
        PUBLIC
)

carbin_cc_library(
        NAMESPACE polaris
        NAME polaris_c_simd
        SOURCES
        ${POLARIS_C_SRC}
        DEFINES ${POLARIS_CXX_DEFINES}
        CXXOPTS
        ${POLARIS_CXX_OPTIONS}
        PLINKS
        ${POLARIS_DEPS_LINKS} polaris::polaris_simd
        PUBLIC
)

function(polaris_install_headers headers p)
    foreach(h ${headers})
        get_filename_component(f ${h} DIRECTORY)
        install(FILES ${h}
                DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/polaris/${p}/${f}
        )
    endforeach()
endfunction()

file(GLOB POLARIS_C_API_HEADERS
        RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
        "capi/*.h"
        "capi/impl/*.h"
        "capi/utils/*.h")

polaris_install_headers("${POLARIS_C_API_HEADERS}" c_api)
