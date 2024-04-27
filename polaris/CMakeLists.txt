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
        core/index.cc
        core/aux_index_structures.cc
        core/heap.cc
        core/utils.cc
        core/sorting.cc
        core/matrix_stats.cc
        index/index_flat.cc
        index/index_ivf.cc
        index/index_ivf_pq.cc
        index/index_ivf_flat.cc
        index/auto_tune.cc
        index/index_ivf_pqr.cc
        index/index_hnsw.cc
        index/index_pq.cc
        index/index_lsh.cc
        index/index_nsg.cc
        index/index_binary.cc
        index/index_binary_flat.cc
        index/index_binary_hnsw.cc
        index/index_binary_ivf.cc
        index/index_factory.cc
        index/index_binary_hash.cc
        index/index_flat_codes.cc
        index/index_id_map.cc
        index/index_2layer.cc
        index/index_lattice.cc
        index/index_refine.cc
        index/index_replicas.cc
        index/index_pq_fast_scan.cc
        index/index_ivf_fast_scan.cc
        index/index_nn_descent.cc
        index/index_shards.cc
        index/index_fast_scan.cc
        index/index_ivf_pq_fast_scan.cc
        index/index_ivf_spectral_hash.cc
        index/index_shards_ivf.cc
        index/index_additive_quantizer.cc
        index/index_binary_from_float.cc
        index/index_additive_quantizer_fast_scan.cc
        index/index_ivf_additive_quantizer.cc
        index/index_ivf_additive_quantizer_fast_scan.cc
        index/index_ivf_independent_quantizer.cc
        index/index_scalar_quantizer.cc
        index/index_rowwise_minmax.cc
        index/clone_index.cc
        index/index_pre_transform.cc
        index/meta_indexes.cc
        index/ivf_lib.cc
        index/vector_transform.cc
        unified_index.cc
        impl/hnsw.cc
        impl/nsg.cc
        impl/polysemous_training.cc
        impl/product_additive_quantizer.cc
        impl/index_read.cc
        impl/index_write.cc
        impl/pq4_fast_scan.cc
        impl/pq4_fast_scan_search_1.cc
        impl/pq4_fast_scan_search_qbs.cc
        impl/residual_quantizer_encode_steps.cc
        impl/nn_descent.cc
        internal/id_selector.cc
        internal/polaris_exception.cc
        internal/kmeans1d.cc
        internal/io.cc
        internal/lattice_zn.cc
        internal/code_packer.cc
        quantization/product_quantizer.cc
        quantization/additive_quantizer.cc
        quantization/local_search_quantizer.cc
        quantization/residual_quantizer.cc
        quantization/scalar_quantizer.cc
        quantization/clustering.cc
        invlists/block_inverted_lists.cc
        invlists/direct_map.cc
        invlists/inverted_lists.cc
        invlists/inverted_lists_io_hook.cc
        utils/worker_thread.cc
        utils/distances.cc
        utils/distances_simd.cc
        utils/extra_distances.cc
        utils/hamming.cc
        utils/partitioning.cc
        utils/quantize_lut.cc
        utils/random.cc
        utils/distances_fused/avx512.cc
        utils/distances_fused/distances_fused.cc
        utils/distances_fused/simdlib_based.cc
)


set(POLARIS_HEADERS
        core/index.h
        core/aux_index_structures.h
        core/result_handler.h
        core/heap.h
        core/ordered_key_value.h
        core/utils.h
        core/sorting.h
        core/matrix_stats.h
        core/metric_type.h
        index/index_flat.h
        index/index_ivf.h
        index/index_ivf_pq.h
        index/index_ivf_flat.h
        index/auto_tune.h
        index/index_ivf_pqr.h
        index/index_io.h
        index/index_hnsw.h
        index/index_pq.h
        index/index_lsh.h
        index/index_nsg.h
        index/index_binary.h
        index/index_binary_flat.h
        index/index_binary_hnsw.h
        index/index_binary_ivf.h
        index/index_factory.h
        index/index_binary_hash.h
        index/index_flat_codes.h
        index/index_id_map.h
        index/index_2layer.h
        index/index_lattice.h
        index/index_refine.h
        index/index_replicas.h
        index/index_pq_fast_scan.h
        index/index_ivf_fast_scan.h
        index/index_nn_descent.h
        index/index_shards.h
        index/index_fast_scan.h
        index/index_ivf_pq_fast_scan.h
        index/index_ivf_spectral_hash.h
        index/index_shards_ivf.h
        index/index_additive_quantizer.h
        index/index_binary_from_float.h
        index/index_additive_quantizer_fast_scan.h
        index/index_ivf_additive_quantizer.h
        index/index_ivf_additive_quantizer_fast_scan.h
        index/index_ivf_independent_quantizer.h
        index/index_scalar_quantizer.h
        index/index_rowwise_minmax.h
        index/clone_index.h
        index/index_pre_transform.h
        index/meta_indexes.h
        index/ivf_lib.h
        index/vector_transform.h
        unified_index.h
        impl/distance_computer.h
        impl/hnsw.h
        impl/product_additive_quantizer.h
        impl/lookup_table_scaler.h
        impl/nn_descent.h
        impl/nsg.h
        impl/polysemous_training.h
        impl/threaded_index-inl.h
        impl/threaded_index.h
        impl/pq4_fast_scan.h
        impl/residual_quantizer_encode_steps.h
        impl/simd_result_handlers.h
        impl/code_distance/code_distance.h
        impl/code_distance/code_distance-generic.h
        impl/code_distance/code_distance-avx2.h
        internal/id_selector.h
        internal/platform_macros.h
        internal/polaris_exception.h
        internal/code_packer.h
        internal/io_macros.h
        internal/io.h
        internal/lattice_zn.h
        internal/kmeans1d.h
        internal/polaris_assert.h
        internal/rwlock.h
        internal/doubly_buffered_data.h
        internal/fp16-fp16c.h
        internal/fp16-inl.h
        internal/fp16-arm.h
        internal/fp16.h
        quantization/quantizer.h
        quantization/product_quantizer-inl.h
        quantization/additive_quantizer.h
        quantization/product_quantizer.h
        quantization/local_search_quantizer.h
        quantization/residual_quantizer.h
        quantization/scalar_quantizer.h
        quantization/clustering.h
        invlists/block_inverted_lists.h
        invlists/direct_map.h`
        invlists/inverted_lists.h
        invlists/inverted_lists_io_hook.h
        utils/aligned_table.h
        utils/worker_thread.h
        utils/distances.h
        utils/extra_distances-inl.h
        utils/extra_distances.h
        utils/hamming-inl.h
        utils/hamming.h
        utils/partitioning.h
        utils/prefetch.h
        utils/quantize_lut.h
        utils/random.h
        utils/simdlib.h
        utils/simdlib_avx2.h
        utils/simdlib_emulated.h
        utils/simdlib_neon.h
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
    list(APPEND POLARIS_SRC invlists/on_disk_inverted_lists.cc)
    list(APPEND POLARIS_HEADERS invlists/on_disk_inverted_lists.h)
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
