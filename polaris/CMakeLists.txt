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
        core/log.cc
        core/executor.cc
        core/random.cc
        core/search_condition.cc
        distance/distance.cc
        distance/distance_impl.cc
        datasets/fvecs.cc
        datasets/ivecs.cc
        datasets/tsv.cc
        datasets/to_float.cc
        datasets/random.cc
        datasets/recall.cc
        faiss/index.cc
        faiss/id_selector.cc
        faiss/utils.cc
        faiss/matrix_stats.cc
        faiss/sorting.cc
        faiss/heap.cc
        faiss/aux_index_structures.cc
        faiss/index_flat.cc
        faiss/index_ivf.cc
        faiss/index_ivf_pq.cc
        faiss/index_ivf_flat.cc
        faiss/auto_tune.cc
        faiss/index_ivf_pqr.cc
        faiss/index_hnsw.cc
        faiss/index_pq.cc
        faiss/index_lsh.cc
        faiss/index_nsg.cc
        faiss/index_binary.cc
        faiss/index_binary_flat.cc
        faiss/index_binary_hnsw.cc
        faiss/index_binary_ivf.cc
        faiss/index_factory.cc
        faiss/index_binary_hash.cc
        faiss/index_flat_codes.cc
        faiss/index_id_map.cc
        faiss/index_2layer.cc
        faiss/index_lattice.cc
        faiss/index_refine.cc
        faiss/index_replicas.cc
        faiss/index_pq_fast_scan.cc
        faiss/index_ivf_fast_scan.cc
        faiss/index_nn_descent.cc
        faiss/index_shards.cc
        faiss/index_fast_scan.cc
        faiss/index_ivf_pq_fast_scan.cc
        faiss/index_ivf_spectral_hash.cc
        faiss/index_shards_ivf.cc
        faiss/index_additive_quantizer.cc
        faiss/index_binary_from_float.cc
        faiss/index_additive_quantizer_fast_scan.cc
        faiss/index_ivf_additive_quantizer.cc
        faiss/index_ivf_additive_quantizer_fast_scan.cc
        faiss/index_ivf_independent_quantizer.cc
        faiss/index_scalar_quantizer.cc
        faiss/index_rowwise_minmax.cc
        faiss/clone_index.cc
        faiss/index_pre_transform.cc
        faiss/meta_indexes.cc
        faiss/ivf_lib.cc
        faiss/vector_transform.cc
        faiss/impl/hnsw.cc
        faiss/impl/nsg.cc
        faiss/impl/polysemous_training.cc
        faiss/impl/product_additive_quantizer.cc
        faiss/impl/index_read.cc
        faiss/impl/index_write.cc
        faiss/impl/pq4_fast_scan.cc
        faiss/impl/pq4_fast_scan_search_1.cc
        faiss/impl/pq4_fast_scan_search_qbs.cc
        faiss/impl/residual_quantizer_encode_steps.cc
        faiss/impl/nn_descent.cc
        faiss/quantization/product_quantizer.cc
        faiss/quantization/additive_quantizer.cc
        faiss/quantization/local_search_quantizer.cc
        faiss/quantization/residual_quantizer.cc
        faiss/quantization/scalar_quantizer.cc
        faiss/quantization/clustering.cc
        faiss/utils/distances.cc
        faiss/utils/distances_simd.cc
        faiss/utils/extra_distances.cc
        faiss/utils/hamming.cc
        faiss/utils/distances_fused/avx512.cc
        faiss/utils/distances_fused/distances_fused.cc
        faiss/utils/distances_fused/simdlib_based.cc
        faiss/invlists/block_inverted_lists.cc
        faiss/invlists/direct_map.cc
        faiss/invlists/inverted_lists.cc
        faiss/invlists/inverted_lists_io_hook.cc
        faiss/internal/kmeans1d.cc
        faiss/internal/io.cc
        faiss/internal/lattice_zn.cc
        faiss/internal/code_packer.cc
        faiss/internal/worker_thread.cc
        faiss/internal/partitioning.cc
        faiss/internal/quantize_lut.cc
        io/linux_aligned_file_reader.cc
        io/array_file.cc
        storage/abstract_data_store.cc
        storage/in_mem_data_store.cc
        storage/in_mem_graph_store.cc
        graph/ngt/command.cc
        graph/ngt/graph.cc
        graph/ngt/index.cc
        graph/ngt/node.cc
        graph/ngt/object_space.cc
        graph/ngt/thread.cc
        graph/ngt/tree.cc
        graph/ngt/version.cc
        graph/ngt/ngtq/hierarchical_kmeans.cc
        graph/ngt/ngtq/optimizer.cc
        graph/ngt/ngtq/qbg_cli.cc
        graph/ngt/ngtq/quantized_graph.cc
        graph/vamana/abstract_index.cc
        graph/vamana/disk_utils.cc
        #graph/vamana/filter_utils.cc
        graph/vamana/index.cc
        graph/vamana/index_factory.cc
        graph/vamana/logger.cc
        graph/vamana/math_utils.cc
        graph/vamana/partition.cc
        graph/vamana/pq.cc
        graph/vamana/pq_data_store.cc
        graph/vamana/pq_flash_index.cc
        graph/vamana/pq_l2_distance.cc
        graph/vamana/scratch.cc
        graph/vamana/utils.cc
        utility/memory_mapper.cc
        utility/mmap_manager.cc
        utility/natural_number_map.cc
        utility/natural_number_set.cc
        utility/polaris_exception.cc
        utility/recall.cc
        utility/shared_memory_allocator.cc
        unified_index.cc
)


set(POLARIS_HEADERS
        core/common.h
        core/defines.h
        core/executor.h
        core/index_build_params.h
        core/index_config.h
        core/lock.h
        core/log.h
        core/memory.h
        core/metric_type.h
        core/search_condition.h
        core/vamana_parameters.h
        core/random.h
        distance/cosine_similarity.h
        distance/distance_impl.h
        distance/primitive.h
        distance/simd_utils.h
        distance/primitive_comparator.h
        datasets/fvecs.h
        datasets/ivecs.h
        datasets/tsv.h
        datasets/to_float.h
        datasets/random.h
        datasets/recall.h
        graph/ngt/command.h
        graph/ngt/clustering.h
        graph/ngt/graph_optimizer.h
        graph/ngt/graph_reconstructor.h
        graph/ngt/index.h
        graph/ngt/node.h
        graph/ngt/object_repository.h
        graph/ngt/object_space.h
        graph/ngt/object_space_repository.h
        graph/ngt/optimizer.h
        graph/ngt/thread.h
        graph/ngt/tree.h
        graph/ngt/version.h
        graph/ngt/ngtq/hierarchical_kmeans.h
        graph/ngt/ngtq/matrix.h
        graph/ngt/ngtq/optimizer.h
        graph/ngt/ngtq/qbg_cli.h
        graph/ngt/ngtq/quantized_blob_graph.h
        graph/ngt/ngtq/quantized_graph.h
        graph/ngt/ngtq/quantizer.h
        graph/vamana/abstract_index.h
        graph/vamana/abstract_scratch.h
        graph/vamana/disk_utils.h
        #graph/vamana/filter_utils.h
        graph/vamana/index.h
        graph/vamana/index_factory.h
        graph/vamana/logger.h
        graph/vamana/logger_impl.h
        graph/vamana/math_utils.h
        graph/vamana/neighbor.h
        graph/vamana/partition.h
        graph/vamana/percentile_stats.h
        graph/vamana/pq.h
        graph/vamana/pq_common.h
        graph/vamana/pq_data_store.h
        graph/vamana/pq_flash_index.h
        graph/vamana/pq_l2_distance.h
        graph/vamana/pq_scratch.h
        graph/vamana/quantized_distance.h
        graph/vamana/scratch.h
        graph/vamana/timer.h
        graph/vamana/utils.h
        faiss/aux_index_structures.h
        faiss/index_io.h
        faiss/id_selector.h
        faiss/matrix_stats.h
        faiss/ordered_key_value.h
        faiss/sorting.h
        faiss/heap.h
        faiss/utils.h
        faiss/result_handler.h
        faiss/index.h
        faiss/index_flat.h
        faiss/index_ivf.h
        faiss/index_ivf_pq.h
        faiss/index_ivf_flat.h
        faiss/auto_tune.h
        faiss/index_ivf_pqr.h
        faiss/index_io.h
        faiss/index_hnsw.h
        faiss/index_pq.h
        faiss/index_lsh.h
        faiss/index_nsg.h
        faiss/index_binary.h
        faiss/index_binary_flat.h
        faiss/index_binary_hnsw.h
        faiss/index_binary_ivf.h
        faiss/index_factory.h
        faiss/index_binary_hash.h
        faiss/index_flat_codes.h
        faiss/index_id_map.h
        faiss/index_2layer.h
        faiss/index_lattice.h
        faiss/index_refine.h
        faiss/index_replicas.h
        faiss/index_pq_fast_scan.h
        faiss/index_ivf_fast_scan.h
        faiss/index_nn_descent.h
        faiss/index_shards.h
        faiss/index_fast_scan.h
        faiss/index_ivf_pq_fast_scan.h
        faiss/index_ivf_spectral_hash.h
        faiss/index_shards_ivf.h
        faiss/index_additive_quantizer.h
        faiss/index_binary_from_float.h
        faiss/index_additive_quantizer_fast_scan.h
        faiss/index_ivf_additive_quantizer.h
        faiss/index_ivf_additive_quantizer_fast_scan.h
        faiss/index_ivf_independent_quantizer.h
        faiss/index_scalar_quantizer.h
        faiss/index_rowwise_minmax.h
        faiss/clone_index.h
        faiss/index_pre_transform.h
        faiss/meta_indexes.h
        faiss/ivf_lib.h
        faiss/vector_transform.h
        faiss/impl/distance_computer.h
        faiss/impl/hnsw.h
        faiss/impl/product_additive_quantizer.h
        faiss/impl/lookup_table_scaler.h
        faiss/impl/nn_descent.h
        faiss/impl/nsg.h
        faiss/impl/polysemous_training.h
        faiss/impl/threaded_index-inl.h
        faiss/impl/threaded_index.h
        faiss/impl/pq4_fast_scan.h
        faiss/impl/residual_quantizer_encode_steps.h
        faiss/impl/simd_result_handlers.h
        faiss/impl/code_distance/code_distance.h
        faiss/impl/code_distance/code_distance-generic.h
        faiss/impl/code_distance/code_distance-avx2.h
        faiss/quantization/quantizer.h
        faiss/quantization/product_quantizer-inl.h
        faiss/quantization/additive_quantizer.h
        faiss/quantization/product_quantizer.h
        faiss/quantization/local_search_quantizer.h
        faiss/quantization/residual_quantizer.h
        faiss/quantization/scalar_quantizer.h
        faiss/quantization/clustering.h
        faiss/utils/distances.h
        faiss/utils/extra_distances-inl.h
        faiss/utils/extra_distances.h
        faiss/utils/hamming-inl.h
        faiss/utils/hamming.h
        faiss/utils/prefetch.h
        faiss/utils/simdlib.h
        faiss/utils/simdlib_avx2.h
        faiss/utils/simdlib_emulated.h
        faiss/utils/simdlib_neon.h
        faiss/utils/distances_fused/avx512.h
        faiss/utils/distances_fused/distances_fused.h
        faiss/utils/distances_fused/simdlib_based.h
        faiss/utils/approx_topk/approx_topk.h
        faiss/utils/approx_topk/avx2-inl.h
        faiss/utils/approx_topk/generic.h
        faiss/utils/approx_topk/mode.h
        faiss/utils/approx_topk_hamming/approx_topk_hamming.h
        faiss/utils/transpose/transpose-avx2-inl.h
        faiss/utils/hamming_distance/common.h
        faiss/utils/hamming_distance/generic-inl.h
        faiss/utils/hamming_distance/hamdis-inl.h
        faiss/utils/hamming_distance/neon-inl.h
        faiss/utils/hamming_distance/avx2-inl.h
        faiss/invlists/block_inverted_lists.h
        faiss/invlists/direct_map.h`
        faiss/invlists/inverted_lists.h
        faiss/invlists/inverted_lists_io_hook.h
        faiss/internal/code_packer.h
        faiss/internal/io_macros.h
        faiss/internal/io.h
        faiss/internal/lattice_zn.h
        faiss/internal/kmeans1d.h
        faiss/internal/fp16-fp16c.h
        faiss/internal/fp16-inl.h
        faiss/internal/fp16-arm.h
        faiss/internal/fp16.h
        faiss/internal/worker_thread.h
        faiss/internal/aligned_table.h
        faiss/internal/partitioning.h
        faiss/internal/quantize_lut.h
        io/aligned_file_reader.h
        io/array_file.h
        io/cached_io.h
        io/linux_aligned_file_reader.h
        io/utils.h
        storage/abstract_data_store.h
        storage/abstract_graph_store.h
        storage/dynamic_length_vector.h
        storage/in_mem_graph_store.h
        storage/in_mem_data_store.h
        storage/repository.h
        utility/any_wrappers.h
        utility/args.h
        utility/bfloat.h
        utility/boolean_set.h
        utility/common_includes.h
        utility/compact.h
        utility/concurrent_queue.h
        utility/doubly_buffered_data.h
        utility/half.hpp
        utility/hash_based_boolean_set.h
        utility/memory_mapper.h
        utility/mmap_manager.h
        utility/mmap_manager_defs.h
        utility/mmap_manager_exception.h
        utility/mmap_manager_impl.h
        utility/natural_number_map.h
        utility/natural_number_set.h
        utility/platform_macros.h
        utility/polaris_assert.h
        utility/polaris_exception.h
        utility/prefetch.h
        utility/property_set.h
        utility/recall.h
        utility/serialize.h
        utility/shared_memory_allocator.h
        utility/tag_uint128.h
        utility/timer.h
        utility/types.h
        utility/utils.h
        unified_index.h
)

if (NOT WIN32)
    list(APPEND POLARIS_SRC faiss/invlists/on_disk_inverted_lists.cc)
    list(APPEND POLARIS_HEADERS faiss/invlists/on_disk_inverted_lists.h)
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
        capi/ngtq_api.cc
        capi/ngt_api.cc
)

carbin_cc_library(
        NAMESPACE polaris
        NAME polaris_c
        SOURCES
        ${POLARIS_C_SRC}
        DEFINES ${POLARIS_CXX_DEFINES}
        CXXOPTS
        ${POLARIS_CXX_OPTIONS}
        PLINKS
        ${POLARIS_DEPS_LINKS} polaris::polaris
        PUBLIC
)
carbin_cc_binary(
        NAMESPACE polaris
        NAME pcli
        SOURCES
        tools/polaris_cli.cc
        tools/datasets/datasets.cc
        tools/datasets/compute_ground_truth.cc
        tools/datasets/generate_synthetic_labels.cc
        tools/info.cc
        tools/vamana/vamana.cc
        tools/vamana/build_disk_index.cc
        tools/vamana/build_memory_index.cc
        tools/vamana/search_disk_index.cc
        tools/vamana/search_memory_index.cc
        tools/vamana/merge_shards.cc
        DEFINES ${POLARIS_CXX_DEFINES}
        CXXOPTS ${POLARIS_CXX_OPTIONS} "-Wno-unused-variable"
        LINKS
        polaris::polaris
        ${POLARIS_DEPS_LINKS}
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
        "capi/faiss/impl/*.h"
        "capi/faiss/utils/*.h")

polaris_install_headers("${POLARIS_C_API_HEADERS}" c_api)
