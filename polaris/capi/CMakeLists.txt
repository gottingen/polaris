# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

cmake_minimum_required(VERSION 3.17 FATAL_ERROR)

project(polaris_c_library LANGUAGES C CXX)

set(CMAKE_C_STANDARD 11)

set(POLARIS_C_SRC
        auto_tune_c.cc
        clustering_c.cc
        index_flat_c.cc
        index_ivf_flat_c.cc
        index_ivf_c.cc
        index_lsh_c.cc
        index_pre_transform_c.cc
        vector_transform_c.cc
        index_shards_c.cc
        index_replicas_c.cc
        index_c.cc
        index_binary_c.cc
        index_scalar_quantizer_c.cc
        meta_indexes_c.cc
  clone_index_c.cc
  error_impl.cc
  index_factory_c.cc
  index_io_c.cc
        impl/auxIndex_structures_c.cc
  utils/distances_c.cc
)
add_library(polaris_c ${POLARIS_C_SRC})
target_link_libraries(polaris_c PRIVATE polaris)

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
     "*.h"
     "impl/*.h"
     "utils/*.h")

polaris_install_headers("${POLARIS_C_API_HEADERS}" c_api)

add_executable(example_c EXCLUDE_FROM_ALL example_c.c)
target_link_libraries(example_c PRIVATE polaris_c)

