# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

cmake_minimum_required(VERSION 3.17 FATAL_ERROR)

project(polaris_c_library LANGUAGES C CXX)

set(CMAKE_C_STANDARD 11)

set(POLARIS_C_SRC
  AutoTune_c.cc
  Clustering_c.cc
  IndexFlat_c.cc
  IndexIVFFlat_c.cc
  IndexIVF_c.cc
  IndexLSH_c.cc
  IndexPreTransform_c.cc
  VectorTransform_c.cc
  IndexShards_c.cc
  IndexReplicas_c.cc
  Index_c.cc
  IndexBinary_c.cc
  IndexScalarQuantizer_c.cc
  MetaIndexes_c.cc
  clone_index_c.cc
  error_impl.cc
  index_factory_c.cc
  index_io_c.cc
  impl/AuxIndexStructures_c.cc
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

if(POLARIS_ENABLE_GPU)
  add_subdirectory(gpu)
endif()
