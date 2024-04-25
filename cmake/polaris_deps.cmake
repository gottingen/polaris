#
# Copyright 2023 The Carbin Authors.
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
############################################################
# system pthread and rt, dl
############################################################
set(CARBIN_SYSTEM_DYLINK)
if (APPLE)
    find_library(CoreFoundation CoreFoundation)
    list(APPEND CARBIN_SYSTEM_DYLINK ${CoreFoundation} pthread)
elseif (${CMAKE_SYSTEM_NAME} STREQUAL "Linux")
    list(APPEND CARBIN_SYSTEM_DYLINK rt dl pthread)
endif ()

if (CARBIN_BUILD_TEST)
    enable_testing()
    #include(require_gtest)
    #include(require_gmock)
    #include(require_doctest)
endif (CARBIN_BUILD_TEST)

if (CARBIN_BUILD_BENCHMARK)
    #include(require_benchmark)
endif ()

find_package(Threads REQUIRED)

############################################################
#
# add you libs to the CARBIN_DEPS_LINK variable eg as turbo
# so you can and system pthread and rt, dl already add to
# CARBIN_SYSTEM_DYLINK, using it for fun.
##########################################################
set(CARBIN_DEPS_LINK
        #${TURBO_LIB}
        ${CARBIN_SYSTEM_DYLINK}
        )
list(REMOVE_DUPLICATES CARBIN_DEPS_LINK)
carbin_print_list_label("Denpendcies:" CARBIN_DEPS_LINK)

list(APPEND CMAKE_PREFIX_PATH "/opt/EA/inf")
find_package(collie REQUIRED)
find_package(turbo REQUIRED)
find_package(OpenMP REQUIRED)
find_package(MKL REQUIRED)

include_directories(${collie_INCLUDE_DIR})
include_directories(${turbo_INCLUDE_DIR})

set(POLARIS_DEPS_LINKS
        OpenMP::OpenMP_CXX
        ${MKL_LIBRARIES}
        ${CARBIN_DEPS_LINK}
        )





