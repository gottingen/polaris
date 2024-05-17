#
# Copyright 2024 The EA Authors.
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

carbin_cc_binary(
        NAMESPACE polaris
        NAME cosine_float
        SOURCES
        cosine_float.cc
        CXXOPTS ${POLARIS_CXX_OPTIONS} "-Wno-unused-variable"
        LINKS
        polaris::polaris
        ${POLARIS_DEPS_LINKS}
)

carbin_cc_binary(
        NAMESPACE polaris
        NAME hamming_uint8
        SOURCES
        hamming_uint8.cc
        CXXOPTS ${POLARIS_CXX_OPTIONS} "-Wno-unused-variable"
        LINKS
        polaris::polaris
        ${POLARIS_DEPS_LINKS}
)

carbin_cc_binary(
        NAMESPACE polaris
        NAME jaccard_sparse
        SOURCES
        jaccard_sparse.cc
        CXXOPTS ${POLARIS_CXX_OPTIONS} "-Wno-unused-variable"
        LINKS
        polaris::polaris
        ${POLARIS_DEPS_LINKS}
)

carbin_cc_binary(
        NAMESPACE polaris
        NAME l2_uint8
        SOURCES
        l2_uint8.cc
        CXXOPTS ${POLARIS_CXX_OPTIONS} "-Wno-unused-variable"
        LINKS
        polaris::polaris
        ${POLARIS_DEPS_LINKS}
)

carbin_cc_binary(
        NAMESPACE polaris
        NAME l2_uint8_range_search
        SOURCES
        l2_uint8_range_search.cc
        CXXOPTS ${POLARIS_CXX_OPTIONS} "-Wno-unused-variable"
        LINKS
        polaris::polaris
        ${POLARIS_DEPS_LINKS}
)

carbin_cc_binary(
        NAMESPACE polaris
        NAME qbg_capi
        SOURCES
        qbg_capi.cc
        CXXOPTS ${POLARIS_CXX_OPTIONS} "-Wno-unused-variable"
        LINKS
        polaris::polaris polaris::polaris_c
        ${POLARIS_DEPS_LINKS}
)

carbin_cc_binary(
        NAMESPACE polaris
        NAME qg_capi
        SOURCES
        qg_capi.cc
        CXXOPTS ${POLARIS_CXX_OPTIONS} "-Wno-unused-variable"
        LINKS
        polaris::polaris_c polaris::polaris
        ${POLARIS_DEPS_LINKS}
)

carbin_cc_binary(
        NAMESPACE polaris
        NAME qg_l2_float
        SOURCES
        qg_l2_float.cc
        CXXOPTS ${POLARIS_CXX_OPTIONS} "-Wno-unused-variable"
        LINKS
        polaris::polaris #polaris::polaris_c
        ${POLARIS_DEPS_LINKS}
)
