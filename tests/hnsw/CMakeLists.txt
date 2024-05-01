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
        NAME epsilon_search_test
        SOURCES
        epsilon_search_test.cc
        CXXOPTS ${POLARIS_CXX_OPTIONS} "-Wno-unused-variable"
        LINKS
        polaris::polaris
        ${POLARIS_DEPS_LINKS}
)
carbin_cc_binary(
        NAMESPACE polaris
        NAME multi_thread_load_test
        SOURCES
        multi_thread_load_test.cc
        CXXOPTS ${POLARIS_CXX_OPTIONS} "-Wno-unused-variable"
        LINKS
        polaris::polaris
        ${POLARIS_DEPS_LINKS}
)
carbin_cc_binary(
        NAMESPACE polaris
        NAME multi_thread_replace_test
        SOURCES
        multi_thread_replace_test.cc
        CXXOPTS ${POLARIS_CXX_OPTIONS} "-Wno-unused-variable"
        LINKS
        polaris::polaris
        ${POLARIS_DEPS_LINKS}
)

carbin_cc_binary(
        NAMESPACE polaris
        NAME multivector_search_test
        SOURCES
        multivector_search_test.cc
        CXXOPTS ${POLARIS_CXX_OPTIONS} "-Wno-unused-variable"
        LINKS
        polaris::polaris
        ${POLARIS_DEPS_LINKS}
)

carbin_cc_binary(
        NAMESPACE polaris
        NAME search_knn_closer_first_test
        SOURCES
        search_knn_closer_first_test.cc
        CXXOPTS ${POLARIS_CXX_OPTIONS} "-Wno-unused-variable"
        LINKS
        polaris::polaris
        ${POLARIS_DEPS_LINKS}
)

carbin_cc_binary(
        NAMESPACE polaris
        NAME search_knn_with_filter_test
        SOURCES
        search_knn_with_filter_test.cc
        CXXOPTS ${POLARIS_CXX_OPTIONS} "-Wno-unused-variable"
        LINKS
        polaris::polaris
        ${POLARIS_DEPS_LINKS}
)

carbin_cc_binary(
        NAMESPACE polaris
        NAME updates_test
        SOURCES
        updates_test.cc
        CXXOPTS ${POLARIS_CXX_OPTIONS} "-Wno-unused-variable"
        LINKS
        polaris::polaris
        ${POLARIS_DEPS_LINKS}
)