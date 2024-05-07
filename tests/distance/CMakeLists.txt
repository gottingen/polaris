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

carbin_cc_test(
        NAME type_trans_test
        MODULE distance
        SOURCES
        type_trans_test.cc
        CXXOPTS ${POLARIS_CXX_OPTIONS} "-Wno-unused-variable"
        LINKS
        polaris::polaris
        ${POLARIS_DEPS_LINKS}
)
carbin_cc_test(
        NAME l1_test
        MODULE distance
        SOURCES
        l1_test.cc
        CXXOPTS ${POLARIS_CXX_OPTIONS} "-Wno-unused-variable"
        LINKS
        polaris::polaris
        ${POLARIS_DEPS_LINKS}
)
carbin_cc_test(
        NAME l2_test
        MODULE distance
        SOURCES
        l2_test.cc
        CXXOPTS ${POLARIS_CXX_OPTIONS} "-Wno-unused-variable"
        LINKS
        polaris::polaris
        ${POLARIS_DEPS_LINKS}
)

