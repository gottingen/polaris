// Copyright 2023 The Elastic-AI Authors.
// part of Elastic AI Search
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <collie/testing/doctest.h>
#include <polaris/distance/norm.h>
#include <polaris/core/defines.h>

namespace polaris {
    class TypeTransTest {
    public:
        TypeTransTest() {
            int8_data = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
            float_data = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0};
        }
        ~TypeTransTest() = default;
        aligned_vector<int8_t> int8_data;
        aligned_vector<float> float_data;
    };

    TEST_CASE_FIXTURE(TypeTransTest, "simd_type_promotion") {
        REQUIRE(std::is_same<simd_type_promotion_t<int8_t>, int32_t>::value);
        REQUIRE(std::is_same<simd_type_promotion_t<uint8_t>, uint32_t>::value);
        REQUIRE(std::is_same<simd_type_promotion_t<int16_t>, int32_t>::value);
        REQUIRE(std::is_same<simd_type_promotion_t<uint16_t>, uint32_t>::value);
    }

    TEST_CASE_FIXTURE(TypeTransTest, "NormL2") {
        float result_float = NormL2<float, collie::simd::best_arch, collie::simd::aligned_mode>::norm_l2_sqr(float_data.data(), float_data.size());
        float result_int8 = NormL2<int8_t, collie::simd::best_arch, collie::simd::aligned_mode>::norm_l2_sqr(int8_data.data(), int8_data.size());
        REQUIRE(result_float == result_int8);
    }
}  // namespace polaris