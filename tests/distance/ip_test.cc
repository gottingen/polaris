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

#include <polaris/distance/comparator.h>
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <collie/testing/doctest.h>
#include <polaris/core/common.h>
#include <turbo/random/random.h>
#include <limits>
#include <cstdint>
#include <iomanip>
#include <numeric>

//#define TEST_COMPARE_L1_TYPE int8_t, uint8_t, int16_t, uint16_t, int32_t, uint32_t, int64_t, uint64_t, float

#define TEST_COMPARE_L1_TYPE float, uint8_t, uint16_t, int16_t, uint32_t, int32_t, uint64_t, int64_t

namespace polaris {
    template<typename T>
    void compare_l2_test(uint32_t dim , uint32_t num = 1000) {
        aligned_vector<T> a(dim);
        aligned_vector<T> b(dim);
        auto d = dim * 2;
        T max;
        T min;
        if(std::is_floating_point<T>::value) {
            max = 100.0f;
            min = 0.0f;
        } else if(sizeof(T) == 4) {
            max = 100;
            min = 0;
        } else if(sizeof(T) == 2) {
            max = 64;
            min = 0;
        } else {
            max = 32;
            min = 0;
        }
        for (int k = 0; k < num; ++k) {
            for (uint32_t i = 0; i < dim; i++) {
                a[i] = turbo::uniform<T>(min, max);
                b[i] = turbo::uniform<T>(min, max);
            }
            float simple_result = primitive::compare_simple_dot_product(a.data(), b.data(), dim);
            float result = polaris::ComparatorInnerProduct<T>(dim)(ArrayView(a.data()), ArrayView(b.data()));
            if (simple_result != result) {
                //std::cout<<"a: "<<collie::format("{}", a)<<" b: "<<collie::format("{}", b)<<std::endl;
                std::cout <<std::fixed<<std::setprecision(6)<< "simple_result: " << simple_result << " result: " << result<< std::endl;
            }
            auto diff = std::abs(simple_result - result);
            // presicion is 0.05
            REQUIRE_LT(diff/simple_result, 0.00001f);
        }
    }

    TEST_CASE_TEMPLATE_DEFINE("distance inner product", T, l1_dis_test) {
        compare_l2_test<T>(128u);
    }

    TEST_CASE_TEMPLATE_INVOKE(l1_dis_test, TEST_COMPARE_L1_TYPE);

}  // namespace polaris


