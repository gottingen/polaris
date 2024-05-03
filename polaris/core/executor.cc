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
//

#include <polaris/core/executor.h>
#include <polaris/utility/polaris_assert.h>
#include <thread>

namespace polaris {

    collie::tf::Executor* g_executor = nullptr;
    static size_t g_thread_num = 4;

    void set_global_executor_thread_num(int num) {
        POLARIS_ASSERT_MSG(g_executor == nullptr, "Executor has been initialized");
        POLARIS_ASSERT_FMT(num >= 4, "%d is not a valid thread number, default is 4, you must set it greater equal then 4", num);
        g_thread_num = num;
    }

    std::once_flag g_executor_flag;


    collie::tf::Executor *get_global_executor() {
        std::call_once(g_executor_flag, []() {
            g_executor = new collie::tf::Executor(g_thread_num);
        });
        return g_executor;
    }

} // namespace polaris