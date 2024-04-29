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

#include <polaris/tools/info.h>
#include <polaris/index_traits.h>
#include <collie/strings/format.h>

namespace polaris {

    static void rt_engine_list() {
        std::vector<std::string> index_types = polaris::list_rt_indexes();
        std::cout << collie::format("Index types: {}\n", collie::join(index_types, "\n")) << std::endl;
    }
    void setup_info_cli(collie::App *app) {
        auto rt_enigne = app->add_subcommand("rt_index", "Real-time index information");
        rt_enigne->callback(rt_engine_list);
        app->require_subcommand(1);
    }

}  // namespace polaris