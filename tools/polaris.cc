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
#include <collie/cli/cli.h>
#include <polaris/index_traits.h>
#include <collie/strings/format.h>
#include <iostream>

int main(int argc, char **argv) {
    collie::App app;
    std::vector<std::string> index_types = polaris::list_rt_indexes();
    std::cout << collie::format("Index types: {}\n", collie::join(index_types, "\n")) << std::endl;

    return 0;
}
