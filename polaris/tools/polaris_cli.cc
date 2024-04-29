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
#include <polaris/tools/datasets/datasets.h>
#include <polaris/tools/info.h>
#include <polaris/tools/vamana/vamana.h>
#include <iostream>

int main(int argc, char **argv) {
    collie::App app;
    auto datasets_cmd = app.add_subcommand("datasets", "Dataset management commands");
    polaris::setup_datasets_cli(datasets_cmd);
    auto info_cmd = app.add_subcommand("info", "Index Information commands");
    polaris::setup_info_cli(info_cmd);
    auto vamana_cmd = app.add_subcommand("vamana", "Vamana commands");
    polaris::setup_vamana_cli(vamana_cmd);
    app.require_subcommand(1);
    COLLIE_CLI_PARSE(app, argc, argv);
    return 0;
}
