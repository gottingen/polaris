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

#include <polaris/tools/vamana/vamana.h>
#include <polaris/tools/vamana/program_options_utils.h>
#include  <polaris/graph/utils.h>
#include  <polaris/graph/disk_utils.h>
#include  <polaris/graph/math_utils.h>
#include  <polaris/graph/index.h>
#include  <polaris/graph/partition.h>

namespace polaris {

    struct BuildDiskIndexContext {
        std::string data_type;
        std::string dist_fn;
        std::string index_path_prefix;
        std::string data_path;
        float B;
        float M;
        uint32_t num_threads;
        uint32_t R;
        uint32_t L;
        uint32_t QD;
        std::string codebook_prefix;
        uint32_t disk_PQ;
        bool append_reorder_data;
        uint32_t build_PQ;
        bool use_opq;
        std::string label_file;
        std::string universal_label;
        uint32_t Lf;
        uint32_t filter_threshold;
        std::string label_type;
    };
    BuildDiskIndexContext build_disk_index_context;

    static void run_build_disk_index();

    void setup_build_disk_index_cli(collie::App *app) {
        app->add_option("--data_type", build_disk_index_context.data_type,
                        program_options_utils::DATA_TYPE_DESCRIPTION)->required(
                true);
        app->add_option("--dist_fn", build_disk_index_context.dist_fn,
                        program_options_utils::DISTANCE_FUNCTION_DESCRIPTION)->required(
                true);
        app->add_option("--index_path_prefix", build_disk_index_context.index_path_prefix,
                        program_options_utils::INDEX_PATH_PREFIX_DESCRIPTION)->required(true);
        app->add_option("--data_path", build_disk_index_context.data_path,
                        program_options_utils::INPUT_DATA_PATH)->required(true);

        app->add_option("-B, --search_dram_budget", build_disk_index_context.B,
                        "DRAM budget in GB for searching the index to set the "
                        "compressed level for data while search happens")->required(true);

        app->add_option("-M, --build_dram_budget", build_disk_index_context.M,
                        "DRAM budget in GB for building the index")->required(true);

        // Optional parameters

        app->add_option("-T, --num_threads", build_disk_index_context.num_threads,
                        program_options_utils::NUMBER_THREADS_DESCRIPTION)->default_val(omp_get_num_procs());

        app->add_option("-R, --max_degree", build_disk_index_context.R,
                        program_options_utils::MAX_BUILD_DEGREE)->default_val(64);

        app->add_option("-L, --lbuid", build_disk_index_context.L,
                        program_options_utils::GRAPH_BUILD_COMPLEXITY)->default_val(100);

        app->add_option("-Q, --QD", build_disk_index_context.QD,
                        "Quantized Dimension for compression")->default_val(0);

        app->add_option("--codebook_prefix", build_disk_index_context.codebook_prefix,
                        "Path prefix for pre-trained codebook")->default_val("");

        app->add_option("--PQ_disk_bytes", build_disk_index_context.disk_PQ,
                        "Number of bytes to which vectors should be compressed "
                        "on SSD; 0 for no compression")->default_val(0);

        app->add_flag("--append_reorder_data", build_disk_index_context.append_reorder_data,
                      "Include full precision data in the index. Use only in "
                      "conjuction with compressed data on SSD.");

        app->add_option("--build_PQ_bytes", build_disk_index_context.build_PQ,
                        program_options_utils::BUIlD_GRAPH_PQ_BYTES)->default_val(0);

        app->add_flag("--use_opq", build_disk_index_context.use_opq,
                      program_options_utils::USE_OPQ);

        app->add_option("--label_file", build_disk_index_context.label_file,
                        program_options_utils::LABEL_FILE)->default_val("");

        app->add_option("--universal_label", build_disk_index_context.universal_label,
                        program_options_utils::UNIVERSAL_LABEL)->default_val("");

        app->add_option("--FilteredLbuild", build_disk_index_context.Lf,
                        program_options_utils::FILTERED_LBUILD)->default_val(0);
        app->add_option("--filter_threshold", build_disk_index_context.filter_threshold,
                        "Threshold to break up the existing nodes to generate new graph "
                        "internally where each node has a maximum F labels.")->default_val(0);

        app->add_option("--label_type", build_disk_index_context.label_type,
                        program_options_utils::LABEL_TYPE_DESCRIPTION)->default_val("uint");
        app->callback(run_build_disk_index);
    }

    void run_build_disk_index() {
        bool use_filters = (build_disk_index_context.label_file != "") ? true : false;
        polaris::MetricType metric;
        if (build_disk_index_context.dist_fn == std::string("l2"))
            metric = polaris::MetricType::METRIC_L2;
        else if (build_disk_index_context.dist_fn == std::string("mips"))
            metric = polaris::MetricType::METRIC_INNER_PRODUCT;
        else if (build_disk_index_context.dist_fn == std::string("cosine"))
            metric = polaris::MetricType::METRIC_COSINE;
        else {
            std::cout << "Error. Only l2 and mips distance functions are supported" << std::endl;
            exit(-1);
        }

        if (build_disk_index_context.append_reorder_data) {
            if (build_disk_index_context.disk_PQ == 0) {
                std::cout << "Error: It is not necessary to append data for reordering "
                             "when vectors are not compressed on disk."
                          << std::endl;
                exit(-1);
            }
            if (build_disk_index_context.data_type != std::string("float")) {
                std::cout << "Error: Appending data for reordering currently only "
                             "supported for float data type."
                          << std::endl;
                exit(-1);
            }
        }

        std::string params = std::string(std::to_string(build_disk_index_context.R)) + " " +
                             std::string(std::to_string(build_disk_index_context.L)) + " " +
                             std::string(std::to_string(build_disk_index_context.B)) + " " +
                             std::string(std::to_string(build_disk_index_context.M)) + " " +
                             std::string(std::to_string(build_disk_index_context.num_threads)) + " " +
                             std::string(std::to_string(build_disk_index_context.disk_PQ)) + " " +
                             std::string(std::to_string(build_disk_index_context.append_reorder_data)) + " " +
                             std::string(std::to_string(build_disk_index_context.build_PQ)) + " " +
                             std::string(std::to_string(build_disk_index_context.QD));

        try {
            if (build_disk_index_context.label_file != "" && build_disk_index_context.label_type == "ushort") {
                int  r;
                if (build_disk_index_context.data_type == std::string("int8"))
                    r = polaris::build_disk_index<int8_t>(build_disk_index_context.data_path.c_str(),
                                                          build_disk_index_context.index_path_prefix.c_str(),
                                                          params.c_str(),
                                                          metric, build_disk_index_context.use_opq,
                                                          build_disk_index_context.codebook_prefix, use_filters,
                                                          build_disk_index_context.label_file,
                                                          build_disk_index_context.universal_label,
                                                          build_disk_index_context.filter_threshold,
                                                          build_disk_index_context.Lf);
                else if (build_disk_index_context.data_type == std::string("uint8"))
                    r = polaris::build_disk_index<uint8_t, uint16_t>(
                            build_disk_index_context.data_path.c_str(),
                            build_disk_index_context.index_path_prefix.c_str(), params.c_str(), metric,
                            build_disk_index_context.use_opq, build_disk_index_context.codebook_prefix,
                            use_filters, build_disk_index_context.label_file, build_disk_index_context.universal_label,
                            build_disk_index_context.filter_threshold, build_disk_index_context.Lf);
                else if (build_disk_index_context.data_type == std::string("float"))
                    r = polaris::build_disk_index<float, uint16_t>(
                            build_disk_index_context.data_path.c_str(),
                            build_disk_index_context.index_path_prefix.c_str(), params.c_str(), metric,
                            build_disk_index_context.use_opq, build_disk_index_context.codebook_prefix,
                            use_filters, build_disk_index_context.label_file, build_disk_index_context.universal_label,
                            build_disk_index_context.filter_threshold, build_disk_index_context.Lf);
                else {
                    polaris::cerr << "Error. Unsupported data type" << std::endl;
                    exit(-1);
                }
                if (r != 0) {
                    polaris::cerr << "Index build failed." << std::endl;
                    exit(-1);
                }
            } else {
                int  r;
                if (build_disk_index_context.data_type == std::string("int8"))
                    r= polaris::build_disk_index<int8_t>(build_disk_index_context.data_path.c_str(),
                                                             build_disk_index_context.index_path_prefix.c_str(),
                                                             params.c_str(),
                                                             metric, build_disk_index_context.use_opq,
                                                             build_disk_index_context.codebook_prefix, use_filters,
                                                             build_disk_index_context.label_file,
                                                             build_disk_index_context.universal_label,
                                                             build_disk_index_context.filter_threshold,
                                                             build_disk_index_context.Lf);
                else if (build_disk_index_context.data_type == std::string("uint8"))
                    r= polaris::build_disk_index<uint8_t>(build_disk_index_context.data_path.c_str(),
                                                              build_disk_index_context.index_path_prefix.c_str(),
                                                              params.c_str(),
                                                              metric, build_disk_index_context.use_opq,
                                                              build_disk_index_context.codebook_prefix, use_filters,
                                                              build_disk_index_context.label_file,
                                                              build_disk_index_context.universal_label,
                                                              build_disk_index_context.filter_threshold,
                                                              build_disk_index_context.Lf);
                else if (build_disk_index_context.data_type == std::string("float"))
                    r= polaris::build_disk_index<float>(build_disk_index_context.data_path.c_str(),
                                                            build_disk_index_context.index_path_prefix.c_str(),
                                                            params.c_str(),
                                                            metric, build_disk_index_context.use_opq,
                                                            build_disk_index_context.codebook_prefix, use_filters,
                                                            build_disk_index_context.label_file,
                                                            build_disk_index_context.universal_label,
                                                            build_disk_index_context.filter_threshold,
                                                            build_disk_index_context.Lf);
                else {
                    polaris::cerr << "Error. Unsupported data type" << std::endl;
                    exit(-1);
                }
                if (r != 0) {
                    polaris::cerr << "Index build failed." << std::endl;
                    exit(-1);
                }
            }
        }
        catch (const std::exception &e) {
            std::cout << std::string(e.what()) << std::endl;
            polaris::cerr << "Index build failed." << std::endl;
            exit(-1);
        }
    }
}  // namespace polaris
