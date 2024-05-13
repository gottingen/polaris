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
#include <polaris/datasets/bin.h>
#include <polaris/graph/vamana/disk_utils.h>
#include <polaris/graph/vamana/math_utils.h>
#include <polaris/graph/vamana/index.h>
#include <polaris/graph/vamana/index_factory.h>
#include <polaris/graph/vamana/partition.h>


namespace polaris {

    struct BuildMemIndexContext {
        std::string data_type;
        std::string dist_fn;
        std::string index_path_prefix;
        std::string data_path;
        uint32_t num_threads;
        uint32_t R;
        uint32_t L;
        float alpha;
        uint32_t build_PQ_bytes;
        std::string codebook_prefix;
        bool use_opq;
        std::string universal_label;
    };

    BuildMemIndexContext ctx;

    static void run_build_memory_index_cli();

    void setup_build_memory_index_cli(collie::App *app) {

        app->add_option("--data_type", ctx.data_type, program_options_utils::DATA_TYPE_DESCRIPTION)->required(true);
        app->add_option("--dist_fn", ctx.dist_fn, program_options_utils::DISTANCE_FUNCTION_DESCRIPTION)->required(true);
        app->add_option("--index_path_prefix", ctx.index_path_prefix,
                        program_options_utils::INDEX_PATH_PREFIX_DESCRIPTION)->required(true);
        app->add_option("--data_path", ctx.data_path, program_options_utils::INPUT_DATA_PATH)->required(true);

        app->add_option("-T,--num_threads", ctx.num_threads,
                        program_options_utils::NUMBER_THREADS_DESCRIPTION)->default_val(omp_get_num_procs());
        app->add_option("-R,--max_degree", ctx.R, program_options_utils::MAX_BUILD_DEGREE)->default_val(64);
        app->add_option("-L,--Lbuild", ctx.L, program_options_utils::GRAPH_BUILD_COMPLEXITY)->default_val(100);
        app->add_option("--alpha", ctx.alpha, program_options_utils::GRAPH_BUILD_ALPHA)->default_val(1.2f);
        app->add_option("--build_PQ_bytes", ctx.build_PQ_bytes,
                        program_options_utils::BUIlD_GRAPH_PQ_BYTES)->default_val(0);
        app->add_flag("--use_opq", ctx.use_opq, program_options_utils::USE_OPQ);
        app->add_option("--universal_label", ctx.universal_label, program_options_utils::UNIVERSAL_LABEL)->default_val(
                "");
        app->callback(run_build_memory_index_cli);
    }

    void run_build_memory_index_cli() {
        polaris::MetricType metric;
        if (ctx.dist_fn == std::string("mips")) {
            metric = polaris::MetricType::METRIC_INNER_PRODUCT;
        } else if (ctx.dist_fn == std::string("l2")) {
            metric = polaris::MetricType::METRIC_L2;
        } else if (ctx.dist_fn == std::string("cosine")) {
            metric = polaris::MetricType::METRIC_COSINE;
        } else {
            std::cout << "Unsupported distance function. Currently only L2/ Inner "
                         "Product/Cosine are supported."
                      << std::endl;
            exit(-1);
        }
        auto use_pq_build = (ctx.build_PQ_bytes > 0);
        try {
            polaris::cout << "Starting index build with R: " << ctx.R << "  Lbuild: " << ctx.L << "  alpha: " << ctx.alpha
                          << "  #threads: " << ctx.num_threads << std::endl;

            size_t data_num, data_dim;
            polaris::get_bin_metadata(ctx.data_path, data_num, data_dim);

            auto index_build_params = polaris::IndexWriteParametersBuilder(ctx.L,ctx.R)
                    .with_alpha(ctx.alpha)
                    .with_saturate_graph(false)
                    .with_num_threads(ctx.num_threads)
                    .build();

            auto config = polaris::IndexConfigBuilder()
                    .with_metric(metric)
                    .with_dimension(data_dim)
                    .with_max_points(data_num)
                    .with_data_type(polaris::string_to_polaris_type(ctx.data_type))
                    .vamana_with_data_load_store_strategy(polaris::DataStoreStrategy::MEMORY)
                    .vamana_with_graph_load_store_strategy(polaris::GraphStoreStrategy::MEMORY)
                    .vamana_is_dynamic_index(false)
                    .vamana_with_index_write_params(index_build_params)
                    .vamana_is_use_opq(ctx.use_opq)
                    .vamana_is_pq_dist_build(use_pq_build)
                    .vamana_with_num_pq_chunks(ctx.build_PQ_bytes)
                    .build_vamana();

            auto index_factory = polaris::IndexFactory(config);
            auto index = index_factory.create_instance();
            index->build(ctx.data_path, data_num);
            index->save(ctx.index_path_prefix.c_str());
            index.reset();
        }
        catch (const std::exception &e) {
            std::cout << std::string(e.what()) << std::endl;
            polaris::cerr << "VamanaIndex build failed." << std::endl;
            exit(-1);
        }
    }
}  // namespace polaris
