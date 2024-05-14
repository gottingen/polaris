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
#include <polaris/utility/common_includes.h>
#include <polaris/utility/recall.h>
#include <polaris/utility/timer.h>
#include <polaris/core/percentile_stats.h>
#include <polaris/unified_index.h>
#include <polaris/datasets/bin.h>
#include <sys/mman.h>


template<typename T>
int search_memory_index(polaris::MetricType &metric, const std::string &index_path, const std::string &result_path_prefix,
                     const std::string &query_file, const std::string &truthset_file, const uint32_t num_threads,
                     const uint32_t recall_at, const bool print_all_recalls, const std::vector<uint32_t> &Lvec,
                     const bool dynamic, const bool show_qps_per_thread, const float fail_if_recall_below) {
    // Load the query file
    T *query = nullptr;
    uint32_t *gt_ids = nullptr;
    float *gt_dists = nullptr;
    size_t query_num, query_dim, query_aligned_dim, gt_num, gt_dim;
    polaris::load_aligned_bin<T>(query_file, query, query_num, query_dim, query_aligned_dim);

    bool calc_recall_flag = false;
    if (truthset_file != std::string("null") && collie::filesystem::exists(truthset_file)) {
        polaris::load_truthset(truthset_file, gt_ids, gt_dists, gt_num, gt_dim);
        if (gt_num != query_num) {
            std::cout << "Error. Mismatch in number of queries and ground truth data" << std::endl;
        }
        calc_recall_flag = true;
    } else {
        polaris::cout << " Truthset file " << truthset_file << " not found. Not computing recall." << std::endl;
    }
    auto num_frozen_pts_rs = polaris::UnifiedIndex::get_frozen_points(polaris::IndexType::IT_VAMANA, index_path);
    if(!num_frozen_pts_rs.ok()) {
        std::cerr << "Failed to get number of frozen points from index" << std::endl;
        exit(-1);
    }
    const size_t num_frozen_pts = num_frozen_pts_rs.value();

    auto config = polaris::IndexConfigBuilder()
            .with_metric(metric)
            .with_dimension(query_dim)
            .with_max_points(*(std::max_element(Lvec.begin(), Lvec.end())))
            .with_load_threads(num_threads)
            .vamana_with_data_load_store_strategy(polaris::DataStoreStrategy::MEMORY)
            .vamana_with_graph_load_store_strategy(polaris::GraphStoreStrategy::MEMORY)
            .with_data_type(polaris::polaris_type_to_name<T>())
            .vamana_is_dynamic_index(dynamic)
            .vamana_is_concurrent_consolidate(false)
            .vamana_is_pq_dist_build(false)
            .vamana_is_use_opq(false)
            .vamana_with_num_pq_chunks(0)
            .vamana_with_num_frozen_pts(num_frozen_pts)
            .build_vamana();

    std::unique_ptr<polaris::UnifiedIndex> unified_index(polaris::UnifiedIndex::create_index(polaris::IndexType::IT_VAMANA));
    unified_index->initialize(config);
    auto lrs = unified_index->load(index_path);
    if(!lrs.ok()) {
        std::cerr << "Failed to load index from " << index_path << " error: " << lrs.message() << std::endl;
        exit(-1);
    }
    std::cout << "VamanaIndex loaded" << std::endl;


    std::cout << "Using " << num_threads << " threads to search" << std::endl;
    std::cout.setf(std::ios_base::fixed, std::ios_base::floatfield);
    std::cout.precision(2);
    const std::string qps_title = show_qps_per_thread ? "QPS/thread" : "QPS";
    uint32_t table_width = 0;
        std::cout << std::setw(4) << "Ls" << std::setw(12) << qps_title << std::setw(18) << "Avg dist cmps"
                  << std::setw(20) << "Mean Latency (mus)" << std::setw(15) << "99.9 Latency";
        table_width += 4 + 12 + 18 + 20 + 15;
    uint32_t recalls_to_print = 0;
    const uint32_t first_recall = print_all_recalls ? 1 : recall_at;
    if (calc_recall_flag) {
        for (uint32_t curr_recall = first_recall; curr_recall <= recall_at; curr_recall++) {
            std::cout << std::setw(12) << ("Recall@" + std::to_string(curr_recall));
        }
        recalls_to_print = recall_at + 1 - first_recall;
        table_width += recalls_to_print * 12;
    }
    std::cout << std::endl;
    std::cout << std::string(table_width, '=') << std::endl;

    std::vector<std::vector<std::unique_ptr<polaris::SearchContext>>> search_contexts(Lvec.size());
    std::vector<float> latency_stats(query_num, 0);
    std::vector<uint32_t> cmp_stats;
    cmp_stats = std::vector<uint32_t>(query_num, 0);
    double best_recall = 0.0;

    for (uint32_t test_id = 0; test_id < Lvec.size(); test_id++) {
        uint32_t L = Lvec[test_id];
        if (L < recall_at) {
            polaris::cout << "Ignoring search with L:" << L << " since it's smaller than K:" << recall_at << std::endl;
            continue;
        }
        search_contexts[test_id].resize(query_num);
        for(int i = 0; i < query_num; i++) {
            search_contexts[test_id][i] = std::make_unique<polaris::SearchContext>();
        }
        std::vector<void *> res = std::vector<void *>();

        auto s = std::chrono::high_resolution_clock::now();
        omp_set_num_threads(num_threads);
#pragma omp parallel for schedule(dynamic, 1)
        for (int64_t i = 0; i < (int64_t) query_num; i++) {
            auto qs = std::chrono::high_resolution_clock::now();
            auto &ctx = *search_contexts[test_id][i];
            ctx.set_meta(polaris::polaris_type_to_name<T>(), query_aligned_dim)
            .set_query(query + i * query_aligned_dim)
                    .set_top_k(recall_at)
                    .set_search_list(L)
                    .set_with_local_ids(true);
            auto rs = unified_index->search(ctx);
            if(!rs.ok()) {
                std::cerr << "Search failed for query " << i <<" error: "<<rs.message()<< std::endl;
                exit(-1);
            }
            cmp_stats[i] = ctx.cmps;
            auto qe = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> diff = qe - qs;
            latency_stats[i] = (float) (diff.count() * 1000000);
        }
        std::chrono::duration<double> diff = std::chrono::high_resolution_clock::now() - s;

        double displayed_qps = query_num / diff.count();

        if (show_qps_per_thread)
            displayed_qps /= num_threads;

        std::vector<double> recalls;
        if (calc_recall_flag) {
            recalls.reserve(recalls_to_print);
            for (uint32_t curr_recall = first_recall; curr_recall <= recall_at; curr_recall++) {
                recalls.push_back(polaris::calculate_recall((uint32_t) query_num, gt_ids, gt_dists, (uint32_t) gt_dim,
                                                            search_contexts[test_id], recall_at, curr_recall));
            }
        }

        std::sort(latency_stats.begin(), latency_stats.end());
        double mean_latency =
                std::accumulate(latency_stats.begin(), latency_stats.end(), 0.0) / static_cast<float>(query_num);

        float avg_cmps = (float) std::accumulate(cmp_stats.begin(), cmp_stats.end(), 0) / (float) query_num;

        std::cout << std::setw(4) << L << std::setw(12) << displayed_qps << std::setw(18) << avg_cmps
                  << std::setw(20) << (float) mean_latency << std::setw(15)
                  << (float) latency_stats[(uint64_t) (0.999 * query_num)];
        for (double recall: recalls) {
            std::cout << std::setw(12) << recall;
            best_recall = std::max(recall, best_recall);
        }
        std::cout << std::endl;
    }

    std::cout << "Done searching. Now saving results " << std::endl;
    uint64_t test_id = 0;
    /*
    for (auto L: Lvec) {
        if (L < recall_at) {
            polaris::cout << "Ignoring search with L:" << L << " since it's smaller than K:" << recall_at << std::endl;
            continue;
        }
        std::string cur_result_path_prefix = result_path_prefix + "_" + std::to_string(L);

        std::string cur_result_path = cur_result_path_prefix + "_idx_uint32.bin";
        polaris::save_bin<uint32_t>(cur_result_path, query_result_ids[test_id].data(), query_num, recall_at);

        cur_result_path = cur_result_path_prefix + "_dists_float.bin";
        polaris::save_bin<float>(cur_result_path, query_result_dists[test_id].data(), query_num, recall_at);

        test_id++;
    }*/

    polaris::aligned_free(query);
    return best_recall >= fail_if_recall_below ? 0 : -1;
}

namespace polaris {
    struct SearchMemoryIndexContext {
        std::string data_type;
        std::string dist_fn;
        std::string index_path_prefix;
        std::string result_path;
        std::string query_file;
        std::string gt_file;
        uint32_t num_threads{0};
        uint32_t K{0};
        std::vector<uint32_t> Lvec;
        bool print_all_recalls{true};
        bool dynamic{false};
        bool show_qps_per_thread{false};
        float fail_if_recall_below{false};
    };
    static SearchMemoryIndexContext ctx;

    void run_search_memory_index_cli();

    void setup_search_memory_index_cli(collie::App *app) {

        app->add_option("--data_type", ctx.data_type, program_options_utils::DATA_TYPE_DESCRIPTION)->required();
        app->add_option("--dist_fn", ctx.dist_fn, program_options_utils::DISTANCE_FUNCTION_DESCRIPTION)->required();
        app->add_option("--index_path_prefix", ctx.index_path_prefix,
                        program_options_utils::INDEX_PATH_PREFIX_DESCRIPTION)->required();
        app->add_option("--result_path", ctx.result_path, program_options_utils::RESULT_PATH_DESCRIPTION)->required();
        app->add_option("--query_file", ctx.query_file, program_options_utils::QUERY_FILE_DESCRIPTION)->required();
        app->add_option("-K,--recall_at", ctx.K, program_options_utils::NUMBER_OF_RESULTS_DESCRIPTION)->required();
        app->add_option("-L, --search_list", ctx.Lvec, program_options_utils::SEARCH_LIST_DESCRIPTION)->required();
        app->add_option("--gt_file", ctx.gt_file, program_options_utils::GROUND_TRUTH_FILE_DESCRIPTION)->default_val(
                "null");
        app->add_option("-T, --num_threads", ctx.num_threads,
                        program_options_utils::NUMBER_THREADS_DESCRIPTION)->default_val(omp_get_num_procs());
        app->add_flag("--dynamic", ctx.dynamic,
                      "Whether the index is dynamic. Dynamic indices must have associated tags.  Default false.");
        app->add_option("--fail_if_recall_below", ctx.fail_if_recall_below,
                        program_options_utils::FAIL_IF_RECALL_BELOW)->default_val(0.0f);
        app->add_flag("--print_all_recalls", ctx.print_all_recalls,
                      "Print recalls at all positions, from 1 up to specified recall_at value");
        app->add_flag("--print_qps_per_thread", ctx.show_qps_per_thread,
                      "Print overall QPS divided by the number of threads in the output table");
        app->callback(run_search_memory_index_cli);
    }

    void run_search_memory_index_cli() {
        polaris::MetricType metric;
        if ((ctx.dist_fn == std::string("mips")) && (ctx.data_type == std::string("float"))) {
            metric = polaris::MetricType::METRIC_INNER_PRODUCT;
        } else if (ctx.dist_fn == std::string("l2")) {
            metric = polaris::MetricType::METRIC_L2;
        } else if (ctx.dist_fn == std::string("cosine")) {
            metric = polaris::MetricType::METRIC_COSINE;
        } else {
            std::cout << "Unsupported distance function. Currently only l2/ cosine are "
                         "supported in general, and mips/fast_l2 only for floating "
                         "point data."
                      << std::endl;
            exit(-1);
        }

        if (ctx.fail_if_recall_below < 0.0 || ctx.fail_if_recall_below >= 100.0) {
            std::cerr << "fail_if_recall_below parameter must be between 0 and 100%" << std::endl;
            exit(-1);
        }

        int r = 0;
        try {
            if (ctx.data_type == std::string("int8")) {
                r = search_memory_index<int8_t>(
                        metric, ctx.index_path_prefix, ctx.result_path, ctx.query_file, ctx.gt_file,
                        ctx.num_threads, ctx.K,
                        ctx.print_all_recalls,
                        ctx.Lvec, ctx.dynamic, ctx.show_qps_per_thread,
                        ctx.fail_if_recall_below);
            } else if (ctx.data_type == std::string("uint8")) {
                r = search_memory_index<uint8_t>(
                        metric, ctx.index_path_prefix, ctx.result_path, ctx.query_file, ctx.gt_file,
                        ctx.num_threads, ctx.K,
                        ctx.print_all_recalls,
                        ctx.Lvec, ctx.dynamic,  ctx.show_qps_per_thread,
                        ctx.fail_if_recall_below);
            } else if (ctx.data_type == std::string("float")) {
                r = search_memory_index<float>(metric, ctx.index_path_prefix, ctx.result_path,
                                                ctx.query_file,
                                                ctx.gt_file,
                                                ctx.num_threads, ctx.K, ctx.print_all_recalls, ctx.Lvec,
                                                ctx.dynamic,
                                                ctx.show_qps_per_thread,
                                                ctx.fail_if_recall_below);
            } else {
                std::cout << "Unsupported type. Use float/int8/uint8" << std::endl;
                r = -1;
            }
            if (r != 0) {
                std::cerr << "Search failed." << std::endl;
                exit(-1);
            }
        }
        catch (std::exception &e) {
            std::cout << std::string(e.what()) << std::endl;
            polaris::cerr << "VamanaIndex search failed." << std::endl;
            exit(-1);
        }

    }
}  // namespace polaris