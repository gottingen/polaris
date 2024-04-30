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
#include <polaris/graph/utils.h>
#include <polaris/graph/disk_utils.h>
#include <polaris/graph/math_utils.h>
#include <polaris/graph/index.h>
#include <polaris/graph/index_factory.h>
#include <polaris/graph/partition.h>

#include <polaris/core/common_includes.h>
#include <polaris/internal/memory_mapper.h>
#include <polaris/graph/partition.h>
#include <polaris/graph/pq_flash_index.h>
#include <polaris/graph/timer.h>
#include <polaris/graph/percentile_stats.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <polaris/io/linux_aligned_file_reader.h>


template<typename T, typename LabelT = uint32_t>
int search_memory_index(polaris::MetricType &metric, const std::string &index_path, const std::string &result_path_prefix,
                        const std::string &query_file, const std::string &truthset_file, const uint32_t num_threads,
                        const uint32_t recall_at, const bool print_all_recalls, const std::vector<uint32_t> &Lvec,
                        const bool dynamic, const bool tags, const bool show_qps_per_thread,
                        const std::vector<std::string> &query_filters, const float fail_if_recall_below) {
    using TagT = uint32_t;
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

    bool filtered_search = false;
    if (!query_filters.empty()) {
        filtered_search = true;
        if (query_filters.size() != 1 && query_filters.size() != query_num) {
            std::cout << "Error. Mismatch in number of queries and size of query "
                         "filters file"
                      << std::endl;
            return -1; // To return -1 or some other error handling?
        }
    }

    const size_t num_frozen_pts = polaris::get_graph_num_frozen_points(index_path);

    auto config = polaris::IndexConfigBuilder()
            .with_metric(metric)
            .with_dimension(query_dim)
            .with_max_points(0)
            .with_data_load_store_strategy(polaris::DataStoreStrategy::MEMORY)
            .with_graph_load_store_strategy(polaris::GraphStoreStrategy::MEMORY)
            .with_data_type(diskann_type_to_name<T>())
            .with_label_type(diskann_type_to_name<LabelT>())
            .with_tag_type(diskann_type_to_name<TagT>())
            .is_dynamic_index(dynamic)
            .is_enable_tags(tags)
            .is_concurrent_consolidate(false)
            .is_pq_dist_build(false)
            .is_use_opq(false)
            .with_num_pq_chunks(0)
            .with_num_frozen_pts(num_frozen_pts)
            .build();

    auto index_factory = polaris::IndexFactory(config);
    auto index = index_factory.create_instance();
    index->load(index_path.c_str(), num_threads, *(std::max_element(Lvec.begin(), Lvec.end())));
    std::cout << "Index loaded" << std::endl;

    if (metric == polaris::MetricType::METRIC_FAST_L2)
        index->optimize_index_layout();

    std::cout << "Using " << num_threads << " threads to search" << std::endl;
    std::cout.setf(std::ios_base::fixed, std::ios_base::floatfield);
    std::cout.precision(2);
    const std::string qps_title = show_qps_per_thread ? "QPS/thread" : "QPS";
    uint32_t table_width = 0;
    if (tags) {
        std::cout << std::setw(4) << "Ls" << std::setw(12) << qps_title << std::setw(20) << "Mean Latency (mus)"
                  << std::setw(15) << "99.9 Latency";
        table_width += 4 + 12 + 20 + 15;
    } else {
        std::cout << std::setw(4) << "Ls" << std::setw(12) << qps_title << std::setw(18) << "Avg dist cmps"
                  << std::setw(20) << "Mean Latency (mus)" << std::setw(15) << "99.9 Latency";
        table_width += 4 + 12 + 18 + 20 + 15;
    }
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

    std::vector<std::vector<uint32_t>> query_result_ids(Lvec.size());
    std::vector<std::vector<float>> query_result_dists(Lvec.size());
    std::vector<float> latency_stats(query_num, 0);
    std::vector<uint32_t> cmp_stats;
    if (not tags || filtered_search) {
        cmp_stats = std::vector<uint32_t>(query_num, 0);
    }

    std::vector<TagT> query_result_tags;
    if (tags) {
        query_result_tags.resize(recall_at * query_num);
    }

    double best_recall = 0.0;

    for (uint32_t test_id = 0; test_id < Lvec.size(); test_id++) {
        uint32_t L = Lvec[test_id];
        if (L < recall_at) {
            polaris::cout << "Ignoring search with L:" << L << " since it's smaller than K:" << recall_at << std::endl;
            continue;
        }

        query_result_ids[test_id].resize(recall_at * query_num);
        query_result_dists[test_id].resize(recall_at * query_num);
        std::vector<T *> res = std::vector<T *>();

        auto s = std::chrono::high_resolution_clock::now();
        omp_set_num_threads(num_threads);
#pragma omp parallel for schedule(dynamic, 1)
        for (int64_t i = 0; i < (int64_t) query_num; i++) {
            auto qs = std::chrono::high_resolution_clock::now();
            if (filtered_search && !tags) {
                std::string raw_filter = query_filters.size() == 1 ? query_filters[0] : query_filters[i];

                auto retval = index->search_with_filters(query + i * query_aligned_dim, raw_filter, recall_at, L,
                                                         query_result_ids[test_id].data() + i * recall_at,
                                                         query_result_dists[test_id].data() + i * recall_at);
                cmp_stats[i] = retval.second;
            } else if (metric == polaris::MetricType::METRIC_FAST_L2) {
                index->search_with_optimized_layout(query + i * query_aligned_dim, recall_at, L,
                                                    query_result_ids[test_id].data() + i * recall_at);
            } else if (tags) {
                if (!filtered_search) {
                    index->search_with_tags(query + i * query_aligned_dim, recall_at, L,
                                            query_result_tags.data() + i * recall_at, nullptr, res);
                } else {
                    std::string raw_filter = query_filters.size() == 1 ? query_filters[0] : query_filters[i];

                    index->search_with_tags(query + i * query_aligned_dim, recall_at, L,
                                            query_result_tags.data() + i * recall_at, nullptr, res, true, raw_filter);
                }

                for (int64_t r = 0; r < (int64_t) recall_at; r++) {
                    query_result_ids[test_id][recall_at * i + r] = query_result_tags[recall_at * i + r];
                }
            } else {
                cmp_stats[i] = index
                        ->search(query + i * query_aligned_dim, recall_at, L,
                                 query_result_ids[test_id].data() + i * recall_at)
                        .second;
            }
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
                                                            query_result_ids[test_id].data(), recall_at, curr_recall));
            }
        }

        std::sort(latency_stats.begin(), latency_stats.end());
        double mean_latency =
                std::accumulate(latency_stats.begin(), latency_stats.end(), 0.0) / static_cast<float>(query_num);

        float avg_cmps = (float) std::accumulate(cmp_stats.begin(), cmp_stats.end(), 0) / (float) query_num;

        if (tags && !filtered_search) {
            std::cout << std::setw(4) << L << std::setw(12) << displayed_qps << std::setw(20) << (float) mean_latency
                      << std::setw(15) << (float) latency_stats[(uint64_t) (0.999 * query_num)];
        } else {
            std::cout << std::setw(4) << L << std::setw(12) << displayed_qps << std::setw(18) << avg_cmps
                      << std::setw(20) << (float) mean_latency << std::setw(15)
                      << (float) latency_stats[(uint64_t) (0.999 * query_num)];
        }
        for (double recall: recalls) {
            std::cout << std::setw(12) << recall;
            best_recall = std::max(recall, best_recall);
        }
        std::cout << std::endl;
    }

    std::cout << "Done searching. Now saving results " << std::endl;
    uint64_t test_id = 0;
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
    }

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
        std::string filter_label;
        std::string label_type;
        std::string query_filters_file;
        uint32_t num_threads;
        uint32_t K;
        std::vector<uint32_t> Lvec;
        bool print_all_recalls;
        bool dynamic;
        bool tags;
        bool show_qps_per_thread;
        float fail_if_recall_below;
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
        app->add_option("--filter_label", ctx.filter_label, program_options_utils::FILTER_LABEL_DESCRIPTION)->default_val(
                "");
        app->add_option("--query_filters_file", ctx.query_filters_file,
                        program_options_utils::FILTERS_FILE_DESCRIPTION)->default_val("");
        app->add_option("--label_type", ctx.label_type, program_options_utils::LABEL_TYPE_DESCRIPTION)->default_val(
                "uint");
        app->add_option("--gt_file", ctx.gt_file, program_options_utils::GROUND_TRUTH_FILE_DESCRIPTION)->default_val(
                "null");
        app->add_option("-T, --num_threads", ctx.num_threads,
                        program_options_utils::NUMBER_THREADS_DESCRIPTION)->default_val(omp_get_num_procs());
        app->add_flag("--dynamic", ctx.dynamic,
                      "Whether the index is dynamic. Dynamic indices must have associated tags.  Default false.");
        app->add_flag("--tags", ctx.tags, "Whether to search with external identifiers (tags). Default false.");
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
        } else if ((ctx.dist_fn == std::string("fast_l2")) && (ctx.data_type == std::string("float"))) {
            metric = polaris::MetricType::METRIC_FAST_L2;
        } else {
            std::cout << "Unsupported distance function. Currently only l2/ cosine are "
                         "supported in general, and mips/fast_l2 only for floating "
                         "point data."
                      << std::endl;
            exit(-1);
        }

        if (ctx.dynamic && !ctx.tags) {
            std::cerr << "Tags must be enabled while searching dynamically built indices" << std::endl;
            exit(-1);
        }

        if (ctx.fail_if_recall_below < 0.0 || ctx.fail_if_recall_below >= 100.0) {
            std::cerr << "fail_if_recall_below parameter must be between 0 and 100%" << std::endl;
            exit(-1);
        }

        if (ctx.filter_label != "" && ctx.query_filters_file != "") {
            std::cerr << "Only one of filter_label and query_filters_file should be provided" << std::endl;
            exit(-1);
        }

        std::vector<std::string> query_filters;
        if (ctx.filter_label != "") {
            query_filters.push_back(ctx.filter_label);
        } else if (ctx.query_filters_file != "") {
            query_filters = read_file_to_vector_of_strings(ctx.query_filters_file);
        }

        int r = 0;
        try {
            if (!query_filters.empty() && ctx.label_type == "ushort") {
                if (ctx.data_type == std::string("int8")) {
                    r = search_memory_index<int8_t, uint16_t>(
                            metric, ctx.index_path_prefix, ctx.result_path, ctx.query_file, ctx.gt_file,
                            ctx.num_threads, ctx.K,
                            ctx.print_all_recalls,
                            ctx.Lvec, ctx.dynamic, ctx.tags, ctx.show_qps_per_thread, query_filters,
                            ctx.fail_if_recall_below);
                } else if (ctx.data_type == std::string("uint8")) {
                    r = search_memory_index<uint8_t, uint16_t>(
                            metric, ctx.index_path_prefix, ctx.result_path, ctx.query_file, ctx.gt_file,
                            ctx.num_threads, ctx.K,
                            ctx.print_all_recalls,
                            ctx.Lvec, ctx.dynamic, ctx.tags, ctx.show_qps_per_thread, query_filters,
                            ctx.fail_if_recall_below);
                } else if (ctx.data_type == std::string("float")) {
                    r = search_memory_index<float, uint16_t>(metric, ctx.index_path_prefix, ctx.result_path,
                                                             ctx.query_file,
                                                             ctx.gt_file,
                                                             ctx.num_threads, ctx.K, ctx.print_all_recalls, ctx.Lvec,
                                                             ctx.dynamic, ctx.tags,
                                                             ctx.show_qps_per_thread, query_filters,
                                                             ctx.fail_if_recall_below);
                } else {
                    std::cout << "Unsupported type. Use float/int8/uint8" << std::endl;
                    r = -1;
                }
            } else {
                if (ctx.data_type == std::string("int8")) {
                    r = search_memory_index<int8_t>(metric, ctx.index_path_prefix, ctx.result_path, ctx.query_file,
                                                    ctx.gt_file,
                                                    ctx.num_threads, ctx.K, ctx.print_all_recalls, ctx.Lvec,
                                                    ctx.dynamic, ctx.tags,
                                                    ctx.show_qps_per_thread, query_filters, ctx.fail_if_recall_below);
                } else if (ctx.data_type == std::string("uint8")) {
                    r = search_memory_index<uint8_t>(metric, ctx.index_path_prefix, ctx.result_path, ctx.query_file,
                                                     ctx.gt_file,
                                                     ctx.num_threads, ctx.K, ctx.print_all_recalls, ctx.Lvec,
                                                     ctx.dynamic, ctx.tags,
                                                     ctx.show_qps_per_thread, query_filters, ctx.fail_if_recall_below);
                } else if (ctx.data_type == std::string("float")) {
                    r = search_memory_index<float>(metric, ctx.index_path_prefix, ctx.result_path, ctx.query_file,
                                                   ctx.gt_file,
                                                   ctx.num_threads, ctx.K, ctx.print_all_recalls, ctx.Lvec, ctx.dynamic,
                                                   ctx.tags,
                                                   ctx.show_qps_per_thread, query_filters, ctx.fail_if_recall_below);
                } else {
                    std::cout << "Unsupported type. Use float/int8/uint8" << std::endl;
                    r = -1;
                }
            }

            if (r != 0) {
                std::cerr << "Search failed." << std::endl;
                exit(-1);
            }
        }
        catch (std::exception &e) {
            std::cout << std::string(e.what()) << std::endl;
            polaris::cerr << "Index search failed." << std::endl;
            exit(-1);
        }

    }
}  // namespace polaris