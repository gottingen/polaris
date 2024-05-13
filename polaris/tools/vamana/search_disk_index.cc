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
#include <polaris/core/log.h>
#include <polaris/tools/vamana/vamana.h>
#include <polaris/tools/vamana/program_options_utils.h>
#include <polaris/utility/common_includes.h>
#include <polaris/utility/recall.h>
#include <polaris/utility/timer.h>
#include <polaris/core/percentile_stats.h>
#include <polaris/datasets/bin.h>
#include <polaris/unified_index.h>

#define WARMUP false

void print_stats(std::string category, std::vector<float> percentiles, std::vector<float> results) {
    polaris::cout << std::setw(20) << category << ": " << std::flush;
    for (uint32_t s = 0; s < percentiles.size(); s++) {
        polaris::cout << std::setw(8) << percentiles[s] << "%";
    }
    polaris::cout << std::endl;
    polaris::cout << std::setw(22) << " " << std::flush;
    for (uint32_t s = 0; s < percentiles.size(); s++) {
        polaris::cout << std::setw(9) << results[s];
    }
    polaris::cout << std::endl;
}

template<typename T>
int search_disk_index(polaris::MetricType &metric, const std::string &index_path_prefix,
                      const std::string &result_output_prefix, const std::string &query_file, std::string &gt_file,
                      const uint32_t num_threads, const uint32_t recall_at, const uint32_t beamwidth,
                      const uint32_t num_nodes_to_cache, const uint32_t search_io_limit,
                      const std::vector<uint32_t> &Lvec, const float fail_if_recall_below,
                      const bool use_reorder_data = false) {
    polaris::cout << "Search parameters: #threads: " << num_threads << ", ";
    if (beamwidth <= 0)
        polaris::cout << "beamwidth to be optimized for each L value" << std::flush;
    else
        polaris::cout << " beamwidth: " << beamwidth << std::flush;
    if (search_io_limit == std::numeric_limits<uint32_t>::max())
        polaris::cout << "." << std::endl;
    else
        polaris::cout << ", io_limit: " << search_io_limit << "." << std::endl;

    std::string warmup_query_file = index_path_prefix + "_sample_data.bin";

    // load query bin
    T *query = nullptr;
    uint32_t *gt_ids = nullptr;
    float *gt_dists = nullptr;
    size_t query_num, query_dim, query_aligned_dim, gt_num, gt_dim;
    polaris::load_aligned_bin<T>(query_file, query, query_num, query_dim, query_aligned_dim);

    bool calc_recall_flag = false;
    if (gt_file != std::string("null") && gt_file != std::string("NULL") && collie::filesystem::exists(gt_file)) {
        polaris::load_truthset(gt_file, gt_ids, gt_dists, gt_num, gt_dim);
        if (gt_num != query_num) {
            polaris::cout << "Error. Mismatch in number of queries and ground truth data" << std::endl;
        }
        calc_recall_flag = true;
    }

    POLARIS_LOG(INFO) << "Loading index from " << index_path_prefix;
    std::unique_ptr<polaris::UnifiedIndex> unified_index(polaris::UnifiedIndex::create_index(polaris::IndexType::IT_VAMANA_DISK));
    polaris::IndexConfig config = polaris::IndexConfigBuilder()
            .with_load_threads(num_threads)
            .with_metric(metric)
            .with_data_type(polaris::polaris_type_to_name<T>())
            .vdisk_with_num_nodes_to_cache(num_nodes_to_cache)
            .build_vdisk();
    unified_index->initialize(config);
    auto res = unified_index->load(index_path_prefix.c_str());
    if (!res.ok()) {
        POLARIS_LOG(ERROR) << "Failed to load index: " << res.message();
        return -1;
    }

    omp_set_num_threads(num_threads);

    uint64_t warmup_L = 20;
    uint64_t warmup_num = 0, warmup_dim = 0, warmup_aligned_dim = 0;
    T *warmup = nullptr;
/*
    if (WARMUP) {
        if (collie::filesystem::exists(warmup_query_file)) {
            polaris::load_aligned_bin<T>(warmup_query_file, warmup, warmup_num, warmup_dim, warmup_aligned_dim);
        } else {
            warmup_num = (std::min)((uint32_t) 150000, (uint32_t) 15000 * num_threads);
            warmup_dim = query_dim;
            warmup_aligned_dim = query_aligned_dim;
            polaris::alloc_aligned(((void **) &warmup), warmup_num * warmup_aligned_dim * sizeof(T), 8 * sizeof(T));
            std::memset(warmup, 0, warmup_num * warmup_aligned_dim * sizeof(T));
            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_int_distribution<> dis(-128, 127);
            for (uint32_t i = 0; i < warmup_num; i++) {
                for (uint32_t d = 0; d < warmup_dim; d++) {
                    warmup[i * warmup_aligned_dim + d] = (T) dis(gen);
                }
            }
        }
        polaris::cout << "Warming up index... " << std::flush;
        std::vector<uint64_t> warmup_result_ids_64(warmup_num, 0);
        std::vector<float> warmup_result_dists(warmup_num, 0);

#pragma omp parallel for schedule(dynamic, 1)
        for (int64_t i = 0; i < (int64_t) warmup_num; i++) {
            _pFlashIndex->cached_beam_search(warmup + (i * warmup_aligned_dim), 1, warmup_L,
                                             warmup_result_ids_64.data() + (i * 1),
                                             warmup_result_dists.data() + (i * 1), 4);
        }
        polaris::cout << "..done" << std::endl;
    }
*/
    polaris::cout.setf(std::ios_base::fixed, std::ios_base::floatfield);
    polaris::cout.precision(2);

    std::string recall_string = "Recall@" + std::to_string(recall_at);
    polaris::cout << std::setw(6) << "L" << std::setw(12) << "Beamwidth" << std::setw(16) << "QPS" << std::setw(16)
                  << "Mean Latency" << std::setw(16) << "99.9 Latency" << std::setw(16) << "Mean IOs" << std::setw(16)
                  << "CPU (s)";
    if (calc_recall_flag) {
        polaris::cout << std::setw(16) << recall_string << std::endl;
    } else
        polaris::cout << std::endl;
    polaris::cout << "==============================================================="
                     "======================================================="
                  << std::endl;
    std::vector<std::vector<std::unique_ptr<polaris::SearchContext>>> search_contexts(Lvec.size());

    uint32_t optimized_beamwidth = 2;

    double best_recall = 0.0;

    for (uint32_t test_id = 0; test_id < Lvec.size(); test_id++) {
        uint32_t L = Lvec[test_id];
        search_contexts[test_id].resize(query_num);
        for (int i = 0; i < query_num; i++) {
            search_contexts[test_id][i] = std::make_unique<polaris::SearchContext>();
        }
        if (L < recall_at) {
            polaris::cout << "Ignoring search with L:" << L << " since it's smaller than K:" << recall_at << std::endl;
            continue;
        }

        if (beamwidth <= 0) {
            polaris::cout << "Tuning beamwidth.." << std::endl;
            auto rs = unified_index->optimize_beam_width(warmup, warmup_num, warmup_aligned_dim, L, optimized_beamwidth);
            if(!rs.ok()) {
                POLARIS_LOG(ERROR) << "Failed to optimize beamwidth";
                exit(-1);
            }
            optimized_beamwidth = rs.value();

        } else
            optimized_beamwidth = beamwidth;

        auto stats = new polaris::QueryStats[query_num];

        auto s = std::chrono::high_resolution_clock::now();

#pragma omp parallel for schedule(dynamic, 1)
        for (int64_t i = 0; i < (int64_t) query_num; i++) {
            auto &ctx = *search_contexts[test_id][i];
            ctx.set_meta(polaris::polaris_type_to_name<T>(), query_aligned_dim)
                    .set_query(query + (i * query_aligned_dim), query_aligned_dim * sizeof(T))
                    .set_top_k(recall_at)
                    .set_search_list(L)
                    .set_beam_width(optimized_beamwidth)
                    .set_with_local_ids(true)
                    .set_query_stats(stats + i)
                    .set_use_reorder_data(use_reorder_data);
            auto rs = unified_index->search(ctx);
            if (!rs.ok()) {
                polaris::cerr << "Search failed for query " << i << std::endl;
                exit(-1);
            }
        }
        auto e = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> diff = e - s;
        double qps = (1.0 * query_num) / (1.0 * diff.count());

        auto mean_latency = polaris::get_mean_stats<float>(
                stats, query_num, [](const polaris::QueryStats &stats) { return stats.total_us; });

        auto latency_999 = polaris::get_percentile_stats<float>(
                stats, query_num, 0.999, [](const polaris::QueryStats &stats) { return stats.total_us; });

        auto mean_ios = polaris::get_mean_stats<uint32_t>(stats, query_num,
                                                          [](const polaris::QueryStats &stats) { return stats.n_ios; });

        auto mean_cpuus = polaris::get_mean_stats<float>(stats, query_num,
                                                         [](const polaris::QueryStats &stats) { return stats.cpu_us; });

        double recall = 0;
        if (calc_recall_flag) {
            recall = polaris::calculate_recall((uint32_t) query_num, gt_ids, gt_dists, (uint32_t) gt_dim,
                                               search_contexts[test_id], recall_at, recall_at);
            best_recall = std::max(recall, best_recall);
        }

        polaris::cout << std::setw(6) << L << std::setw(12) << optimized_beamwidth << std::setw(16) << qps
                      << std::setw(16) << mean_latency << std::setw(16) << latency_999 << std::setw(16) << mean_ios
                      << std::setw(16) << mean_cpuus;
        if (calc_recall_flag) {
            polaris::cout << std::setw(16) << recall << std::endl;
        } else
            polaris::cout << std::endl;
        delete[] stats;
    }

    polaris::cout << "Done searching. Now saving results " << std::endl;
    uint64_t test_id = 0;
    /*
    for (auto L: Lvec) {
        if (L < recall_at)
            continue;

        std::string cur_result_path = result_output_prefix + "_" + std::to_string(L) + "_idx_uint32.bin";
        polaris::save_bin<uint32_t>(cur_result_path, search_contexts[test_id]->data(), query_num, recall_at);

        cur_result_path = result_output_prefix + "_" + std::to_string(L) + "_dists_float.bin";
        polaris::save_bin<float>(cur_result_path, query_result_dists[test_id++].data(), query_num, recall_at);
    }
    */
    polaris::aligned_free(query);
    if (warmup != nullptr)
        polaris::aligned_free(warmup);
    return best_recall >= fail_if_recall_below ? 0 : -1;
}

namespace polaris {
    struct SearchDiskIndexContext {
        std::string data_type;
        std::string dist_fn;
        std::string index_path_prefix;
        std::string result_path_prefix;
        std::string query_file;
        std::string gt_file;
        uint32_t num_threads;
        uint32_t K;
        uint32_t W;
        uint32_t num_nodes_to_cache;
        uint32_t search_io_limit;
        std::vector<uint32_t> Lvec;
        bool use_reorder_data;
        float fail_if_recall_below;
    };

    static SearchDiskIndexContext ctx;

    void run_search_disk_index_cli();

    void setup_search_disk_index_cli(collie::App *app) {

        app->add_option("--data_type", ctx.data_type, program_options_utils::DATA_TYPE_DESCRIPTION)->required(true);
        app->add_option("--dist_fn", ctx.dist_fn, program_options_utils::DISTANCE_FUNCTION_DESCRIPTION)->required(true);
        app->add_option("--index_path_prefix", ctx.index_path_prefix,
                        program_options_utils::INDEX_PATH_PREFIX_DESCRIPTION)->required(true);
        app->add_option("--result_path", ctx.result_path_prefix,
                        program_options_utils::RESULT_PATH_DESCRIPTION)->required(true);

        app->add_option("--query_file", ctx.query_file, program_options_utils::QUERY_FILE_DESCRIPTION)->required(true);
        app->add_option("-K, --recall_at", ctx.K, program_options_utils::NUMBER_OF_RESULTS_DESCRIPTION)->required(true);
        app->add_option("-L, --search_list", ctx.Lvec, program_options_utils::SEARCH_LIST_DESCRIPTION)->required(true);

        app->add_option("--gt_file", ctx.gt_file, program_options_utils::GROUND_TRUTH_FILE_DESCRIPTION)->default_val(
                "null");
        app->add_option("--beamwidth", ctx.W, program_options_utils::BEAMWIDTH)->default_val(2);
        app->add_option("--num_nodes_to_cache", ctx.num_nodes_to_cache,
                        program_options_utils::NUMBER_OF_NODES_TO_CACHE)->default_val(0);
        app->add_option("--search_io_limit", ctx.search_io_limit,
                        "Max #IOs for search.  Default value: uint32::max()")->default_val(
                std::numeric_limits<uint32_t>::max());
        app->add_option("--num_threads", ctx.num_threads,
                        program_options_utils::NUMBER_THREADS_DESCRIPTION)->default_val(
                omp_get_num_procs());
        app->add_flag("--use_reorder_data", ctx.use_reorder_data,
                      "Include full precision data in the index. Use only in conjuction with compressed data on SSD.  Default value: false");
        app->add_option("--fail_if_recall_below", ctx.fail_if_recall_below,
                        program_options_utils::FAIL_IF_RECALL_BELOW)->default_val(0.0f);
        app->callback(run_search_disk_index_cli);

    }

    void run_search_disk_index_cli() {
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

        if ((ctx.data_type != std::string("float")) && (metric == polaris::MetricType::METRIC_INNER_PRODUCT)) {
            std::cout << "Currently support only floating point data for Inner Product." << std::endl;
            exit(-1);
        }

        if (ctx.use_reorder_data && ctx.data_type != std::string("float")) {
            std::cout << "Error: Reorder data for reordering currently only "
                         "supported for float data type."
                      << std::endl;
            exit(-1);
        }


        try {
            int r;
            if (ctx.data_type == std::string("float"))
                r = search_disk_index<float>(metric, ctx.index_path_prefix, ctx.result_path_prefix, ctx.query_file,
                                             ctx.gt_file,
                                             ctx.num_threads, ctx.K, ctx.W, ctx.num_nodes_to_cache,
                                             ctx.search_io_limit,
                                             ctx.Lvec,
                                             ctx.fail_if_recall_below, ctx.use_reorder_data);
            else if (ctx.data_type == std::string("int8"))
                r = search_disk_index<int8_t>(metric, ctx.index_path_prefix, ctx.result_path_prefix, ctx.query_file,
                                              ctx.gt_file,
                                              ctx.num_threads, ctx.K, ctx.W, ctx.num_nodes_to_cache,
                                              ctx.search_io_limit, ctx.Lvec,
                                              ctx.fail_if_recall_below, ctx.use_reorder_data);
            else if (ctx.data_type == std::string("uint8"))
                r = search_disk_index<uint8_t>(metric, ctx.index_path_prefix, ctx.result_path_prefix,
                                               ctx.query_file,
                                               ctx.gt_file,
                                               ctx.num_threads, ctx.K, ctx.W, ctx.num_nodes_to_cache,
                                               ctx.search_io_limit, ctx.Lvec,
                                               ctx.fail_if_recall_below, ctx.use_reorder_data);
            else {
                std::cerr << "Unsupported data type. Use float or int8 or uint8" << std::endl;
                r = -1;
            }

            if (r != 0) {
                exit(-1);
            }
        }
        catch (const std::exception &e) {
            std::cout << std::string(e.what()) << std::endl;
            polaris::cerr << "VamanaIndex search failed." << std::endl;
            exit(-1);
        }
    }
}  // namespace polaris