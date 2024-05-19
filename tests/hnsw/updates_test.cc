// Copyright 2024 The EA Authors.
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
#include <polaris/graph/hnsw/hnswlib.h>
#include <polaris/core/common.h>
#include <thread>


class StopW {
    std::chrono::steady_clock::time_point time_begin;

 public:
    StopW() {
        time_begin = std::chrono::steady_clock::now();
    }

    float getElapsedTimeMicro() {
        std::chrono::steady_clock::time_point time_end = std::chrono::steady_clock::now();
        return (std::chrono::duration_cast<std::chrono::microseconds>(time_end - time_begin).count());
    }

    void reset() {
        time_begin = std::chrono::steady_clock::now();
    }
};


/*
 * replacement for the openmp '#pragma omp parallel for' directive
 * only handles a subset of functionality (no reductions etc)
 * Process ids from start (inclusive) to end (EXCLUSIVE)
 *
 * The method is borrowed from nmslib 
 */
template<class Function>
inline void ParallelFor(size_t start, size_t end, size_t numThreads, Function fn) {
    if (numThreads <= 0) {
        numThreads = std::thread::hardware_concurrency();
    }

    if (numThreads == 1) {
        for (size_t id = start; id < end; id++) {
            fn(id, 0);
        }
    } else {
        std::vector<std::thread> threads;
        std::atomic<size_t> current(start);

        // keep track of exceptions in threads
        // https://stackoverflow.com/a/32428427/1713196
        std::exception_ptr lastException = nullptr;
        std::mutex lastExceptMutex;

        for (size_t threadId = 0; threadId < numThreads; ++threadId) {
            threads.push_back(std::thread([&, threadId] {
                while (true) {
                    size_t id = current.fetch_add(1);

                    if ((id >= end)) {
                        break;
                    }

                    try {
                        fn(id, threadId);
                    } catch (...) {
                        std::unique_lock<std::mutex> lastExcepLock(lastExceptMutex);
                        lastException = std::current_exception();
                        /*
                         * This will work even when current is the largest value that
                         * size_t can fit, because fetch_add returns the previous value
                         * before the increment (what will result in overflow
                         * and produce 0 instead of current + 1).
                         */
                        current = end;
                        break;
                    }
                }
            }));
        }
        for (auto &thread : threads) {
            thread.join();
        }
        if (lastException) {
            std::rethrow_exception(lastException);
        }
    }
}


template <typename datatype>
std::vector<datatype> load_batch(std::string path, int size) {
    std::cout << "Loading " << path << "...";
    // float or int32 (python)
    assert(sizeof(datatype) == 4);

    std::ifstream file;
    file.open(path, std::ios::binary);
    if (!file.is_open()) {
        std::cout << "Cannot open " << path << "\n";
        exit(1);
    }
    std::vector<datatype> batch(size);

    file.read((char *)batch.data(), size * sizeof(float));
    std::cout << " DONE\n";
    return batch;
}


template <typename d_type>
static float
test_approx(std::vector<float> &queries, size_t qsize, hnswlib::HierarchicalNSW<d_type> &appr_alg, size_t vecdim,
            std::vector<std::unordered_set<polaris::vid_t>> &answers, size_t K) {
    size_t correct = 0;
    size_t total = 0;

    for (int i = 0; i < qsize; i++) {
        std::priority_queue<std::pair<d_type, polaris::vid_t>> result = appr_alg.searchKnn((char *)(queries.data() + vecdim * i), K);
        total += K;
        while (result.size()) {
            if (answers[i].find(result.top().second) != answers[i].end()) {
                correct++;
            } else {
            }
            result.pop();
        }
    }
    return 1.0f * correct / total;
}


static void
test_vs_recall(
    std::vector<float> &queries,
    size_t qsize,
    hnswlib::HierarchicalNSW<float> &appr_alg,
    size_t vecdim,
    std::vector<std::unordered_set<polaris::vid_t>> &answers,
    size_t k) {

    std::vector<size_t> efs = {1};
    for (int i = k; i < 30; i++) {
        efs.push_back(i);
    }
    for (int i = 30; i < 400; i+=10) {
        efs.push_back(i);
    }
    for (int i = 1000; i < 100000; i += 5000) {
        efs.push_back(i);
    }
    std::cout << "ef\trecall\ttime\thops\tdistcomp\n";

    bool test_passed = false;
    for (size_t ef : efs) {
        appr_alg.setEf(ef);

        appr_alg.metric_hops = 0;
        appr_alg.metric_distance_computations = 0;
        StopW stopw = StopW();

        float recall = test_approx<float>(queries, qsize, appr_alg, vecdim, answers, k);
        float time_us_per_query = stopw.getElapsedTimeMicro() / qsize;
        float distance_comp_per_query =  appr_alg.metric_distance_computations / (1.0f * qsize);
        float hops_per_query =  appr_alg.metric_hops / (1.0f * qsize);

        std::cout << ef << "\t" << recall << "\t" << time_us_per_query << "us \t" << hops_per_query << "\t" << distance_comp_per_query << "\n";
        if (recall > 0.99) {
            test_passed = true;
            std::cout << "Recall is over 0.99! " << recall << "\t" << time_us_per_query << "us \t" << hops_per_query << "\t" << distance_comp_per_query << "\n";
            break;
        }
    }
    if (!test_passed) {
        std::cerr << "Test failed\n";
        exit(1);
    }
}


int main(int argc, char **argv) {
    int M = 16;
    int efConstruction = 200;
    int num_threads = std::thread::hardware_concurrency();

    bool update = false;

    if (argc == 2) {
        if (std::string(argv[1]) == "update") {
            update = true;
            std::cout << "Updates are on\n";
        } else {
            std::cout << "Usage ./test_updates [update]\n";
            exit(1);
        }
    } else if (argc > 2) {
        std::cout << "Usage ./test_updates [update]\n";
        exit(1);
    }

    std::string path = "../tests/cpp/data/";

    int N;
    int dummy_data_multiplier;
    int N_queries;
    int d;
    int K;
    {
        std::ifstream configfile;
        configfile.open(path + "/config.txt");
        if (!configfile.is_open()) {
            std::cout << "Cannot open config.txt\n";
            return 1;
        }
        configfile >> N >> dummy_data_multiplier >> N_queries >> d >> K;

        printf("Loaded config: N=%d, d_mult=%d, Nq=%d, dim=%d, K=%d\n", N, dummy_data_multiplier, N_queries, d, K);
    }

    hnswlib::L2Space l2space(d);
    hnswlib::HierarchicalNSW<float> appr_alg(&l2space);
    auto rs = appr_alg.initialize(&l2space, N + 1, M, efConstruction);
    if (!rs.ok()) {
        std::cerr << "Failed to initialize hnsw algorithm\n";
        return 1;
    }

    std::vector<float> dummy_batch = load_batch<float>(path + "batch_dummy_00.bin", N * d);

    // Adding enterpoint:

    rs = appr_alg.addPoint((void *)dummy_batch.data(), (size_t)0);
    if (!rs.ok()) {
        std::cerr << "Failed to add enterpoint\n";
        return 1;
    }

    StopW stopw = StopW();

    if (update) {
        std::cout << "Update iteration 0\n";

        ParallelFor(1, N, num_threads, [&](size_t i, size_t threadId) {
            auto rs = appr_alg.addPoint((void *)(dummy_batch.data() + i * d), i);
            if (!rs.ok()) {
                std::cerr << "Failed to add element " << i << "\n";
                exit(1);
            }
        });
        appr_alg.checkIntegrity();

        ParallelFor(1, N, num_threads, [&](size_t i, size_t threadId) {
            auto rs = appr_alg.addPoint((void *)(dummy_batch.data() + i * d), i);
            if (!rs.ok()) {
                std::cerr << "Failed to add element " << i << "\n";
                exit(1);
            }
        });
        appr_alg.checkIntegrity();

        for (int b = 1; b < dummy_data_multiplier; b++) {
            std::cout << "Update iteration " << b << "\n";
            char cpath[1024];
            snprintf(cpath, sizeof(cpath), "batch_dummy_%02d.bin", b);
            std::vector<float> dummy_batchb = load_batch<float>(path + cpath, N * d);

            ParallelFor(0, N, num_threads, [&](size_t i, size_t threadId) {
                auto rs = appr_alg.addPoint((void *)(dummy_batch.data() + i * d), i);
                if (!rs.ok()) {
                    std::cerr << "Failed to add element " << i << "\n";
                    exit(1);
                }
            });
            appr_alg.checkIntegrity();
        }
    }

    std::cout << "Inserting final elements\n";
    std::vector<float> final_batch = load_batch<float>(path + "batch_final.bin", N * d);

    stopw.reset();
    ParallelFor(0, N, num_threads, [&](size_t i, size_t threadId) {
                    auto rs = appr_alg.addPoint((void *)(final_batch.data() + i * d), i);
                    if (!rs.ok()) {
                        std::cerr << "Failed to add element " << i << "\n";
                        exit(1);
                    }
                });
    std::cout << "Finished. Time taken:" << stopw.getElapsedTimeMicro()*1e-6 << " s\n";
    std::cout << "Running tests\n";
    std::vector<float> queries_batch = load_batch<float>(path + "queries.bin", N_queries * d);

    std::vector<int> gt = load_batch<int>(path + "gt.bin", N_queries * K);

    std::vector<std::unordered_set<polaris::vid_t>> answers(N_queries);
    for (int i = 0; i < N_queries; i++) {
        for (int j = 0; j < K; j++) {
            answers[i].insert(gt[i * K + j]);
        }
    }

    for (int i = 0; i < 3; i++) {
        std::cout << "Test iteration " << i << "\n";
        test_vs_recall(queries_batch, N_queries, appr_alg, d, answers, K);
    }

    return 0;
}
