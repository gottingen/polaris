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

#include <iostream>
#include <random>
#include <math.h>
#include <cmath>
#include <polaris/graph/utils.h>
#include <collie/cli/cli.h>

namespace polaris {
    class ZipfDistribution {
    public:
        ZipfDistribution(uint64_t num_points, uint32_t num_labels)
                : num_labels(num_labels), num_points(num_points),
                  uniform_zero_to_one(std::uniform_real_distribution<>(0.0, 1.0)) {
        }

        std::unordered_map<uint32_t, uint32_t> createDistributionMap() {
            std::unordered_map<uint32_t, uint32_t> map;
            uint32_t primary_label_freq = (uint32_t) ceil(num_points * distribution_factor);
            for (uint32_t i{1}; i < num_labels + 1; i++) {
                map[i] = (uint32_t) ceil(primary_label_freq / i);
            }
            return map;
        }

        int writeDistribution(std::ofstream &outfile) {
            auto distribution_map = createDistributionMap();
            for (uint32_t i{0}; i < num_points; i++) {
                bool label_written = false;
                for (auto it = distribution_map.cbegin(); it != distribution_map.cend(); it++) {
                    auto label_selection_probability = std::bernoulli_distribution(
                            distribution_factor / (double) it->first);
                    if (label_selection_probability(rand_engine) && distribution_map[it->first] > 0) {
                        if (label_written) {
                            outfile << ',';
                        }
                        outfile << it->first;
                        label_written = true;
                        // remove label from map if we have used all labels
                        distribution_map[it->first] -= 1;
                    }
                }
                if (!label_written) {
                    outfile << 0;
                }
                if (i < num_points - 1) {
                    outfile << '\n';
                }
            }
            return 0;
        }

        int writeDistribution(std::string filename) {
            std::ofstream outfile(filename);
            if (!outfile.is_open()) {
                std::cerr << "Error: could not open output file " << filename << '\n';
                return -1;
            }
            writeDistribution(outfile);
            outfile.close();
        }

    private:
        const uint32_t num_labels;
        const uint64_t num_points;
        const double distribution_factor = 0.7;
        std::knuth_b rand_engine;
        const std::uniform_real_distribution<double> uniform_zero_to_one;
    };
    std::string output_file, distribution_type;
    uint32_t num_labels;
    uint64_t num_points;
    void run_generate_synthetic_labels_cli();
    void setup_generate_synthetic_labels_cli(collie::App *app) {


        app->add_option("--output_file", output_file, "Filename for saving the label file")->required(true);
        app->add_option("--num_points", num_points, "Number of points in dataset")->required(true);
        app->add_option("--num_labels", num_labels, "Number of unique labels, up to 5000")->required(true);
        app->add_option("--distribution_type", distribution_type,
                        "Distribution function for labels <random/zipf/one_per_point> defaults to random")->default_val(
                "random");
        app->callback([]() { return run_generate_synthetic_labels_cli(); });

    }
    void run_generate_synthetic_labels_cli() {
        if (num_labels > 5000) {
            std::cerr << "Error: num_labels must be 5000 or less" << '\n';
            exit(-1);
        }

        if (num_points <= 0) {
            std::cerr << "Error: num_points must be greater than 0" << '\n';
            exit(-1);
        }

        std::cout << "Generating synthetic labels for " << num_points << " points with " << num_labels
                  << " unique labels"
                  << '\n';

        try {
            std::ofstream outfile(output_file);
            if (!outfile.is_open()) {
                std::cerr << "Error: could not open output file " << output_file << '\n';
                exit(-1);
            }

            if (distribution_type == "zipf") {
                ZipfDistribution zipf(num_points, num_labels);
                zipf.writeDistribution(outfile);
            } else if (distribution_type == "random") {
                for (size_t i = 0; i < num_points; i++) {
                    bool label_written = false;
                    for (size_t j = 1; j <= num_labels; j++) {
                        // 50% chance to assign each label
                        if (rand() > (RAND_MAX / 2)) {
                            if (label_written) {
                                outfile << ',';
                            }
                            outfile << j;
                            label_written = true;
                        }
                    }
                    if (!label_written) {
                        outfile << 0;
                    }
                    if (i < num_points - 1) {
                        outfile << '\n';
                    }
                }
            } else if (distribution_type == "one_per_point") {
                std::random_device rd;                                // obtain a random number from hardware
                std::mt19937 gen(rd());                               // seed the generator
                std::uniform_int_distribution<> distr(0, num_labels); // define the range

                for (size_t i = 0; i < num_points; i++) {
                    outfile << distr(gen);
                    if (i != num_points - 1)
                        outfile << '\n';
                }
            }
            if (outfile.is_open()) {
                outfile.close();
            }

            std::cout << "Labels written to " << output_file << '\n';
        }
        catch (const std::exception &ex) {
            std::cerr << "Label generation failed: " << ex.what() << '\n';
            exit(-1);
        }
    }
}