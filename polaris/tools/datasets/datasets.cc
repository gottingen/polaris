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

#include <polaris/tools/datasets/datasets.h>
#include <polaris/datasets/fvecs.h>
#include <polaris/datasets/ivecs.h>
#include <polaris/datasets/to_float.h>
#include <polaris/datasets/tsv.h>
#include <polaris/datasets/random.h>
#include <polaris/datasets/recall.h>

namespace polaris {

    struct DatasetsContext {

        std::string input;
        std::string output;
        float bias{0.0};
        float scale{0.0};
        uint32_t ndims{0};
        uint32_t nvec{0};
        uint32_t num_pq_chunks{0};
        float sampling_rate{0.0};
        bool opq{false};
        uint32_t recall{0};
    };

    DatasetsContext datasets_ctx;


    void setup_datasets_cli(collie::App *app) {
        auto compute_ground_truth = app->add_subcommand("compute_ground_truth", "Compute ground truth");
        setup_compute_ground_truth_cli(compute_ground_truth);
        auto generate_synthetic_labels = app->add_subcommand("generate_synthetic_labels", "Generate synthetic labels");
        setup_generate_synthetic_labels_cli(generate_synthetic_labels);
        /// float_vecs_to_bin
        auto *fvecs_to_bin = app->add_subcommand("float_vecs_to_bin", "Convert float vectors to binary format");
        fvecs_to_bin->add_option("-i, --input", datasets_ctx.input, "Input file path")->required(true);
        fvecs_to_bin->add_option("-o, --output", datasets_ctx.output, "Output file path")->required(true);
        fvecs_to_bin->callback([&]() {
            auto rs = float_vecs_to_bin(datasets_ctx.input, datasets_ctx.output);
            if(!rs.ok()) {
                std::cerr << rs.message() << std::endl;
            }
        });
        /// uint8_vecs_to_bin
        auto *u8vecs_to_bin = app->add_subcommand("uint8_vecs_to_bin", "Convert uint8 vectors to binary format");
        u8vecs_to_bin->add_option("-i, --input", datasets_ctx.input, "Input file path")->required(true);
        u8vecs_to_bin->add_option("-o, --output", datasets_ctx.output, "Output file path")->required(true);
        u8vecs_to_bin->callback([&]() {
            auto rs = uint8_vecs_to_bin(datasets_ctx.input, datasets_ctx.output);
            if(!rs.ok()) {
                std::cerr << rs.message() << std::endl;
            }
        });
        /// float_vecs_to_uint8_vecs
        auto *fvecs_to_u8vecs = app->add_subcommand("float_vecs_to_uint8_vecs", "Convert float vectors to uint8 vectors");
        fvecs_to_u8vecs->add_option("-i, --input", datasets_ctx.input, "Input file path")->required(true);
        fvecs_to_u8vecs->add_option("-o, --output", datasets_ctx.output, "Output file path")->required(true);
        fvecs_to_u8vecs->callback([&]() {
            auto rs = float_vecs_to_uint8_vecs(datasets_ctx.input, datasets_ctx.output);
            if(!rs.ok()) {
                std::cerr << rs.message() << std::endl;
            }
        });
        /// float_bin_to_int8
        auto *fbin_to_i8 = app->add_subcommand("float_bin_to_int8", "Convert float binary to int8 binary");
        fbin_to_i8->add_option("-i, --input", datasets_ctx.input, "Input file path")->required(true);
        fbin_to_i8->add_option("-o, --output", datasets_ctx.output, "Output file path")->required(true);
        fbin_to_i8->add_option("-b, --bias", datasets_ctx.bias, "Bias value")->required(true);
        fbin_to_i8->add_option("-s, --scale", datasets_ctx.scale, "Scale value")->required(true);
        fbin_to_i8->callback([&]() {
            auto rs = float_bin_to_int8(datasets_ctx.input, datasets_ctx.output, datasets_ctx.bias, datasets_ctx.scale);
            if(!rs.ok()) {
                std::cerr << rs.message() << std::endl;
            }
        });

        /// ivecs_to_bin
        auto *iv_to_bin = app->add_subcommand("ivecs_to_bin", "Convert int vectors to binary format");
        iv_to_bin->add_option("-i, --input", datasets_ctx.input, "Input file path")->required(true);
        iv_to_bin->add_option("-o, --output", datasets_ctx.output, "Output file path")->required(true);
        iv_to_bin->callback([&]() {
            auto rs = ivecs_to_bin(datasets_ctx.input, datasets_ctx.output);
            if(!rs.ok()) {
                std::cerr << rs.message() << std::endl;
            }
        });

        /// int8_to_float
        auto *i8_to_f = app->add_subcommand("int8_to_float", "Convert int8 binary to float binary");
        i8_to_f->add_option("-i, --input", datasets_ctx.input, "Input file path")->required(true);
        i8_to_f->add_option("-o, --output", datasets_ctx.output, "Output file path")->required(true);
        i8_to_f->callback([&]() {
            auto rs = int8_to_float(datasets_ctx.input, datasets_ctx.output);
            if(!rs.ok()) {
                std::cerr << rs.message() << std::endl;
            }
        });

        /// int8_to_float_scalar
        auto *i8_to_f_scalar = app->add_subcommand("int8_to_float_scalar", "Convert int8 binary to float binary with scalar");
        i8_to_f_scalar->add_option("-i, --input", datasets_ctx.input, "Input file path")->required(true);
        i8_to_f_scalar->add_option("-o, --output", datasets_ctx.output, "Output file path")->required(true);
        i8_to_f_scalar->add_option("-b, --bias", datasets_ctx.bias, "Bias value")->required(true);
        i8_to_f_scalar->add_option("-s, --scale", datasets_ctx.scale, "Scale value")->required(true);
        i8_to_f_scalar->callback([&]() {
            auto rs = int8_to_float_scalar(datasets_ctx.input, datasets_ctx.output, datasets_ctx.bias, datasets_ctx.scale);
            if(!rs.ok()) {
                std::cerr << rs.message() << std::endl;
            }
        });

        /// uint8_to_float
        auto *u8_to_f = app->add_subcommand("uint8_to_float", "Convert uint8 binary to float binary");
        u8_to_f->add_option("-i, --input", datasets_ctx.input, "Input file path")->required(true);
        u8_to_f->add_option("-o, --output", datasets_ctx.output, "Output file path")->required(true);
        u8_to_f->callback([&]() {
            auto rs = uint8_to_float(datasets_ctx.input, datasets_ctx.output);
            if(!rs.ok()) {
                std::cerr << rs.message() << std::endl;
            }
        });
        app->require_subcommand(1);

        /// tsv_to_float_bin
        auto *tsv_to_fbin = app->add_subcommand("tsv_to_float_bin", "Convert tsv to float binary");
        tsv_to_fbin->add_option("-i, --input", datasets_ctx.input, "Input file path")->required(true);
        tsv_to_fbin->add_option("-o, --output", datasets_ctx.output, "Output file path")->required(true);
        tsv_to_fbin->add_option("-d, --ndims", datasets_ctx.ndims, "Number of dimensions")->required(true);
        tsv_to_fbin->add_option("-n, --nvec", datasets_ctx.nvec, "Number of vectors")->required(true);
        tsv_to_fbin->callback([&]() {
            auto rs = tsv_to_float_bin(datasets_ctx.input, datasets_ctx.output, datasets_ctx.ndims, datasets_ctx.nvec);
            if(!rs.ok()) {
                std::cerr << rs.message() << std::endl;
            }
        });
        /// tsv_to_uint8_bin
        auto *tsv_to_u8bin = app->add_subcommand("tsv_to_uint8_bin", "Convert tsv to uint8 binary");
        tsv_to_u8bin->add_option("-i, --input", datasets_ctx.input, "Input file path")->required(true);
        tsv_to_u8bin->add_option("-o, --output", datasets_ctx.output, "Output file path")->required(true);
        tsv_to_u8bin->add_option("-d, --ndims", datasets_ctx.ndims, "Number of dimensions")->required(true);
        tsv_to_u8bin->add_option("-n, --nvec", datasets_ctx.nvec, "Number of vectors")->required(true);
        tsv_to_u8bin->callback([&]() {
            auto rs = tsv_to_uint8_bin(datasets_ctx.input, datasets_ctx.output, datasets_ctx.ndims, datasets_ctx.nvec);
            if(!rs.ok()) {
                std::cerr << rs.message() << std::endl;
            }
        });
        /// tsv_to_int8_bin
        auto *tsv_to_i8bin = app->add_subcommand("tsv_to_int8_bin", "Convert tsv to int8 binary");
        tsv_to_i8bin->add_option("-i, --input", datasets_ctx.input, "Input file path")->required(true);
        tsv_to_i8bin->add_option("-o, --output", datasets_ctx.output, "Output file path")->required(true);
        tsv_to_i8bin->add_option("-d, --ndims", datasets_ctx.ndims, "Number of dimensions")->required(true);
        tsv_to_i8bin->add_option("-n, --nvec", datasets_ctx.nvec, "Number of vectors")->required(true);
        tsv_to_i8bin->callback([&]() {
            auto rs = tsv_to_int8_bin(datasets_ctx.input, datasets_ctx.output, datasets_ctx.ndims, datasets_ctx.nvec);
            if(!rs.ok()) {
                std::cerr << rs.message() << std::endl;
            }
        });
        /// int8_bin_to_tsv
        auto *i8bin_to_tsv = app->add_subcommand("int8_bin_to_tsv", "Convert int8 binary to tsv");
        i8bin_to_tsv->add_option("-i, --input", datasets_ctx.input, "Input file path")->required(true);
        i8bin_to_tsv->add_option("-o, --output", datasets_ctx.output, "Output file path")->required(true);
        i8bin_to_tsv->callback([&]() {
            auto rs = int8_bin_to_tsv(datasets_ctx.input, datasets_ctx.output);
            if(!rs.ok()) {
                std::cerr << rs.message() << std::endl;
            }
        });
        /// uint8_bin_to_tsv
        auto *u8bin_to_tsv = app->add_subcommand("uint8_bin_to_tsv", "Convert uint8 binary to tsv");
        u8bin_to_tsv->add_option("-i, --input", datasets_ctx.input, "Input file path")->required(true);
        u8bin_to_tsv->add_option("-o, --output", datasets_ctx.output, "Output file path")->required(true);
        u8bin_to_tsv->callback([&]() {
            auto rs = uint8_bin_to_tsv(datasets_ctx.input, datasets_ctx.output);
            if(!rs.ok()) {
                std::cerr << rs.message() << std::endl;
            }
        });
        /// float_bin_to_tsv
        auto *fbin_to_tsv = app->add_subcommand("float_bin_to_tsv", "Convert float binary to tsv");
        fbin_to_tsv->add_option("-i, --input", datasets_ctx.input, "Input file path")->required(true);
        fbin_to_tsv->add_option("-o, --output", datasets_ctx.output, "Output file path")->required(true);
        fbin_to_tsv->callback([&]() {
            auto rs = float_bin_to_tsv(datasets_ctx.input, datasets_ctx.output);
            if(!rs.ok()) {
                std::cerr << rs.message() << std::endl;
            }
        });

        /// gen_random_float
        auto *gen_rnd_f = app->add_subcommand("gen_random_float", "Generate random float vectors");
        gen_rnd_f->add_option("-i, --input", datasets_ctx.input, "Input file path")->required(true);
        gen_rnd_f->add_option("-o, --output", datasets_ctx.output, "Output file path")->required(true);
        gen_rnd_f->add_option("-s, --sampling_rate", datasets_ctx.sampling_rate, "Sampling rate")->required(true);
        gen_rnd_f->callback([&]() {
            gen_random_float(datasets_ctx.input, datasets_ctx.output, datasets_ctx.sampling_rate);
        });

        /// gen_random_uint8
        auto *gen_rnd_u8 = app->add_subcommand("gen_random_uint8", "Generate random uint8 vectors");
        gen_rnd_u8->add_option("-i, --input", datasets_ctx.input, "Input file path")->required(true);
        gen_rnd_u8->add_option("-o, --output", datasets_ctx.output, "Output file path")->required(true);
        gen_rnd_u8->add_option("-s, --sampling_rate", datasets_ctx.sampling_rate, "Sampling rate")->required(true);
        gen_rnd_u8->callback([&]() {
            gen_random_uint8(datasets_ctx.input, datasets_ctx.output, datasets_ctx.sampling_rate);
        });

        /// gen_random_int8
        auto *gen_rnd_i8 = app->add_subcommand("gen_random_int8", "Generate random int8 vectors");
        gen_rnd_i8->add_option("-i, --input", datasets_ctx.input, "Input file path")->required(true);
        gen_rnd_i8->add_option("-o, --output", datasets_ctx.output, "Output file path")->required(true);
        gen_rnd_i8->add_option("-s, --sampling_rate", datasets_ctx.sampling_rate, "Sampling rate")->required(true);
        gen_rnd_i8->callback([&]() {
            gen_random_int8(datasets_ctx.input, datasets_ctx.output, datasets_ctx.sampling_rate);
        });

        /// generate_pq_float
        auto *gen_pq_f = app->add_subcommand("generate_pq_float", "Generate PQ float vectors");
        gen_pq_f->add_option("-i, --input", datasets_ctx.input, "Input file path")->required(true);
        gen_pq_f->add_option("-o, --output", datasets_ctx.output, "Output file path")->required(true);
        gen_pq_f->add_option("-n, --num_pq_chunks", datasets_ctx.num_pq_chunks, "Number of PQ chunks")->required(true);
        gen_pq_f->add_option("-s, --sampling_rate", datasets_ctx.sampling_rate, "Sampling rate")->required(true);
        gen_pq_f->add_option("--opq", datasets_ctx.bias, "OPQ")->required(true);
        gen_pq_f->callback([&]() {
            generate_pq_float(datasets_ctx.input, datasets_ctx.output, datasets_ctx.num_pq_chunks, datasets_ctx.sampling_rate, datasets_ctx.bias);
        });

        /// generate_pq_int8
        auto *gen_pq_i8 = app->add_subcommand("generate_pq_int8", "Generate PQ int8 vectors");
        gen_pq_i8->add_option("-i, --input", datasets_ctx.input, "Input file path")->required(true);
        gen_pq_i8->add_option("-o, --output", datasets_ctx.output, "Output file path")->required(true);
        gen_pq_i8->add_option("-n, --num_pq_chunks", datasets_ctx.num_pq_chunks, "Number of PQ chunks")->required(true);
        gen_pq_i8->add_option("-s, --sampling_rate", datasets_ctx.sampling_rate, "Sampling rate")->required(true);
        gen_pq_i8->add_option("--opq", datasets_ctx.bias, "OPQ")->required(true);
        gen_pq_i8->callback([&]() {
            generate_pq_int8(datasets_ctx.input, datasets_ctx.output, datasets_ctx.num_pq_chunks, datasets_ctx.sampling_rate, datasets_ctx.bias);
        });

        /// generate_pq_uint8
        auto *gen_pq_u8 = app->add_subcommand("generate_pq_uint8", "Generate PQ uint8 vectors");
        gen_pq_u8->add_option("-i, --input", datasets_ctx.input, "Input file path")->required(true);
        gen_pq_u8->add_option("-o, --output", datasets_ctx.output, "Output file path")->required(true);
        gen_pq_u8->add_option("-n, --num_pq_chunks", datasets_ctx.num_pq_chunks, "Number of PQ chunks")->required(true);
        gen_pq_u8->add_option("-s, --sampling_rate", datasets_ctx.sampling_rate, "Sampling rate")->required(true);
        gen_pq_u8->add_option("--opq", datasets_ctx.opq, "OPQ")->required(true);
        gen_pq_u8->callback([&]() {
            generate_pq_uint8(datasets_ctx.input, datasets_ctx.output, datasets_ctx.num_pq_chunks, datasets_ctx.sampling_rate, datasets_ctx.opq);
        });

        /// uin32_to_uint8_bin
        auto *u32_to_u8 = app->add_subcommand("uin32_to_uint8_bin", "Convert uint32 to uint8 binary");
        u32_to_u8->add_option("-i, --input", datasets_ctx.input, "Input file path")->required(true);
        u32_to_u8->add_option("-o, --output", datasets_ctx.output, "Output file path")->required(true);
        u32_to_u8->callback([&]() {
            auto rs = uint32_to_uint8_bin(datasets_ctx.input, datasets_ctx.output);
            if(!rs.ok()) {
                std::cerr << rs.message() << std::endl;
            }
        });

        /// calculate_recall
        auto *calc_recall = app->add_subcommand("calculate_recall", "Calculate recall");
        calc_recall->add_option("-v, --vecs_file", datasets_ctx.input, "Vectors file path")->required(true);
        calc_recall->add_option("-b, --bin_file", datasets_ctx.output, "Binary file path")->required(true);
        calc_recall->add_option("-r, --recall", datasets_ctx.recall, "Recall value")->required(true);
        calc_recall->callback([&]() {
            auto rs = calculate_recall(datasets_ctx.input, datasets_ctx.output, datasets_ctx.recall);
            if(!rs.ok()) {
                std::cerr << rs.message() << std::endl;
            }
        });

    }


}  // namespace polaris
