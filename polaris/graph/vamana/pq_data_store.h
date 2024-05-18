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
#pragma once

#include <memory>
#include <polaris/distance/distance.h>
#include <polaris/graph/vamana/quantized_distance.h>
#include <polaris/graph/vamana/pq.h>
#include <polaris/storage/abstract_data_store.h>

namespace polaris {
    // REFACTOR TODO: By default, the PQDataStore is an in-memory datastore because both Vamana and
    // DiskANN treat it the same way. But with DiskPQ, that may need to change.
    template<typename data_t>
    class PQDataStore : public AbstractDataStore<data_t> {

    public:
        PQDataStore(size_t dim, location_t num_points, size_t num_pq_chunks,
                    std::unique_ptr<Distance<data_t>> distance_fn,
                    std::unique_ptr<QuantizedDistance<data_t>> pq_distance_fn);

        PQDataStore(const PQDataStore &) = delete;

        PQDataStore &operator=(const PQDataStore &) = delete;

        ~PQDataStore();

        // Load quantized vectors from a set of files. Here filename is treated
        // as a prefix and the files are assumed to be named with DiskANN
        // conventions.
        turbo::ResultStatus<location_t> load(const std::string &file_prefix) override;

        // Save quantized vectors to a set of files whose names start with
        // file_prefix.
        //  Currently, the plan is to save the quantized vectors to the quantized
        //  vectors file.
        turbo::ResultStatus<size_t> save(const std::string &file_prefix, const location_t num_points) override;

        // Since base class function is pure virtual, we need to declare it here, even though alignent concept is not needed
        // for Quantized data stores.
        size_t get_aligned_dim() const override;

        // Populate quantized data from unaligned data using PQ functionality
        turbo::Status populate_data(const data_t *vectors, const location_t num_pts) override;

        turbo::Status populate_data(const std::string &filename, const size_t offset) override;

        void extract_data_to_bin(const std::string &filename, const location_t num_pts) override;

        void get_vector(const location_t i, data_t *target) const override;

        void set_vector(const location_t i, const data_t *const vector) override;

        void prefetch_vector(const location_t loc) override;

        void move_vectors(const location_t old_location_start, const location_t new_location_start,
                                  const location_t num_points) override;

        void
        copy_vectors(const location_t from_loc, const location_t to_loc, const location_t num_points) override;

        turbo::Status preprocess_query(const data_t *query, AbstractScratch<data_t> *scratch) const override;

        float get_distance(const ArrayView&query, const location_t loc) const override;

        float get_distance(const location_t loc1, const location_t loc2) const override;

        // NOTE: Caller must invoke "PQDistance->preprocess_query" ONCE before calling
        // this function.
        void get_distance(const ArrayView&preprocessed_query, const location_t *locations,
                                  const uint32_t location_count, float *distances,
                                  AbstractScratch<data_t> *scratch_space) const override;

        // NOTE: Caller must invoke "PQDistance->preprocess_query" ONCE before calling
        // this function.
        void get_distance(const ArrayView&preprocessed_query, const std::vector<location_t> &ids,
                                  std::vector<float> &distances, AbstractScratch<data_t> *scratch_space) const override;

        location_t calculate_medoid() const override;

        size_t get_alignment_factor() const override;

    protected:
        turbo::ResultStatus<location_t> expand(const location_t new_size) override;

        turbo::ResultStatus<location_t> shrink(const location_t new_size) override;

        turbo::ResultStatus<location_t> load_impl(const std::string &filename);

    private:
        uint8_t *_quantized_data = nullptr;
        size_t _num_chunks = 0;

        // REFACTOR TODO: Doing this temporarily before refactoring OPQ into
        // its own class. Remove later.
        bool _use_opq = false;

        MetricType _distance_metric;
        std::unique_ptr<Distance<data_t>> _distance_fn = nullptr;
        std::unique_ptr<QuantizedDistance<data_t>> _pq_distance_fn = nullptr;
    };
} // namespace polaris
