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

#include <shared_mutex>
#include <memory>

#include <turbo/container/flat_hash_map.h>
#include <turbo/container/flat_hash_set.h>
#include <turbo/container/flat_hash_map.h>
#include <polaris/storage/abstract_data_store.h>

#include <polaris/distance/distance.h>
#include <polaris/utility/natural_number_map.h>
#include <polaris/utility/natural_number_set.h>
#include <polaris/io/aligned_file_reader.h>

namespace polaris {
    template<typename data_t>
    class InMemDataStore : public AbstractDataStore<data_t> {
    public:
        InMemDataStore(const location_t capacity, const size_t dim, std::unique_ptr<Distance<data_t>> distance_fn);

        ~InMemDataStore() override;

        turbo::ResultStatus<location_t> load(const std::string &filename) override;

        turbo::ResultStatus<size_t> save(const std::string &filename, const location_t num_points) override;

        size_t get_aligned_dim() const override;

        // Populate internal data from unaligned data while doing alignment and any
        // normalization that is required.
        [[nodiscard]] turbo::Status populate_data(const data_t *vectors, const location_t num_pts) override;

        [[nodiscard]] turbo::Status populate_data(const std::string &filename, const size_t offset) override;

        void extract_data_to_bin(const std::string &filename, const location_t num_pts) override;

        void get_vector(const location_t i, data_t *target) const override;

        void set_vector(const location_t i, const data_t *const vector) override;

        void prefetch_vector(const location_t loc) override;

        void move_vectors(const location_t old_location_start, const location_t new_location_start,
                                  const location_t num_points) override;

        void
        copy_vectors(const location_t from_loc, const location_t to_loc, const location_t num_points) override;

        turbo::Status preprocess_query(const data_t *query, AbstractScratch<data_t> *query_scratch) const override;

        float get_distance(const ArrayView &preprocessed_query, const location_t loc) const override;

        float get_distance(const location_t loc1, const location_t loc2) const override;

        void get_distance(const ArrayView &preprocessed_query, const location_t *locations,
                                  const uint32_t location_count, float *distances,
                                  AbstractScratch<data_t> *scratch) const override;

        void get_distance(const ArrayView &preprocessed_query, const std::vector<location_t> &ids,
                                  std::vector<float> &distances, AbstractScratch<data_t> *scratch_space) const override;

        location_t calculate_medoid() const override;

        size_t get_alignment_factor() const override;

    protected:
        turbo::ResultStatus<location_t> expand(const location_t new_size) override;

        turbo::ResultStatus<location_t> shrink(const location_t new_size) override;

        turbo::ResultStatus<location_t> load_impl(const std::string &filename);

    private:
        data_t *_data = nullptr;

        size_t _aligned_dim;

        // It may seem weird to put distance metric along with the data store class,
        // but this gives us perf benefits as the datastore can do distance
        // computations during search and compute norms of vectors internally without
        // have to copy data back and forth.
        std::unique_ptr<Distance<data_t>> _distance_fn;

        // in case we need to save vector norms for optimization
        std::vector<float> _pre_computed_norms_l2;
    };

} // namespace polaris