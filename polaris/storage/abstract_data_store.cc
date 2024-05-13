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

#include <vector>
#include <polaris/storage/abstract_data_store.h>

namespace polaris {

    template<typename data_t>
    AbstractDataStore<data_t>::AbstractDataStore(const location_t capacity, const size_t dim)
            : _capacity(capacity), _dim(dim) {
    }

    template<typename data_t>
    location_t AbstractDataStore<data_t>::capacity() const {
        return _capacity;
    }

    template<typename data_t>
    size_t AbstractDataStore<data_t>::get_dims() const {
        return _dim;
    }

    template<typename data_t>
    turbo::ResultStatus<location_t> AbstractDataStore<data_t>::resize(const location_t new_num_points) {
        if (new_num_points > _capacity) {
            return expand(new_num_points);
        } else if (new_num_points < _capacity) {
            return shrink(new_num_points);
        } else {
            return _capacity;
        }
    }

    template POLARIS_API
    class AbstractDataStore<float>;

    template POLARIS_API
    class AbstractDataStore<int8_t>;

    template POLARIS_API
    class AbstractDataStore<uint8_t>;
} // namespace polaris
