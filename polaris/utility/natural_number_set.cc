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

#include <boost/dynamic_bitset.hpp>

#include <polaris/utility/polaris_exception.h>
#include <polaris/utility/natural_number_set.h>

namespace polaris {
    template<typename T>
    natural_number_set<T>::natural_number_set() : _values_bitset(std::make_unique<turbo::Roaring>()) {
    }

    template<typename T>
    bool natural_number_set<T>::is_empty() const {
        return _values_vector.empty();
    }

    template<typename T>
    void natural_number_set<T>::reserve(size_t count) {
        _values_vector.reserve(count);
    }

    template<typename T>
    void natural_number_set<T>::insert(T id) {
        _values_vector.emplace_back(id);
        _values_bitset->add(id);
    }

    template<typename T>
    T natural_number_set<T>::pop_any() {
        if (_values_vector.empty()) {
            throw polaris::PolarisException("No values available", -1, __PRETTY_FUNCTION__, __FILE__, __LINE__);
        }

        const T id = _values_vector.back();
        _values_vector.pop_back();

        _values_bitset->remove(id);

        return id;
    }

    template<typename T>
    void natural_number_set<T>::clear() {
        _values_vector.clear();
        _values_bitset = std::make_unique<turbo::Roaring>();
    }

    template<typename T>
    size_t natural_number_set<T>::size() const {
        return _values_vector.size();
    }

    template<typename T>
    bool natural_number_set<T>::is_in_set(T id) const {
        return _values_bitset->contains(id);
    }

    // Instantiate used templates.
    template
    class natural_number_set<unsigned>;
} // namespace polaris
