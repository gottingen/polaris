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

#include <polaris/core/array_view.h>

#pragma once
namespace polaris {

    template<typename data_t>
    class PQScratch;

    // By somewhat more than a coincidence, it seems that both InMemQueryScratch
    // and SSDQueryScratch have the aligned query and PQScratch objects. So we
    // can put them in a neat hierarchy and keep PQScratch as a standalone class.
    template<typename data_t>
    class AbstractScratch {
    public:
        AbstractScratch() = default;

        // This class does not take any responsibilty for memory management of
        // its members. It is the responsibility of the derived classes to do so.
        virtual ~AbstractScratch() = default;

        // Scratch objects should not be copied
        AbstractScratch(const AbstractScratch &) = delete;

        AbstractScratch &operator=(const AbstractScratch &) = delete;

        data_t *aligned_query_T() {
            return _aligned_query_T;
        }

        PQScratch<data_t> *pq_scratch() {
            return _pq_scratch;
        }
        ArrayView  query_view;
    protected:
        data_t *_aligned_query_T = nullptr;
        PQScratch<data_t> *_pq_scratch = nullptr;
    };
} // namespace polaris
