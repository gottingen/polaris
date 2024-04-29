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

#include <cstdint>
#include <cstddef>
#include <vector>
#include <any>
#include <turbo/container/flat_hash_set.h>

namespace AnyWrapper {

    /*
     * Base Struct to hold refrence to the data.
     * Note: No memory mamagement, caller need to keep object alive.
     */
    struct AnyReference {
        template<typename Ty>
        AnyReference(Ty &reference) : _data(&reference) {
        }

        template<typename Ty>
        Ty &get() {
            auto ptr = std::any_cast<Ty *>(_data);
            return *ptr;
        }

    private:
        std::any _data;
    };

    struct AnyRobinSet : public AnyReference {
        template<typename T>
        AnyRobinSet(const turbo::flat_hash_set<T> &robin_set) : AnyReference(robin_set) {
        }

        template<typename T>
        AnyRobinSet(turbo::flat_hash_set<T> &robin_set) : AnyReference(robin_set) {
        }
    };

    struct AnyVector : public AnyReference {
        template<typename T>
        AnyVector(const std::vector<T> &vector) : AnyReference(vector) {
        }

        template<typename T>
        AnyVector(std::vector<T> &vector) : AnyReference(vector) {
        }
    };
} // namespace AnyWrapper
