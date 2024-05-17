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

#pragma once

#include <polaris/core/parameter/basic.h>

namespace polaris {

    struct HnswParameters : public BasicParameters {
        uint32_t m{16};
        uint32_t ef{50};
        uint32_t ef_construction{200};
        uint32_t random_seed{100};

        HnswParameters() = default;

        ~HnswParameters() = default;

        HnswParameters(const HnswParameters &) = default;

        HnswParameters &operator=(const HnswParameters &) = default;

        void set_default() {
            BasicParameters::set_default();
            m = 16;
            ef = 50;
            ef_construction = 200;
            random_seed = 100;
        }

        void clear() {
            BasicParameters::clear();
            m = 16;
            ef = 50;
            ef_construction = 200;
            random_seed = 100;
        }

        [[nodiscard]] turbo::Status export_property(polaris::PropertySet &p) const;
        [[nodiscard]] turbo::Status import_property(polaris::PropertySet &p);
    };
}  // namespace polaris
