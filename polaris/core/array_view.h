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

#pragma once

#include <cstddef>

namespace polaris {

    class ArrayView {
    public:
        ArrayView() = default;
        explicit ArrayView(void *data) : _data(data) {}

        [[nodiscard]] const void *data() const { return _data; }

        [[nodiscard]] float l2_norm() const { return _l2_norm; }

        [[nodiscard]] float l2_norm_sq() const { return _l2_norm_sq; }

        ArrayView &set_data(const void *data) {
            _data = data;
            return *this;
        }

        ArrayView &set_l2_norm(float l2_norm) {
            _l2_norm = l2_norm;
            return *this;
        }

        ArrayView &set_l2_norm_sq(float l2_norm_sq) {
            _l2_norm_sq = l2_norm_sq;
            return *this;
        }

        explicit operator bool() const { return _data != nullptr; }


    private:
        const void *_data{nullptr};
        float _l2_norm{0.0f};
        float _l2_norm_sq{0.0f};
    };
}  // namespace polaris

