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

#include <polaris/core/common.h>
#include <turbo/bitmap/bitmap.h>

namespace polaris {

    class BaseSearchCondition {
    public:
        virtual bool is_in_blacklist(vid_t label) const= 0;

        virtual bool is_whitelist(vid_t label) const = 0;

        virtual bool should_stop_search(distance_t lowerBound) const = 0;

        virtual ~BaseSearchCondition() = default;
    };

    class NullSearchCondition : public BaseSearchCondition {
    public:
        bool is_in_blacklist(vid_t label) const override {
            return false;
        }

        bool is_whitelist(vid_t label) const override {
            return false;
        }

        bool should_stop_search(distance_t lowerBound) const override {
            return false;
        }

        static BaseSearchCondition* instance() {
            static NullSearchCondition instance;
            return &instance;
        }
    };


    class BitmapSearchCondition : public BaseSearchCondition {
    public:
        BitmapSearchCondition() = default;

        bool is_in_blacklist(vid_t label) const override;

        void add_to_blacklist(vid_t label);

        void add_to_blacklist(const vid_t *labels, size_t num_labels);

        bool add_to_blacklist(const char* labels, size_t len, bool is_compressed = true);

        bool is_whitelist(vid_t label) const override;

        void add_to_whitelist(vid_t label);

        void add_to_whitelist(const vid_t *labels, size_t num_labels);
        bool add_to_whitelist(const char* labels, size_t len, bool is_compressed = true);

        bool should_stop_search(distance_t lowerBound) const override;
    private:
        turbo::Roaring _blacklist;
        turbo::Roaring _whitelist;
    };

    class CompositeSearchCondition : public BaseSearchCondition {
    public:
        CompositeSearchCondition() = default;

        void add_condition(BaseSearchCondition* condition);

        void add_condition(BaseSearchCondition* condition, size_t len);

        bool is_in_blacklist(vid_t label) const override;

        bool is_whitelist(vid_t label) const override;

        bool should_stop_search(distance_t lowerBound) const override;
    private:
        std::vector<BaseSearchCondition*> _conditions;
    };

}  // namespace polaris
