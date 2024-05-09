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

#include <polaris/core/search_condition.h>

namespace polaris {

    bool BitmapSearchCondition::is_in_blacklist(vid_t label) const {
        return _blacklist.contains(label);
    }

    void BitmapSearchCondition::add_to_blacklist(vid_t label) {
        _blacklist.add(label);
    }

    void BitmapSearchCondition::add_to_blacklist(const vid_t *labels, size_t num_labels) {
        for (size_t i = 0; i < num_labels; i++) {
            _blacklist.add(labels[i]);
        }
    }

    bool BitmapSearchCondition::add_to_blacklist(const char* labels, size_t len, bool is_compressed) {
        return turbo::load_bitmap(labels, len, _blacklist, is_compressed);
    }

    void BitmapSearchCondition::add_to_whitelist(vid_t label) {
        _whitelist.add(label);
    }

    void BitmapSearchCondition::add_to_whitelist(const vid_t *labels, size_t num_labels) {
        for (size_t i = 0; i < num_labels; i++) {
            _whitelist.add(labels[i]);
        }
    }

    bool BitmapSearchCondition::add_to_whitelist(const char* labels, size_t len, bool is_compressed) {
        return turbo::load_bitmap(labels, len, _whitelist, is_compressed);
    }

    bool BitmapSearchCondition::is_whitelist(vid_t label) const {
        return _whitelist.contains(label);
    }

    bool BitmapSearchCondition::should_stop_search(distance_t lowerBound) const {
        return false;
    }

    void CompositeSearchCondition::add_condition(polaris::BaseSearchCondition *condition) {
        _conditions.push_back(condition);
    }

    void CompositeSearchCondition::add_condition(polaris::BaseSearchCondition *condition, size_t len) {
        for (size_t i = 0; i < len; i++) {
            _conditions.push_back(condition);
        }
    }

    bool CompositeSearchCondition::is_in_blacklist(vid_t label) const {
        for (auto &condition : _conditions) {
            if (condition->is_in_blacklist(label)) {
                return true;
            }
        }
        return false;
    }

    bool CompositeSearchCondition::is_whitelist(vid_t label) const {
        for (auto &condition : _conditions) {
            if (condition->is_whitelist(label)) {
                return true;
            }
        }
        return false;
    }

    bool CompositeSearchCondition::should_stop_search(distance_t lowerBound) const {
        for (auto &condition : _conditions) {
            if (condition->should_stop_search(lowerBound)) {
                return true;
            }
        }
        return false;
    }


}  // namespace polaris