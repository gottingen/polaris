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

#include <cstddef>

namespace polaris {

    struct consolidation_report {
        enum status_code {
            SUCCESS = 0,
            FAIL = 1,
            LOCK_FAIL = 2,
            INCONSISTENT_COUNT_ERROR = 3
        };
        status_code _status{status_code::SUCCESS};
        size_t _active_points{0};
        size_t _max_points{0};
        size_t _empty_slots{0};
        size_t _slots_released{0};
        size_t _delete_set_size{0};
        size_t _num_calls_to_process_delete{0};
        double _time{0.0};
        consolidation_report() = default;
        consolidation_report(status_code status, size_t active_points, size_t max_points, size_t empty_slots,
                             size_t slots_released, size_t delete_set_size, size_t num_calls_to_process_delete,
                             double time_secs)
                : _status(status), _active_points(active_points), _max_points(max_points), _empty_slots(empty_slots),
                  _slots_released(slots_released), _delete_set_size(delete_set_size),
                  _num_calls_to_process_delete(num_calls_to_process_delete), _time(time_secs) {
        }
    };


}  // namespace polaris
