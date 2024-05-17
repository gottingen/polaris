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

#include <polaris/core/common.h>
#include <turbo/status/status.h>

namespace polaris {

    struct BasicParameters {
        BasicParameters() = default;

        BasicParameters(MetricType metric, ObjectType object_type, size_t dimension, size_t max_points)
                : metric(metric), object_type(object_type), dimension(dimension), max_points(max_points) {
        }

        BasicParameters(const BasicParameters &) = default;

        BasicParameters &operator=(const BasicParameters &) = default;

        MetricType metric{MetricType::METRIC_NONE};
        ObjectType object_type{ObjectType::ObjectTypeNone};
        size_t dimension{0};
        size_t max_points{0};
        size_t load_threads{1};
        size_t work_threads{1};
        DatabaseType databaseType{DatabaseType::DatabaseTypeNone};
        ObjectAlignment objectAlignment{ObjectAlignment::ObjectAlignmentNone};

        void set_default() {
            metric = MetricType::METRIC_NONE;
            object_type = ObjectType::ObjectTypeNone;
            dimension = 0;
            max_points = 0;
            load_threads = 1;
            work_threads = 1;
            databaseType = DatabaseType::DatabaseTypeNone;
            objectAlignment = ObjectAlignment::ObjectAlignmentNone;
        }

        void clear() {
            metric = MetricType::METRIC_NONE;
            object_type = ObjectType::ObjectTypeNone;
            dimension = 0;
            max_points = 0;
            load_threads = 1;
            work_threads = 1;
            databaseType = DatabaseType::DatabaseTypeNone;
            objectAlignment = ObjectAlignment::ObjectAlignmentNone;
        }

        turbo::Status validate() const {
            if (metric == MetricType::METRIC_NONE) {
                return turbo::make_status(turbo::kInvalidArgument, "metric is not set");
            }
            if (object_type == ObjectType::ObjectTypeNone) {
                return turbo::make_status(turbo::kInvalidArgument, "object_type is not set");
            }
            if (dimension == 0) {
                return turbo::make_status(turbo::kInvalidArgument, "dimension is not set");
            }
            return turbo::ok_status();
        }

    };
}  // namespace polaris
