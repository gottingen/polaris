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

#include <turbo/times/time.h>
#include <iostream>
#include <iomanip>

namespace polaris {
    class Timer {
    public:
        Timer() = default;

        void reset_delta() {
            delta = turbo::Duration::zero();
        }

        void start() {
            reset_delta();
            startTime = turbo::Time::time_now();
        }

        void reset_start_time() {
            startTime = turbo::Time::time_now();
        }

        void stop() {
            stopTime = turbo::Time::time_now();
            delta = stopTime - startTime;
        }

        [[nodiscard]] turbo::Duration elapsed() const {
            return turbo::Time::time_now() - startTime;
        }

        [[nodiscard]] std::string elapsed_seconds_for_step(const std::string &step) const {
            return std::string("Time for ") + step + std::string(": ") + std::to_string(elapsed().to_seconds<double>()) +
                   std::string(" seconds");
        }

        void add(Timer &t) {
            delta += t.delta;
        }

        friend std::ostream &operator<<(std::ostream &os, Timer &t) {
            auto time = t.delta.to_seconds<double>();
            if (time < 1.0) {
                time *= 1000.0;
                os << std::setprecision(6) << time << " (ms)";
                return os;
            }
            if (time < 60.0) {
                os << std::setprecision(6) << time << " (s)";
                return os;
            }
            time /= 60.0;
            if (time < 60.0) {
                os << std::setprecision(6) << time << " (m)";
                return os;
            }
            time /= 60.0;
            os << std::setprecision(6) << time << " (h)";
            return os;
        }

        turbo::Time startTime;
        turbo::Time stopTime;

        int64_t sec;
        int64_t nsec;
        turbo::Duration delta;    // second
    };
}  // namespace polaris