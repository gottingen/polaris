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

#include <polaris/core/log.h>
#include <collie/log/sinks/rotating_file_sink.h>
#include <thread>
#include <mutex>

namespace polaris {

    std::unique_ptr<LoggerConfig> logger_config;
    std::shared_ptr<clog::logger> logger_raw = clog::default_logger();
    std::shared_ptr<clog::logger> g_logger = logger_raw;

    std::once_flag logger_init_flag;
    std::once_flag warn_logger_init_flag;

    void rotating_example() {
        // Create a file rotating logger with 5mb size max and 3 rotated files.
        auto rotating_logger =
                clog::rotating_logger_mt("some_logger_name", "logs/rotating.txt", 1048576 * 5, 3);
    }


    void init_logger(const LoggerConfig &config) {
        std::call_once(logger_init_flag, [&config]() {
            logger_config = std::make_unique<LoggerConfig>();
            *logger_config = config;
            g_logger = clog::rotating_logger_mt(config.log_name, config.log_path, config.max_size_mb * 1048576,
                                                config.max_files);
        });
    }
    clog::logger *get_logger() {
        std::call_once(warn_logger_init_flag, []() {
            if(g_logger == clog::default_logger()) {
                clog::warn("Logger is not initialized, using default logger");
            }
        });
        return g_logger.get();
    }

}  // namespace polaris
