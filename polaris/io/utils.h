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

#include <fstream>
#include <collie/filesystem/fs.h>
#include <polaris/core/log.h>
#include <polaris/utility/polaris_exception.h>
#include <collie/utility/status.h>

namespace polaris {

    inline collie::Status open_file_to_write(std::ofstream &writer, const std::string &filename) {
        writer.exceptions(std::ofstream::failbit | std::ofstream::badbit);
        std::error_code ec;
        if (!collie::filesystem::exists(filename, ec))
            writer.open(filename, std::ios::binary | std::ios::out);
        else
            writer.open(filename, std::ios::binary | std::ios::in | std::ios::out);

        if (writer.fail()) {
            char buff[1024];
            auto ret = std::string(strerror_r(errno, buff, 1024));
            auto message = std::string("Failed to open file") + filename + " for write because " + buff + ", ret=" + ret;
            POLARIS_LOG(ERROR)<< message;
            return collie::Status::from_errno(errno, message);
        }
        return collie::Status::ok_status();
    }

    inline void open_to_write(std::ofstream &writer, const std::string &filename, bool truncate = false) {
        writer.exceptions(std::ofstream::failbit | std::ofstream::badbit);
        std::error_code ec;
        if (!collie::filesystem::exists(filename, ec)) {
            writer.open(filename, std::ios::binary | std::ios::out);
        } else {
            if (truncate) {
                writer.open(filename, std::ios::binary | std::ios::out | std::ios::trunc);
            } else {
                writer.open(filename, std::ios::binary |  std::ios::out | std::ios::app);
            }
        }

        if (writer.fail()) {
            char buff[1024];
            auto ret = std::string(strerror_r(errno, buff, 1024));
            auto message = std::string("Failed to open file") + filename + " for write because " + buff + ", ret=" + ret;
            //POLARIS_LOG(FATAL)<< message;
            std::cerr << message << std::endl;
            throw polaris::PolarisException(message, -1);
        }
    }

    inline void open_to_read(std::ifstream &reader, const std::string &filename) {
        reader.exceptions(std::ifstream::failbit | std::ifstream::badbit);
        reader.open(filename, std::ios::binary | std::ios::in);
        if (reader.fail()) {
            char buff[1024];
            auto ret = std::string(strerror_r(errno, buff, 1024));
            auto message = std::string("Failed to open file") + filename + " for read because " + buff + ", ret=" + ret;
            //POLARIS_LOG(FATAL)<< message;
            std::cerr << message << std::endl;
            throw polaris::PolarisException(message, -1);
        }
    }

    template<typename T>
    static void write_binary_pod(std::ostream &out, const T &podRef) {
        out.write((char *) &podRef, sizeof(T));
    }
    template<typename T>
    static void read_binary_pod(std::istream &in, T &podRef) {
        in.read((char *) &podRef, sizeof(T));
    }

}  // namespace polaris
