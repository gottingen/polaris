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

#include <polaris/internal/memory_mapper.h>
#include <iostream>
#include <sstream>

namespace polaris {

    MemoryMapper::MemoryMapper(const std::string &filename) : MemoryMapper(filename.c_str()) {
    }

    MemoryMapper::MemoryMapper(const char *filename) {
        _fd = open(filename, O_RDONLY);
        if (_fd <= 0) {
            std::cerr << "Inner vertices file not found" << std::endl;
            return;
        }
        struct stat sb;
        if (fstat(_fd, &sb) != 0) {
            std::cerr << "Inner vertices file not dound. " << std::endl;
            return;
        }
        _fileSize = sb.st_size;
        std::cout << "File Size: " << _fileSize << std::endl;
        _buf = (char *) mmap(NULL, _fileSize, PROT_READ, MAP_PRIVATE, _fd, 0);
    }

    char *MemoryMapper::getBuf() {
        return _buf;
    }

    size_t MemoryMapper::getFileSize() {
        return _fileSize;
    }

    MemoryMapper::~MemoryMapper() {
        if (munmap(_buf, _fileSize) != 0)
            std::cerr << "ERROR unmapping. CHECK!" << std::endl;
        close(_fd);
    }
}  // namespace polaris