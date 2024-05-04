//
// Copyright (C) 2015 Yahoo Japan Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//

#pragma once

#include <polaris/utility/mmap_manager.h>

#include <unistd.h>

namespace MemoryManager {
    const uint64_t MMAP_MANAGER_VERSION = 5;

    const bool MMAP_DEFAULT_ALLOW_EXPAND = false;
    const uint64_t MMAP_CNTL_FILE_RANGE = 16;
    const size_t MMAP_CNTL_FILE_SIZE = MMAP_CNTL_FILE_RANGE * sysconf(_SC_PAGESIZE);
    const uint64_t MMAP_MAX_FILE_NAME_LENGTH = 1024;
    const std::string MMAP_CNTL_FILE_SUFFIX = "c";

    const size_t MMAP_LOWER_SIZE = 1;
    const size_t MMAP_MEMORY_ALIGN = 8;
    const size_t MMAP_MEMORY_ALIGN_EXP = 3;

#ifndef MMANAGER_TEST_MODE
    const uint64_t MMAP_MAX_UNIT_NUM = 1024;
#else
    const uint64_t MMAP_MAX_UNIT_NUM = 8;
#endif

    const uint64_t MMAP_FREE_QUEUE_SIZE = 1024;

    const uint64_t MMAP_FREE_LIST_NUM = 64;

    typedef struct _boot_st {
        uint32_t version{0};
        uint64_t reserve{0};
        size_t size{0};
    } boot_st;

    typedef struct _head_st {
        off_t break_p{0};
        uint64_t chunk_num{0};
        uint64_t reserve{0};
    } head_st;


    typedef struct _free_list_st {
        off_t free_p{0};
        off_t free_last_p{0};
    } free_list_st;


    typedef struct _free_st {
        free_list_st large_list;
        free_list_st free_lists[MMAP_FREE_LIST_NUM];
    } free_st;


    typedef struct _free_queue_st {
        off_t data{0};
        size_t capacity{0};
        uint64_t tail{0};
    } free_queue_st;


    typedef struct _control_st {
        bool use_expand{false};
        uint16_t unit_num{0};
        uint16_t active_unit{0};
        uint64_t reserve{0};
        size_t base_size{0};
        off_t entry_p{0};
        option_reuse_t reuse_type;
        free_st free_data{0};
        free_queue_st free_queue;
        head_st data_headers[MMAP_MAX_UNIT_NUM];
    } control_st;

    typedef struct _chunk_head_st {
        bool delete_flg{false};
        uint16_t unit_id{0};
        off_t free_next{0};
        size_t size{0};
    } chunk_head_st;
}
