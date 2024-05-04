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

#include <collie/log/logging.h>

namespace polaris {

    struct LoggerConfig {
        std::string log_name;
        std::string log_path;
        size_t max_size_mb;
        size_t max_files;
    };

    void init_logger(const LoggerConfig &config);

    clog::logger *get_logger();
}  // namespace polaris

#define POLARIS_LOG(SEVERITY) LOG_LOGGER(SEVERITY, get_logger())
#define POLARIS_LOG_IF(SEVERITY, condition) LOG_LOGGER_IF(SEVERITY, (condition), get_logger())
#define POLARIS_LOG_EVERY_N(SEVERITY, N) LOG_LOGGER_EVERY_N(SEVERITY, N, get_logger())
#define POLARIS_LOG_IF_EVERY_N(SEVERITY, N, condition) LOG_LOGGER_IF_EVERY_N(SEVERITY, (N), (condition), get_logger())
#define POLARIS_LOG_FIRST_N(SEVERITY, N) LOG_LOGGER_FIRST_N(SEVERITY, (N), get_logger())
#define POLARIS_LOG_IF_FIRST_N(SEVERITY, N, condition) LOG_LOGGER_IF_FIRST_N(SEVERITY, (N), (condition), get_logger())
#define POLARIS_LOG_EVERY_T(SEVERITY, seconds) LOG_LOGGER_EVERY_T(SEVERITY, (seconds), get_logger())
#define POLARIS_LOG_IF_EVERY_T(SEVERITY, seconds, condition) LOG_LOGGER_IF_EVERY_T(SEVERITY, (seconds), (condition), get_logger())
#define POLARIS_LOG_ONCE(SEVERITY) LOG_LOGGER_ONCE(SEVERITY, get_logger())
#define POLARIS_LOG_IF_ONCE(SEVERITY, condition) LOG_LOGGER_IF_ONCE(SEVERITY, (condition), get_logger())

#define POLARIS_CHECK(condition) CHECK_LOGGER(condition, get_logger())
#define POLARIS_CHECK_NOTNULL(val) CHECK_LOGGER(collie::ptr(val) != nullptr,get_logger())
#define POLARIS_CHECK_EQ(val1, val2) CHECK_OP_LOGGER(_EQ, ==, val1, val2, get_logger())
#define POLARIS_CHECK_NE(val1, val2) CHECK_OP_LOGGER(_NE, !=, val1, val2, get_logger())
#define POLARIS_CHECK_LE(val1, val2) CHECK_OP_LOGGER(_LE, <=, val1, val2, get_logger())
#define POLARIS_CHECK_LT(val1, val2) CHECK_OP_LOGGER(_LT, <, val1, val2, get_logger())
#define POLARIS_CHECK_GE(val1, val2) CHECK_OP_LOGGER(_GE, >=, val1, val2, get_logger())
#define POLARIS_CHECK_GT(val1, val2) CHECK_OP_LOGGER(_GT, >, val1, val2, get_logger())
#define POLARIS_CHECK_DOUBLE_EQ(val1, val2) CHECK_DOUBLE_EQ_LOGGER(val1, val2, get_logger())
#define POLARIS_CHECK_NEAR(val1, val2, margin) CHECK_NEAR_LOGGER(val1, val2, margin, get_logger())
#define POLARIS_CHECK_INDEX(I, A) CHECK_LOGGER(I < (sizeof(A) / sizeof(A[0])), get_logger())
#define POLARIS_CHECK_BOUND(B, A) CHECK_LOGGER(B <= (sizeof(A) / sizeof(A[0])), get_logger())

#define POLARIS_PLOG(SEVERITY) PLOG_LOGGER(SEVERITY, get_logger())
#define POLARIS_PLOG_IF(SEVERITY, condition) PLOG_LOGGER_IF(SEVERITY, (condition), get_logger())
#define POLARIS_PLOG_EVERY_N(SEVERITY, N) PLOG_LOGGER_EVERY_N(SEVERITY, N, get_logger())
#define POLARIS_PLOG_IF_EVERY_N(SEVERITY, N, condition) PLOG_LOGGER_IF_EVERY_N(SEVERITY, (N), (condition), get_logger())
#define POLARIS_PLOG_FIRST_N(SEVERITY, N) PLOG_LOGGER_FIRST_N(SEVERITY, (N), true, get_logger())
#define POLARIS_PLOG_IF_FIRST_N(SEVERITY, N, condition) PLOG_LOGGER_IF_FIRST_N(SEVERITY, (N), (condition), get_logger())
#define POLARIS_PLOG_EVERY_T(SEVERITY, seconds) PLOG_LOGGER_EVERY_T(SEVERITY, (seconds), get_logger())
#define POLARIS_PLOG_IF_EVERY_T(SEVERITY, seconds, condition) PLOG_LOGGER_IF_EVERY_T(SEVERITY, (seconds), (condition), get_logger())
#define POLARIS_PLOG_ONCE(SEVERITY) PLOG_LOGGER_ONCE(SEVERITY, get_logger())
#define POLARIS_PLOG_IF_ONCE(SEVERITY, condition) PLOG_LOGGER_IF_ONCE(SEVERITY, (condition), get_logger())

#define POLARIS_PCHECK(condition) PCHECK_LOGGER(condition, get_logger())
#define POLARIS_PCHECK_PTREQ(val1, val2) PCHECK_PTREQ_LOGGER(val1, val2, get_logger())
#define POLARIS_PCHECK_NOTNUL(val) PCHECK_NOTNULL_LOGGER(collie::ptr(val) != nullptr, log::default_logger_raw())
#define POLARIS_PCHECK_EQ(val1, val2) PCHECK_EQ_LOGGER(val1, val2, get_logger())
#define POLARIS_PCHECK_NE(val1, val2) PCHECK_NE_LOGGER(val1, val2, get_logger())
#define POLARIS_PCHECK_LE(val1, val2) PCHECK_LE_LOGGER(val1, val2, get_logger())
#define POLARIS_PCHECK_LT(val1, val2) PCHECK_LT_LOGGER(val1, val2, get_logger())
#define POLARIS_PCHECK_GE(val1, val2) PCHECK_GE_LOGGER(val1, val2, get_logger())
#define POLARIS_PCHECK_GT(val1, val2) PCHECK_GT_LOGGER(val1, val2, get_logger())

#define POLARIS_VLOG(verboselevel) VLOG_LOGGER(verboselevel, get_logger())
#define POLARIS_VLOG_IF(verboselevel, condition) VLOG_IF_LOGGER(verboselevel, (condition), get_logger())
#define POLARIS_VLOG_EVERY_N(verboselevel, N) VLOG_EVERY_N_LOGGER(verboselevel, N, get_logger())
#define POLARIS_VLOG_IF_EVERY_N(verboselevel, N, condition) VLOG_IF_EVERY_N_LOGGER(verboselevel, (N), (condition), get_logger())
#define POLARIS_VLOG_FIRST_N(verboselevel, N) VLOG_FIRST_N_LOGGER(verboselevel, (N), get_logger())
#define POLARIS_VLOG_IF_FIRST_N(verboselevel, N, condition) VLOG_IF_FIRST_N_LOGGER(verboselevel, (N), (condition), get_logger())
#define POLARIS_VLOG_EVERY_T(verboselevel, seconds) VLOG_EVERY_T_LOGGER(verboselevel, (seconds), get_logger())
#define POLARIS_VLOG_IF_EVERY_T(verboselevel, seconds, condition) VLOG_IF_EVERY_T_LOGGER(verboselevel, (seconds), (condition), get_logger())
#define POLARIS_VLOG_ONCE(verboselevel) VLOG_ONCE_LOGGER(verboselevel, get_logger())
#define POLARIS_VLOG_IF_ONCE(verboselevel, condition) VLOG_IF_ONCE_LOGGER(verboselevel, (condition), get_logger())


#if CLOG_DCHECK_IS_ON()

#define POLARIS_DLOG(SEVERITY)                                LOG_LOGGER(SEVERITY, get_logger())
#define POLARIS_DLOG_IF(SEVERITY, condition)                  LOG_LOGGER_IF(SEVERITY, (condition), get_logger())
#define POLARIS_DLOG_EVERY_N(SEVERITY, N)                     LOG_LOGGER_EVERY_N(SEVERITY, N, get_logger())
#define POLARIS_DLOG_IF_EVERY_N(SEVERITY, N, condition)       LOG_LOGGER_IF_EVERY_N(SEVERITY, (N), (condition), get_logger())
#define POLARIS_DLOG_FIRST_N(SEVERITY, N)                     LOG_LOGGER_FIRST_N(SEVERITY, (N), true, get_logger())
#define POLARIS_DLOG_IF_FIRST_N(SEVERITY, N, condition)       LOG_LOGGER_IF_FIRST_N(SEVERITY, (N), (condition), get_logger())
#define POLARIS_DLOG_EVERY_T(SEVERITY, seconds)               LOG_LOGGER_EVERY_T(SEVERITY, (seconds), get_logger())
#define POLARIS_DLOG_IF_EVERY_T(SEVERITY, seconds, condition) LOG_LOGGER_IF_EVERY_T(SEVERITY, (seconds), (condition), get_logger())
#define POLARIS_DLOG_ONCE(SEVERITY)                           LOG_LOGGER_ONCE(SEVERITY, get_logger())
#define POLARIS_DLOG_IF_ONCE(SEVERITY, condition)             LOG_LOGGER_IF_ONCE(SEVERITY, (condition), get_logger())

#define POLARIS_DCHECK(condition) DCHECK_LOGGER(condition, get_logger())
#define POLARIS_DCHECK_NOTNULL(val) DCHECK_NOTNULL_LOGGER(val, get_logger())
#define POLARIS_DCHECK_PTREQ(val1, val2) DCHECK_PTREQ_LOGGER(val1, val2, get_logger())
#define POLARIS_DCHECK_EQ(val1, val2) CHECK_EQ_LOGGER(val1, val2, get_logger())
#define POLARIS_DCHECK_NE(val1, val2) CHECK_NE_LOGGER(val1, val2, get_logger())
#define POLARIS_DCHECK_LE(val1, val2) CHECK_LE_LOGGER(val1, val2, get_logger())
#define POLARIS_DCHECK_LT(val1, val2) CHECK_LT_LOGGER(val1, val2, get_logger())
#define POLARIS_DCHECK_GE(val1, val2) CHECK_GE_LOGGER(val1, val2, get_logger())
#define POLARIS_DCHECK_GT(val1, val2) CHECK_GT_LOGGER(val1, val2, get_logger())

#define POLARIS_DPLOG(SEVERITY)                                  PLOG_LOGGER(SEVERITY, get_logger())
#define POLARIS_DPLOG_IF(SEVERITY, condition)                    PLOG_LOGGER_IF(SEVERITY, (condition), get_logger())
#define POLARIS_DPLOG_EVERY_N(SEVERITY, N)                       PLOG_LOGGER_EVERY_N(SEVERITY, N, get_logger())
#define POLARIS_DPLOG_IF_EVERY_N(SEVERITY, N, condition)         PLOG_LOGGER_IF_EVERY_N(SEVERITY, (N), (condition), get_logger())
#define POLARIS_DPLOG_FIRST_N(SEVERITY, N)                       PLOG_LOGGER_FIRST_N(SEVERITY, (N), true, get_logger())
#define POLARIS_DPLOG_IF_FIRST_N(SEVERITY, N, condition)         PLOG_LOGGER_IF_FIRST_N(SEVERITY, (N), (condition), get_logger())
#define POLARIS_DPLOG_EVERY_T(SEVERITY, seconds)                 PLOG_LOGGER_EVERY_T(SEVERITY, (seconds), get_logger())
#define POLARIS_DPLOG_IF_EVERY_T(SEVERITY, seconds, condition)   PLOG_LOGGER_IF_EVERY_T(SEVERITY, (seconds), (condition), get_logger())
#define POLARIS_DPLOG_ONCE(SEVERITY)                             PLOG_LOGGER_ONCE(SEVERITY, get_logger())
#define POLARIS_DPLOG_IF_ONCE(SEVERITY, condition)               PLOG_LOGGER_IF_ONCE(SEVERITY, (condition), get_logger())

#define POLARIS_DPCHECK(condition) DPCHECK_LOGGER(condition, get_logger())
#define POLARIS_DPCHECK_NOTNULL(val) DPCHECK_NOTNULL_LOGGER(val, get_logger())
#define POLARIS_DPCHECK_PTREQ(val1, val2) DPCHECK_PTREQ_LOGGER(val1, val2, get_logger())
#define POLARIS_DPCHECK_EQ(val1, val2) PCHECK_EQ_LOGGER(val1, val2, get_logger())
#define POLARIS_DPCHECK_NE(val1, val2) PCHECK_NE_LOGGER(val1, val2, get_logger())
#define POLARIS_DPCHECK_LE(val1, val2) PCHECK_LE_LOGGER(val1, val2, get_logger())
#define POLARIS_DPCHECK_LT(val1, val2) PCHECK_LT_LOGGER(val1, val2, get_logger())
#define POLARIS_DPCHECK_GE(val1, val2) PCHECK_GE_LOGGER(val1, val2, get_logger())
#define POLARIS_DPCHECK_GT(val1, val2) PCHECK_GT_LOGGER(val1, val2, get_logger())

#define POLARIS_DVLOG(verboselevel) DVLOG_LOGGER(verboselevel, get_logger())
#define POLARIS_DVLOG_IF(verboselevel, condition) DVLOG_IF_LOGGER(verboselevel, (condition), get_logger())
#define POLARIS_DVLOG_EVERY_N(verboselevel, N) DVLOG_EVERY_N_LOGGER(verboselevel, N, get_logger())
#define POLARIS_DVLOG_IF_EVERY_N(verboselevel, N, condition) DVLOG_IF_EVERY_N_LOGGER(verboselevel, (N), (condition), get_logger())
#define POLARIS_DVLOG_FIRST_N(verboselevel, N) DVLOG_FIRST_N_LOGGER(verboselevel, (N), get_logger())
#define POLARIS_DVLOG_IF_FIRST_N(verboselevel, N, condition) DVLOG_IF_FIRST_N_LOGGER(verboselevel, (N), (condition), get_logger())
#define POLARIS_DVLOG_EVERY_T(verboselevel, seconds) DVLOG_EVERY_T_LOGGER(verboselevel, (seconds), get_logger())
#define POLARIS_DVLOG_IF_EVERY_T(verboselevel, seconds, condition) DVLOG_IF_EVERY_T_LOGGER(verboselevel, (seconds), (condition), get_logger())
#define POLARIS_DVLOG_ONCE(verboselevel) DVLOG_ONCE_LOGGER(verboselevel, get_logger())
#define POLARIS_DVLOG_IF_ONCE(verboselevel, condition) DVLOG_IF_ONCE_LOGGER(verboselevel, (condition), get_logger())

#else // NDEBUG

#define POLARIS_DLOG(SEVERITY)                                DLOG_LOGGER(SEVERITY, get_logger())
#define POLARIS_DLOG_IF(SEVERITY, condition)                  DLOG_LOGGER_IF(SEVERITY, (condition), get_logger())
#define POLARIS_DLOG_EVERY_N(SEVERITY, N)                     DLOG_LOGGER_EVERY_N(SEVERITY, N, get_logger())
#define POLARIS_DLOG_IF_EVERY_N(SEVERITY, N, condition)       DLOG_LOGGER_IF_EVERY_N(SEVERITY, (N), (condition), get_logger())
#define POLARIS_DLOG_FIRST_N(SEVERITY, N)                     DLOG_LOGGER_FIRST_N(SEVERITY, (N), true, get_logger())
#define POLARIS_DLOG_IF_FIRST_N(SEVERITY, N, condition)       DLOG_LOGGER_IF_FIRST_N(SEVERITY, (N), (condition), get_logger())
#define POLARIS_DLOG_EVERY_T(SEVERITY, seconds)               DLOG_LOGGER_EVERY_T(SEVERITY, (seconds), get_logger())
#define POLARIS_DLOG_IF_EVERY_T(SEVERITY, seconds, condition) DLOG_LOGGER_IF_EVERY_T(SEVERITY, (seconds), (condition), get_logger())
#define POLARIS_DLOG_ONCE(SEVERITY)                           DLOG_LOGGER_ONCE(SEVERITY, get_logger())
#define POLARIS_DLOG_IF_ONCE(SEVERITY, condition)             DLOG_LOGGER_IF_ONCE(SEVERITY, (condition), get_logger())

#define POLARIS_DCHECK(condition) DCHECK_LOGGER(condition, get_logger())
#define POLARIS_DCHECK_NOTNULL(val) DCHECK_NOTNULL_LOGGER(val, get_logger())
#define POLARIS_DCHECK_PTREQ(val1, val2) DCHECK_PTREQ_LOGGER(val1, val2, get_logger())
#define POLARIS_DCHECK_EQ(val1, val2) CHECK_EQ_LOGGER(val1, val2, get_logger())
#define POLARIS_DCHECK_NE(val1, val2) CHECK_NE_LOGGER(val1, val2, get_logger())
#define POLARIS_DCHECK_LE(val1, val2) CHECK_LE_LOGGER(val1, val2, get_logger())
#define POLARIS_DCHECK_LT(val1, val2) CHECK_LT_LOGGER(val1, val2, get_logger())
#define POLARIS_DCHECK_GE(val1, val2) CHECK_GE_LOGGER(val1, val2, get_logger())
#define POLARIS_DCHECK_GT(val1, val2) CHECK_GT_LOGGER(val1, val2, get_logger())


#define POLARIS_DPLOG(SEVERITY)                                  DPLOG_LOGGER(SEVERITY, get_logger())
#define POLARIS_DPLOG_IF(SEVERITY, condition)                    DPLOG_LOGGER_IF(SEVERITY, (condition), get_logger())
#define POLARIS_DPLOG_EVERY_N(SEVERITY, N)                       DPLOG_LOGGER_EVERY_N(SEVERITY, N, get_logger())
#define POLARIS_DPLOG_IF_EVERY_N(SEVERITY, N, condition)         DPLOG_LOGGER_IF_EVERY_N(SEVERITY, (N), (condition), get_logger())
#define POLARIS_DPLOG_FIRST_N(SEVERITY, N)                       DPLOG_LOGGER_FIRST_N(SEVERITY, (N), true, get_logger())
#define POLARIS_DPLOG_IF_FIRST_N(SEVERITY, N, condition)         DPLOG_LOGGER_IF_FIRST_N(SEVERITY, (N), (condition), get_logger())
#define POLARIS_DPLOG_EVERY_T(SEVERITY, seconds)                 DPLOG_LOGGER_EVERY_T(SEVERITY, (seconds), get_logger())
#define POLARIS_DPLOG_IF_EVERY_T(SEVERITY, seconds, condition)   DPLOG_LOGGER_IF_EVERY_T(SEVERITY, (seconds), (condition), get_logger())
#define POLARIS_DPLOG_ONCE(SEVERITY)                             DPLOG_LOGGER_ONCE(SEVERITY, get_logger())
#define POLARIS_DPLOG_IF_ONCE(SEVERITY, condition)               DPLOG_LOGGER_IF_ONCE(SEVERITY, (condition), get_logger())

#define POLARIS_DPCHECK(condition)        DPCHECK_LOGGER(condition, get_logger())
#define POLARIS_DPCHECK_NOTNULL(val)      DPCHECK_NOTNULL_LOGGER(val, get_logger())
#define POLARIS_DPCHECK_PTREQ(val1, val2) DPCHECK_PTREQ_LOGGER(val1, val2, get_logger())
#define POLARIS_DPCHECK_EQ(val1, val2) DPCHECK_EQ_LOGGER(val1, val2, get_logger())
#define POLARIS_DPCHECK_NE(val1, val2) DPCHECK_NE_LOGGER(val1, val2, get_logger())
#define POLARIS_DPCHECK_LE(val1, val2) DPCHECK_LE_LOGGER(val1, val2, get_logger())
#define POLARIS_DPCHECK_LT(val1, val2) DPCHECK_LT_LOGGER(val1, val2, get_logger())
#define POLARIS_DPCHECK_GE(val1, val2) DPCHECK_GE_LOGGER(val1, val2, get_logger())
#define POLARIS_DPCHECK_GT(val1, val2) DPCHECK_GT_LOGGER(val1, val2, get_logger())

#define POLARIS_DVLOG(verboselevel) DVLOG_LOGGER(verboselevel, get_logger())
#define POLARIS_DVLOG_IF(verboselevel, condition) DVLOG_IF_LOGGER(verboselevel, (condition), get_logger())
#define POLARIS_DVLOG_EVERY_N(verboselevel, N) DVLOG_EVERY_N_LOGGER(verboselevel, N, get_logger())
#define POLARIS_DVLOG_IF_EVERY_N(verboselevel, N, condition) DVLOG_IF_EVERY_N_LOGGER(verboselevel, (N), (condition), get_logger())
#define POLARIS_DVLOG_FIRST_N(verboselevel, N) DVLOG_FIRST_N_LOGGER(verboselevel, (N), get_logger())
#define POLARIS_DVLOG_IF_FIRST_N(verboselevel, N, condition) DVLOG_IF_FIRST_N_LOGGER(verboselevel, (N), (condition), get_logger())
#define POLARIS_DVLOG_EVERY_T(verboselevel, seconds) DVLOG_EVERY_T_LOGGER(verboselevel, (seconds), get_logger())
#define POLARIS_DVLOG_IF_EVERY_T(verboselevel, seconds, condition) DVLOG_IF_EVERY_T_LOGGER(verboselevel, (seconds), (condition), get_logger())
#define POLARIS_DVLOG_ONCE(verboselevel) DVLOG_ONCE_LOGGER(verboselevel, get_logger())
#define POLARIS_DVLOG_IF_ONCE(verboselevel, condition) DVLOG_IF_ONCE_LOGGER(verboselevel, (condition), get_logger())

#endif

