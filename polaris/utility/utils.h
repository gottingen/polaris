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

#include <string>
#include <vector>
#include <sstream>
#include <iostream>
#include <fstream>
#include <sys/sysinfo.h>
#include <unistd.h>
#include <cmath>
#include <polaris/utility/polaris_exception.h>
#include <sys/time.h>
#include <fcntl.h>

#if __linux__

#include <sys/sysinfo.h>

#endif

namespace polaris {

    class Common {
    public:
        static void tokenize(const std::string &str, std::vector<std::string> &token, const std::string seps) {
            std::string::size_type current = 0;
            std::string::size_type next;
            while ((next = str.find_first_of(seps, current)) != std::string::npos) {
                token.push_back(str.substr(current, next - current));
                current = next + 1;
            }
            std::string t = str.substr(current);
            token.push_back(t);
        }

        static double strtod(const std::string &str) {
            char *e;
            double val = std::strtod(str.c_str(), &e);
            if (*e != 0) {
                std::stringstream msg;
                msg << "Invalid string. " << e;
                POLARIS_THROW_EX(msg);
            }
            return val;
        }

        static float strtof(const std::string &str) {
            char *e;
            double val = std::strtof(str.c_str(), &e);
            if (*e != 0) {
                std::stringstream msg;
                msg << "Invalid string. " << e;
                POLARIS_THROW_EX(msg);
            }
            return val;
        }

        static long strtol(const std::string &str, int base = 10) {
            char *e;
            long val = std::strtol(str.c_str(), &e, base);
            if (*e != 0) {
                std::stringstream msg;
                msg << "Invalid string. " << e;
                POLARIS_THROW_EX(msg);
            }
            return val;
        }

#if __linux__

        static unsigned long getTotalRam() {
            struct sysinfo info;
            sysinfo(&info);
            return info.totalram;
        }

#endif

        template<typename T>
        static void extractVector(const std::string &textLine, const std::string &sep, T &object) {
            std::vector<std::string> tokens;
            polaris::Common::tokenize(textLine, tokens, sep);
            size_t idx;
            for (idx = 0; idx < tokens.size(); idx++) {
                if (tokens[idx].size() == 0) {
                    if (idx + 1 == tokens.size()) {
                        break;
                    }
                    std::stringstream msg;
                    msg << "Common::extractVector: No data. sep=(" << sep << "):" << idx << ": " << textLine;
                    POLARIS_THROW_EX(msg);
                }
                char *e;
                double v = ::strtod(tokens[idx].c_str(), &e);
                if (*e != 0) {
                    std::cerr << "Common::extractVector: Warning! Not numerical value. [" << e << "] " << std::endl;
                    break;
                }
                object.push_back(v);
            }
        }


        static std::string getProcessStatus(const std::string &stat) {
            pid_t pid = getpid();
            std::stringstream str;
            str << "/proc/" << pid << "/status";
            std::ifstream procStatus(str.str());
            if (!procStatus.fail()) {
                std::string line;
                while (getline(procStatus, line)) {
                    std::vector<std::string> tokens;
                    polaris::Common::tokenize(line, tokens, ": \t");
                    if (tokens[0] == stat) {
                        for (size_t i = 1; i < tokens.size(); i++) {
                            if (tokens[i].empty()) {
                                continue;
                            }
                            return tokens[i];
                        }
                    }
                }
            }
            return "-1";
        }

        // size unit is kbyte
        static int getProcessVmSize() { return strtol(getProcessStatus("VmSize")); }

        static int getProcessVmPeak() { return strtol(getProcessStatus("VmPeak")); }

        static int getProcessVmRSS() { return strtol(getProcessStatus("VmRSS")); }

        static int getProcessVmHWM() { return strtol(getProcessStatus("VmHWM")); }

        static int getSystemHWM() {
#if __linux__
            struct sysinfo info;
            sysinfo(&info);
            return info.totalram / 1024;
#else
            return 0;
#endif
        }

        static std::string sizeToString(float size) {
            char unit = 'K';
            if (size > 1024) {
                size /= 1024;
                unit = 'M';
            }
            if (size > 1024) {
                size /= 1024;
                unit = 'G';
            }
            size = ::round(size * 100) / 100;
            std::stringstream str;
            str << size << " " << unit;
            return str.str();
        }

        static std::string getProcessVmSizeStr() { return sizeToString(getProcessVmSize()); }

        static std::string getProcessVmPeakStr() { return sizeToString(getProcessVmPeak()); }

        static std::string getProcessVmRSSStr() { return sizeToString(getProcessVmRSS()); }

        static std::string getProcessVmHWMStr() { return sizeToString(getProcessVmHWM()); }

        static std::string getSystemHWMStr() { return sizeToString(getSystemHWM()); }
    };

    class CpuInfo {
    public:
        enum SimdType {
            SimdTypeAVX = 0,
            SimdTypeAVX2 = 1,
            SimdTypeAVX512F = 2,
            SimdTypeAVX512VL = 3,
            SimdTypeAVX512BW = 4,
            SimdTypeAVX512DQ = 5,
            SimdTypeAVX512CD = 6,
            SimdTypeAVX512ER = 7,
            SimdTypeAVX512PF = 8,
            SimdTypeAVX512VBMI = 9,
            SimdTypeAVX512IFMA = 10,
            SimdTypeAVX5124VNNIW = 11,
            SimdTypeAVX5124FMAPS = 12,
            SimdTypeAVX512VPOPCNTDQ = 13,
            SimdTypeAVX512VBMI2 = 14,
            SimdTypeAVX512VNNI = 15
        };

        CpuInfo() {}

        static bool is(SimdType type) {
#ifndef __BUILTIN_CPU_SUPPORTS__
            return true;
#else
            __builtin_cpu_init();
            switch (type) {
#if defined(__AVX__)
            case SimdTypeAVX: return __builtin_cpu_supports("avx") > 0; break;
#endif
#if defined(__AVX2__)
            case SimdTypeAVX2: return __builtin_cpu_supports("avx2") > 0; break;
#endif
#if defined(__AVX512F__)
            case SimdTypeAVX512F: return __builtin_cpu_supports("avx512f") > 0; break;
#endif
#if defined(__AVX512VL__)
            case SimdTypeAVX512VL: return __builtin_cpu_supports("avx512vl") > 0; break;
#endif
#if defined(__AVX512BW__)
            case SimdTypeAVX512BW: return __builtin_cpu_supports("avx512bw") > 0; break;
#endif
#if defined(__AVX512DQ__)
            case SimdTypeAVX512DQ: return __builtin_cpu_supports("avx512dq") > 0; break;
#endif
#if defined(__AVX512CD__)
            case SimdTypeAVX512CD: return __builtin_cpu_supports("avx512cd") > 0; break;
#endif
#if defined(__AVX512ER__)
            case SimdTypeAVX512ER: return __builtin_cpu_supports("avx512er") > 0; break;
#endif
#if defined(__AVX512PF__)
            case SimdTypeAVX512PF: return __builtin_cpu_supports("avx512pf") > 0; break;
#endif
#if defined(__AVX512VBMI__)
            case SimdTypeAVX512VBMI: return __builtin_cpu_supports("avx512vbmi") > 0; break;
#endif
#if defined(__AVX512IFMA__)
            case SimdTypeAVX512IFMA: return __builtin_cpu_supports("avx512ifma") > 0; break;
#endif
#if defined(__AVX5124VNNIW__)
            case SimdTypeAVX5124VNNIW: return __builtin_cpu_supports("avx5124vnniw") > 0; break;
#endif
#if defined(__AVX5124FMAPS__)
            case SimdTypeAVX5124FMAPS: return __builtin_cpu_supports("avx5124fmaps") > 0; break;
#endif
#if defined(__AVX512VPOPCNTDQ__)
            case SimdTypeAVX512VPOPCNTDQ: return __builtin_cpu_supports("avx512vpopcntdq") > 0; break;
#endif
#if defined(__AVX512VBMI2__)
            case SimdTypeAVX512VBMI2: return __builtin_cpu_supports("avx512vbmi2") > 0; break;
#endif
#if defined(__AVX512VNNI__)
            case SimdTypeAVX512VNNI: return __builtin_cpu_supports("avx512vnni") > 0; break;
#endif
            default: break;
            }
            return false;
#endif
        }

        static bool isAVX512() { return is(SimdTypeAVX512F); };

        static bool isAVX2() { return is(SimdTypeAVX2); };

        static void showSimdTypes() {
            std::cout << getSupportedSimdTypes() << std::endl;
        }

        static std::string getSupportedSimdTypes() {
            static constexpr char const *simdTypes[] = {"avx", "avx2", "avx512f", "avx512vl",
                                                        "avx512bw", "avx512dq", "avx512cd",
                                                        "avx512er", "avx512pf", "avx512vbmi",
                                                        "avx512ifma", "avx5124vnniw",
                                                        "avx5124fmaps", "avx512vpopcntdq",
                                                        "avx512vbmi2", "avx512vnni"};
            std::string types;
            int size = sizeof(simdTypes) / sizeof(simdTypes[0]);
            for (int i = 0; i < size; i++) {
                if (is(static_cast<SimdType>(i))) {
                    types += simdTypes[i];
                }
                if (i != size) {
                    types += " ";
                }
            }
            return types;
        }
    };

    class StdOstreamRedirector {
    public:
        StdOstreamRedirector(bool e = false, const std::string path = "/dev/null",
                             mode_t m = S_IRUSR | S_IWUSR | S_IRGRP | S_IROTH, int f = 2) {
            logFilePath = path;
            mode = m;
            logFD = -1;
            fdNo = f;
            enabled = e;
        }

        ~StdOstreamRedirector() { end(); }

        void enable() { enabled = true; }

        void disable() { enabled = false; }

        void set(bool e) { enabled = e; }

        void bgin(bool e) {
            set(e);
            begin();
        }

        void begin() {
            if (!enabled) {
                return;
            }
            if (logFilePath == "/dev/null") {
                logFD = open(logFilePath.c_str(), O_WRONLY | O_APPEND, mode);
            } else {
                logFD = open(logFilePath.c_str(), O_CREAT | O_WRONLY | O_APPEND, mode);
            }
            if (logFD < 0) {
                std::cerr << "Logger: Cannot begin logging." << std::endl;
                logFD = -1;
                return;
            }
            savedFdNo = dup(fdNo);
            std::cerr << std::flush;
            dup2(logFD, fdNo);
        }

        void end() {
            if (logFD < 0) {
                return;
            }
            std::cerr << std::flush;
            dup2(savedFdNo, fdNo);
            close(savedFdNo);
            savedFdNo = -1;
            close(logFD);
            logFD = -1;
        }

        std::string logFilePath;
        mode_t mode;
        int logFD;
        int savedFdNo;
        int fdNo;
        bool enabled;
    };

}  // namespace polaris
