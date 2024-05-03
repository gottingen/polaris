
/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// -*- c++ -*-

#pragma once

#include <exception>
#include <string>
#include <utility>
#include <sstream>
#include <vector>
#include <stdexcept>
#include <system_error>
#include <polaris/utility/platform_macros.h>

namespace polaris {

    /// Base class for Faiss exceptions
    class PolarisException : public std::exception {
    public:
        PolarisException() = default;

        explicit PolarisException(const std::string &msg);

        POLARIS_API PolarisException(const std::string &message, int errorCode);

        POLARIS_API PolarisException(const std::string &message, int errorCode, const std::string &funcSig,
                                     const std::string &fileName, uint32_t lineNum);

        PolarisException(const std::string &msg, const std::string &funcName, const std::string &file, int line);

        PolarisException(const std::stringstream &imsg, const std::string &funcName, const std::string &file,
                         int line) {
            set(imsg.str(), funcName, file, line);
        }

        void set(const std::string &m, const std::string &function, const std::string &file, size_t line);

        /// from std::exception
        const char *what() const noexcept override;

        PolarisException &operator=(const PolarisException &e) {
            msg = e.msg;
            _errorCode = e._errorCode;
            return *this;
        }

        std::string &get_message() { return msg; }

        std::string msg;
        int _errorCode{0};
    };

    class ThreadTerminationException : public PolarisException {
    public:
        ThreadTerminationException(const std::string &msg, const std::string &funcName, const std::string &file,
                                   int line) { set(msg, funcName, file, line); }

        ThreadTerminationException(const std::stringstream &imsg, const std::string &funcName, const std::string &file,
                                   int line) { set(imsg.str(), funcName, file, line); }
    };

    class FileException : public PolarisException {
    public:
        POLARIS_API FileException(const std::string &filename, std::system_error &e, const std::string &funcSig,
                                  const std::string &fileName, uint32_t lineNum);
    };

    /// Handle multiple exceptions from worker threads, throwing an appropriate
    /// exception that aggregates the information
    /// The pair int is the thread that generated the exception
    void handleExceptions(
            std::vector<std::pair<int, std::exception_ptr>> &exceptions);

    /** RAII object for a set of possibly transformed vectors (deallocated only if
     * they are indeed transformed)
     */
    struct TransformedVectors {
        const float *x;
        bool own_x;

        TransformedVectors(const float *x_orig, const float *x) : x(x) {
            own_x = x_orig != x;
        }

        ~TransformedVectors() {
            if (own_x) {
                delete[] x;
            }
        }
    };

    /// make typeids more readable
    std::string demangle_cpp_symbol(const char *name);

} // namespace polaris

#define POLARIS_THROW_EX(MESSAGE) throw polaris::PolarisException(MESSAGE, __PRETTY_FUNCTION__, __FILE__, (size_t)__LINE__)
#define POLARIS_THROW_TYPED_EX(MESSAGE, TYPE)    throw polaris::TYPE(MESSAGE, __PRETTY_FUNCTION__, __FILE__, (size_t)__LINE__)