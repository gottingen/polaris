
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
#include <vector>
#include <stdexcept>
#include <system_error>
#include <polaris/utility/platform_macros.h>

namespace polaris {

    /// Base class for Faiss exceptions
    class PolarisException : public std::exception {
    public:
        explicit PolarisException(const std::string &msg);
        POLARIS_API PolarisException(const std::string &message, int errorCode);
        POLARIS_API PolarisException(const std::string &message, int errorCode, const std::string &funcSig,
                                 const std::string &fileName, uint32_t lineNum);

        PolarisException(
                const std::string &msg,
                const char *funcName,
                const char *file,
                int line);

        /// from std::exception
        const char *what() const noexcept override;

        std::string msg;
        int _errorCode;
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
