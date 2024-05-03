/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// -*- c++ -*-

#include <polaris/utility/polaris_exception.h>
#include <sstream>

#ifdef __GNUG__

#include <cxxabi.h>

#endif

namespace polaris {
    std::string package_string(const std::string &item_name, const std::string &item_val)
    {
        return std::string("[") + item_name + ": " + std::string(item_val) + std::string("]");
    }


    PolarisException::PolarisException(const std::string &m) : msg(m) {}
    PolarisException::PolarisException(const std::string &message, int errorCode)
            : msg(message), _errorCode(errorCode)
    {
    }

    PolarisException::PolarisException(const std::string &message, int errorCode, const std::string &funcSig,
                               const std::string &fileName, uint32_t lineNum)
            : PolarisException(package_string(std::string("FUNC"), funcSig) + package_string(std::string("FILE"), fileName) +
                           package_string(std::string("LINE"), std::to_string(lineNum)) + "  " + message,
                           errorCode)
    {
    }

    PolarisException::PolarisException(const std::string &m,const std::string &funcName,const std::string&file,int line) {
        set(m, funcName, file, line);
    }

    void PolarisException::set(const std::string &m, const std::string &function,const std::string &file,  size_t line) {
        std::stringstream ss;
        ss << file << ":" << function << ":" << line << ": " << m;
        msg = ss.str();
    }

    const char *PolarisException::what() const noexcept {
        return msg.c_str();
    }

    FileException::FileException(const std::string &filename, std::system_error &e, const std::string &funcSig,
                                 const std::string &fileName, uint32_t lineNum)
            : PolarisException(std::string(" While opening file \'") + filename + std::string("\', error code: ") +
                           std::to_string(e.code().value()) + "  " + e.code().message(),
                           e.code().value(), funcSig, fileName, lineNum)
    {
    }

    void handleExceptions(
            std::vector<std::pair<int, std::exception_ptr>> &exceptions) {
        if (exceptions.size() == 1) {
            // throw the single received exception directly
            std::rethrow_exception(exceptions.front().second);

        } else if (exceptions.size() > 1) {
            // multiple exceptions; aggregate them and return a single exception
            std::stringstream ss;

            for (auto &p: exceptions) {
                try {
                    std::rethrow_exception(p.second);
                } catch (std::exception &ex) {
                    if (ex.what()) {
                        // exception message available
                        ss << "Exception thrown from index " << p.first << ": "
                           << ex.what() << "\n";
                    } else {
                        // No message available
                        ss << "Unknown exception thrown from index " << p.first
                           << "\n";
                    }
                } catch (...) {
                    ss << "Unknown exception thrown from index " << p.first << "\n";
                }
            }

            throw PolarisException(ss.str());
        }
    }

// From
// https://stackoverflow.com/questions/281818/unmangling-the-result-of-stdtype-infoname

    std::string demangle_cpp_symbol(const char *name) {
#ifdef __GNUG__
        int status = -1;
        const char *res = abi::__cxa_demangle(name, nullptr, nullptr, &status);
        std::string sres;
        if (status == 0) {
            sres = res;
        }
        free((void *) res);
        return sres;
#else
        // don't know how to do this on other platforms
        return std::string(name);
#endif
    }

} // namespace polaris
