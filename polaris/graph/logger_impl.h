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

#pragma once

#include <sstream>
#include <mutex>

#include <polaris/internal/polaris_exception.h>
#include <polaris/graph/logger.h>

namespace polaris
{
#ifdef ENABLE_CUSTOM_LOGGER
class ANNStreamBuf : public std::basic_streambuf<char>
{
  public:
    POLARIS_API explicit ANNStreamBuf(FILE *fp);
    POLARIS_API ~ANNStreamBuf();

    POLARIS_API bool is_open() const
    {
        return true; // because stdout and stderr are always open.
    }
    POLARIS_API void close();
    POLARIS_API virtual int underflow();
    POLARIS_API virtual int overflow(int c);
    POLARIS_API virtual int sync();

  private:
    FILE *_fp;
    char *_buf;
    int _bufIndex;
    std::mutex _mutex;
    LogLevel _logLevel;

    int flush();
    void logImpl(char *str, int numchars);

    // Why the two buffer-sizes? If we are running normally, we are basically
    // interacting with a character output system, so we short-circuit the
    // output process by keeping an empty buffer and writing each character
    // to stdout/stderr. But if we are running in OLS, we have to take all
    // the text that is written to polaris::cout/diskann:cerr, consolidate it
    // and push it out in one-shot, because the OLS infra does not give us
    // character based output. Therefore, we use a larger buffer that is large
    // enough to store the longest message, and continuously add characters
    // to it. When the calling code outputs a std::endl or std::flush, sync()
    // will be called and will output a log level, component name, and the text
    // that has been collected. (sync() is also called if the buffer is full, so
    // overflows/missing text are not a concern).
    // This implies calling code _must_ either print std::endl or std::flush
    // to ensure that the message is written immediately.

    static const int BUFFER_SIZE = 1024;

    ANNStreamBuf(const ANNStreamBuf &);
    ANNStreamBuf &operator=(const ANNStreamBuf &);
};
#endif
} // namespace polaris
