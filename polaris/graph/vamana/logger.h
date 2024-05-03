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

#include <functional>
#include <iostream>
#include <polaris/utility/platform_macros.h>


namespace polaris
{
#ifdef ENABLE_CUSTOM_LOGGER
POLARIS_API extern std::basic_ostream<char> cout;
POLARIS_API extern std::basic_ostream<char> cerr;
#else
using std::cerr;
using std::cout;
#endif

enum class POLARIS_API LogLevel
{
    LL_Info = 0,
    LL_Error,
    LL_Count
};

#ifdef ENABLE_CUSTOM_LOGGER
POLARIS_API void SetCustomLogger(std::function<void(LogLevel, const char *)> logger);
#endif
} // namespace polaris
