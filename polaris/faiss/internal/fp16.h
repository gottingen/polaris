/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cstdint>

#include <polaris/utility/platform_macros.h>

#if defined(__F16C__)
#include <polaris/faiss/internal/fp16-fp16c.h>
#elif defined(__aarch64__)
#include <polaris/faiss/internal/fp16-arm.h>
#else
#include <polaris/faiss/internal/fp16-inl.h>
#endif
