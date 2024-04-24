/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Copyright 2004-present Facebook. All Rights Reserved.
// -*- c++ -*-

#include "GpuIndex_c.h"
#include <polaris/gpu/GpuIndex.h>
#include <polaris/capi/macros_impl.h>

using polaris::gpu::GpuIndexConfig;

DEFINE_GETTER(GpuIndexConfig, int, device)
