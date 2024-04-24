/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Copyright 2004-present Facebook. All Rights Reserved.
// -*- c++ -*-

#include "index_factory_c.h"
#include <polaris/index_factory.h>
#include <cstring>
#include <polaris/capi/macros_impl.h>

using polaris::Index;

/** Build and index with the sequence of processing steps described in
 *  the string.
 */
int faiss_index_factory(
        FaissIndex** p_index,
        int d,
        const char* description,
        FaissMetricType metric) {
    try {
        *p_index = reinterpret_cast<FaissIndex*>(polaris::index_factory(
                d, description, static_cast<polaris::MetricType>(metric)));
    }
    CATCH_AND_HANDLE
}
