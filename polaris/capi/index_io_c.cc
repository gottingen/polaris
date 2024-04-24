/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

//  Copyright 2004-present Facebook. All Rights Reserved
// -*- c++ -*-
// I/O code for indexes

#include "index_io_c.h"
#include <polaris/index_io.h>
#include <polaris/capi/macros_impl.h>

using polaris::Index;
using polaris::IndexBinary;

int faiss_write_index(const FaissIndex* idx, FILE* f) {
    try {
        polaris::write_index(reinterpret_cast<const Index*>(idx), f);
    }
    CATCH_AND_HANDLE
}

int faiss_write_index_fname(const FaissIndex* idx, const char* fname) {
    try {
        polaris::write_index(reinterpret_cast<const Index*>(idx), fname);
    }
    CATCH_AND_HANDLE
}

int faiss_read_index(FILE* f, int io_flags, FaissIndex** p_out) {
    try {
        auto out = polaris::read_index(f, io_flags);
        *p_out = reinterpret_cast<FaissIndex*>(out);
    }
    CATCH_AND_HANDLE
}

int faiss_read_index_fname(
        const char* fname,
        int io_flags,
        FaissIndex** p_out) {
    try {
        auto out = polaris::read_index(fname, io_flags);
        *p_out = reinterpret_cast<FaissIndex*>(out);
    }
    CATCH_AND_HANDLE
}

int faiss_write_index_binary(const FaissIndexBinary* idx, FILE* f) {
    try {
        polaris::write_index_binary(reinterpret_cast<const IndexBinary*>(idx), f);
    }
    CATCH_AND_HANDLE
}

int faiss_write_index_binary_fname(
        const FaissIndexBinary* idx,
        const char* fname) {
    try {
        polaris::write_index_binary(
                reinterpret_cast<const IndexBinary*>(idx), fname);
    }
    CATCH_AND_HANDLE
}

int faiss_read_index_binary(FILE* f, int io_flags, FaissIndexBinary** p_out) {
    try {
        auto out = polaris::read_index_binary(f, io_flags);
        *p_out = reinterpret_cast<FaissIndexBinary*>(out);
    }
    CATCH_AND_HANDLE
}

int faiss_read_index_binary_fname(
        const char* fname,
        int io_flags,
        FaissIndexBinary** p_out) {
    try {
        auto out = polaris::read_index_binary(fname, io_flags);
        *p_out = reinterpret_cast<FaissIndexBinary*>(out);
    }
    CATCH_AND_HANDLE
}
