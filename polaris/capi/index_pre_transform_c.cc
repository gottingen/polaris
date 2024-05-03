/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Copyright 2004-present Facebook. All Rights Reserved.
// -*- c++ -*-

#include <polaris/capi/index_pre_transform_c.h>
#include <polaris/faiss/index_pre_transform.h>
#include <polaris/faiss/vector_transform.h>
#include <polaris/capi/macros_impl.h>

using polaris::Index;
using polaris::IndexPreTransform;
using polaris::VectorTransform;

extern "C" {

DEFINE_DESTRUCTOR(IndexPreTransform)
DEFINE_INDEX_DOWNCAST(IndexPreTransform)

DEFINE_GETTER_PERMISSIVE(IndexPreTransform, FaissIndex*, index)

DEFINE_GETTER(IndexPreTransform, int, own_fields)
DEFINE_SETTER(IndexPreTransform, int, own_fields)

int faiss_IndexPreTransform_new(FaissIndexPreTransform** p_index) {
    try {
        *p_index = reinterpret_cast<FaissIndexPreTransform*>(
                new IndexPreTransform());
    }
    CATCH_AND_HANDLE
}

int faiss_IndexPreTransform_new_with(
        FaissIndexPreTransform** p_index,
        FaissIndex* index) {
    try {
        auto ind = reinterpret_cast<Index*>(index);
        *p_index = reinterpret_cast<FaissIndexPreTransform*>(
                new IndexPreTransform(ind));
    }
    CATCH_AND_HANDLE
}

int faiss_IndexPreTransform_new_with_transform(
        FaissIndexPreTransform** p_index,
        FaissVectorTransform* ltrans,
        FaissIndex* index) {
    try {
        auto lt = reinterpret_cast<VectorTransform*>(ltrans);
        auto ind = reinterpret_cast<Index*>(index);
        *p_index = reinterpret_cast<FaissIndexPreTransform*>(
                new IndexPreTransform(lt, ind));
    }
    CATCH_AND_HANDLE
}

int faiss_IndexPreTransform_prepend_transform(
        FaissIndexPreTransform* index,
        FaissVectorTransform* ltrans) {
    try {
        auto lt = reinterpret_cast<VectorTransform*>(ltrans);
        reinterpret_cast<IndexPreTransform*>(index)->prepend_transform(lt);
    }
    CATCH_AND_HANDLE
}
}
