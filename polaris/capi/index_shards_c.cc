/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <polaris/capi/index_shards_c.h>
#include <polaris/index/index_shards.h>
#include <polaris/capi/macros_impl.h>

using polaris::Index;
using polaris::IndexShards;

DEFINE_DESTRUCTOR(IndexShards)

DEFINE_GETTER(IndexShards, int, own_indices)
DEFINE_SETTER(IndexShards, int, own_indices)

DEFINE_GETTER(IndexShards, int, successive_ids)
DEFINE_SETTER(IndexShards, int, successive_ids)

int faiss_IndexShards_new(FaissIndexShards** p_index, idx_t d) {
    try {
        auto out = new IndexShards(d);
        *p_index = reinterpret_cast<FaissIndexShards*>(out);
    }
    CATCH_AND_HANDLE
}

int faiss_IndexShards_new_with_options(
        FaissIndexShards** p_index,
        idx_t d,
        int threaded,
        int successive_ids) {
    try {
        auto out = new IndexShards(
                d,
                static_cast<bool>(threaded),
                static_cast<bool>(successive_ids));
        *p_index = reinterpret_cast<FaissIndexShards*>(out);
    }
    CATCH_AND_HANDLE
}

int faiss_IndexShards_add_shard(FaissIndexShards* index, FaissIndex* shard) {
    try {
        reinterpret_cast<IndexShards*>(index)->add_shard(
                reinterpret_cast<Index*>(shard));
    }
    CATCH_AND_HANDLE
}

int faiss_IndexShards_remove_shard(FaissIndexShards* index, FaissIndex* shard) {
    try {
        reinterpret_cast<IndexShards*>(index)->remove_shard(
                reinterpret_cast<Index*>(shard));
    }
    CATCH_AND_HANDLE
}

FaissIndex* faiss_IndexShards_at(FaissIndexShards* index, int i) {
    auto shard = reinterpret_cast<IndexShards*>(index)->at(i);
    return reinterpret_cast<FaissIndex*>(shard);
}
