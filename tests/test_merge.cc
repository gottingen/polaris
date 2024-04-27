/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <cstdio>
#include <cstdlib>
#include <random>

#include <unistd.h>

#include <gtest/gtest.h>

#include <polaris/index/ivf_lib.h>
#include <polaris/index/index_flat.h>
#include <polaris/index/index_ivf_flat.h>
#include <polaris/index/index_ivf_pq.h>
#include <polaris/index/index_pre_transform.h>
#include <polaris/index/meta_indexes.h>
#include <polaris/invlists/on_disk_inverted_lists.h>

namespace {

struct Tempfilename {
    static pthread_mutex_t mutex;

    std::string filename = "/tmp/faiss_tmp_XXXXXX";

    Tempfilename() {
        pthread_mutex_lock(&mutex);
        int fd = mkstemp(&filename[0]);
        close(fd);
        pthread_mutex_unlock(&mutex);
    }

    ~Tempfilename() {
        if (access(filename.c_str(), F_OK)) {
            unlink(filename.c_str());
        }
    }

    const char* c_str() {
        return filename.c_str();
    }
};

pthread_mutex_t Tempfilename::mutex = PTHREAD_MUTEX_INITIALIZER;

typedef polaris::idx_t idx_t;

// parameters to use for the test
int d = 64;
size_t nb = 1000;
size_t nq = 100;
int nindex = 4;
int k = 10;
int nlist = 40;

struct CommonData {
    std::vector<float> database;
    std::vector<float> queries;
    std::vector<idx_t> ids;
    polaris::IndexFlatL2 quantizer;

    CommonData() : database(nb * d), queries(nq * d), ids(nb), quantizer(d) {
        std::mt19937 rng;
        std::uniform_real_distribution<> distrib;
        for (size_t i = 0; i < nb * d; i++) {
            database[i] = distrib(rng);
        }
        for (size_t i = 0; i < nq * d; i++) {
            queries[i] = distrib(rng);
        }
        for (int i = 0; i < nb; i++) {
            ids[i] = 123 + 456 * i;
        }
        { // just to train the quantizer
            polaris::IndexIVFFlat iflat(&quantizer, d, nlist);
            iflat.train(nb, database.data());
        }
    }
};

CommonData cd;

/// perform a search on shards, then merge and search again and
/// compare results.
int compare_merged(
        polaris::IndexShards* index_shards,
        bool shift_ids,
        bool standard_merge = true) {
    std::vector<idx_t> refI(k * nq);
    std::vector<float> refD(k * nq);

    index_shards->search(nq, cd.queries.data(), k, refD.data(), refI.data());
    Tempfilename filename;

    std::vector<idx_t> newI(k * nq);
    std::vector<float> newD(k * nq);

    if (standard_merge) {
        for (int i = 1; i < nindex; i++) {
            polaris::ivflib::merge_into(
                    index_shards->at(0), index_shards->at(i), shift_ids);
        }

        index_shards->syncWithSubIndexes();
    } else {
        std::vector<const polaris::InvertedLists*> lists;
        polaris::IndexIVF* index0 = nullptr;
        size_t ntotal = 0;
        for (int i = 0; i < nindex; i++) {
            auto index_ivf =
                    dynamic_cast<polaris::IndexIVF*>(index_shards->at(i));
            assert(index_ivf);
            if (i == 0) {
                index0 = index_ivf;
            }
            lists.push_back(index_ivf->invlists);
            ntotal += index_ivf->ntotal;
        }

        auto il = new polaris::OnDiskInvertedLists(
                index0->nlist, index0->code_size, filename.c_str());

        il->merge_from(lists.data(), lists.size());

        index0->replace_invlists(il, true);
        index0->ntotal = ntotal;
    }
    // search only on first index
    index_shards->at(0)->search(
            nq, cd.queries.data(), k, newD.data(), newI.data());

    size_t ndiff = 0;
    for (size_t i = 0; i < k * nq; i++) {
        if (refI[i] != newI[i]) {
            ndiff++;
        }
    }
    return ndiff;
}

} // namespace

// test on IVFFlat with implicit numbering
TEST(MERGE, merge_flat_no_ids) {
    polaris::IndexShards index_shards(d);
    index_shards.own_indices = true;
    for (int i = 0; i < nindex; i++) {
        index_shards.add_shard(
                new polaris::IndexIVFFlat(&cd.quantizer, d, nlist));
    }
    EXPECT_TRUE(index_shards.is_trained);
    index_shards.add(nb, cd.database.data());
    size_t prev_ntotal = index_shards.ntotal;
    int ndiff = compare_merged(&index_shards, true);
    EXPECT_EQ(prev_ntotal, index_shards.ntotal);
    EXPECT_EQ(0, ndiff);
}

// test on IVFFlat, explicit ids
TEST(MERGE, merge_flat) {
    polaris::IndexShards index_shards(d, false, false);
    index_shards.own_indices = true;

    for (int i = 0; i < nindex; i++) {
        index_shards.add_shard(
                new polaris::IndexIVFFlat(&cd.quantizer, d, nlist));
    }

    EXPECT_TRUE(index_shards.is_trained);
    index_shards.add_with_ids(nb, cd.database.data(), cd.ids.data());
    int ndiff = compare_merged(&index_shards, false);
    EXPECT_GE(0, ndiff);
}

// test on IVFFlat and a VectorTransform
TEST(MERGE, merge_flat_vt) {
    polaris::IndexShards index_shards(d, false, false);
    index_shards.own_indices = true;

    // here we have to retrain because of the vectorTransform
    polaris::RandomRotationMatrix rot(d, d);
    rot.init(1234);
    polaris::IndexFlatL2 quantizer(d);

    { // just to train the quantizer
        polaris::IndexIVFFlat iflat(&quantizer, d, nlist);
        polaris::IndexPreTransform ipt(&rot, &iflat);
        ipt.train(nb, cd.database.data());
    }

    for (int i = 0; i < nindex; i++) {
        polaris::IndexPreTransform* ipt = new polaris::IndexPreTransform(
                new polaris::RandomRotationMatrix(rot),
                new polaris::IndexIVFFlat(&quantizer, d, nlist));
        ipt->own_fields = true;
        index_shards.add_shard(ipt);
    }
    EXPECT_TRUE(index_shards.is_trained);
    index_shards.add_with_ids(nb, cd.database.data(), cd.ids.data());
    size_t prev_ntotal = index_shards.ntotal;
    int ndiff = compare_merged(&index_shards, false);
    EXPECT_EQ(prev_ntotal, index_shards.ntotal);
    EXPECT_GE(0, ndiff);
}

// put the merged invfile on disk
TEST(MERGE, merge_flat_ondisk) {
    polaris::IndexShards index_shards(d, false, false);
    index_shards.own_indices = true;
    Tempfilename filename;

    for (int i = 0; i < nindex; i++) {
        auto ivf = new polaris::IndexIVFFlat(&cd.quantizer, d, nlist);
        if (i == 0) {
            auto il = new polaris::OnDiskInvertedLists(
                    ivf->nlist, ivf->code_size, filename.c_str());
            ivf->replace_invlists(il, true);
        }
        index_shards.add_shard(ivf);
    }

    EXPECT_TRUE(index_shards.is_trained);
    index_shards.add_with_ids(nb, cd.database.data(), cd.ids.data());
    int ndiff = compare_merged(&index_shards, false);

    EXPECT_EQ(ndiff, 0);
}

// now use ondisk specific merge
TEST(MERGE, merge_flat_ondisk_2) {
    polaris::IndexShards index_shards(d, false, false);
    index_shards.own_indices = true;

    for (int i = 0; i < nindex; i++) {
        index_shards.add_shard(
                new polaris::IndexIVFFlat(&cd.quantizer, d, nlist));
    }
    EXPECT_TRUE(index_shards.is_trained);
    index_shards.add_with_ids(nb, cd.database.data(), cd.ids.data());
    int ndiff = compare_merged(&index_shards, false, false);
    EXPECT_GE(0, ndiff);
}
