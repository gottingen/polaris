// Copyright 2023 The Elastic-AI Authors.
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

#include <cstdint>
#include <vector>
#include <turbo/container/flat_hash_set.h>
#include <polaris/core/lock.h>

namespace polaris {

    struct SearchOptions {
        int64_t n;
        int64_t k;
        std::vector<float> query;
        uint32_t nprobe;
        float radius;
    };

    struct EngineConfig {
        size_t max_elements;
        size_t batch_size;
        uint32_t dimension;
    };

    /**
     * UnifiedImpl is the base class for all index implementations.
     * It provides the basic interface for all index implementations.
     * The UnifiedIndex create a UnifiedImpl instance by follow steps:
     * @code
     * UnifiedImpl* impl = UnifiedImpl::create("index_type");
     * impl->set_conf(config);
     * impl->init();
     * @endcode
     */

    class UnifiedImpl {
    public:
        static const size_t DEFAULT_MAX_ELEMENTS = 1000000;
        static const size_t DEFAULT_BATCH_SIZE = 1000;
    public:
        UnifiedImpl();

        virtual ~UnifiedImpl();

        virtual int init() = 0;

        virtual size_t size() = 0;

        virtual bool support_update() = 0;

        virtual bool support_delete() = 0;

        virtual bool need_model() = 0;

        virtual void add_with_ids(const std::vector<int64_t> &ids, std::vector<float> &vecs) = 0;

        virtual void search(
                SearchOptions &option, std::vector<float> &distances, std::vector<int64_t> &labels) = 0;

        // do not using template to avoid c api compatibility
        virtual void remove(const turbo::flat_hash_set<uint64_t> &delete_ids) = 0;

        virtual void remove(const std::vector<uint64_t> &delete_ids) = 0;

        virtual void update(const std::vector<int64_t> &ids, std::vector<float> &vecs) = 0;

        virtual int load(const std::string &file, WriteLock &index_wlock) = 0;

        virtual int save(const std::string &file) = 0;

        void set_max_elements(size_t max_elements) {
            if (max_elements > max_elements_) {
                max_elements_ = max_elements;
            }
        }

        void set_conf(const EngineConfig &conf) {
            config_ = conf;
        }

        virtual int build_batch_size() {
            return DEFAULT_BATCH_SIZE;
        }

        virtual void shrink_to_fit() {
        }

    protected:
        EngineConfig config_;
        size_t max_elements_ = DEFAULT_MAX_ELEMENTS;
    };

    using UnifiedImplPtr = std::shared_ptr<UnifiedImpl>;

    class UnifiedIndex {
    public:
        UnifiedIndex();
    };
}  // namespace polaris
