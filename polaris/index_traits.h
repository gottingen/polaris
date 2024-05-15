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
//

#pragma once

#include <collie/base/engine_registry.h>
#include <polaris/unified_index.h>
#include <vector>
#include <string>

namespace polaris {
    /*
    struct unavailable {
        static constexpr bool supported() noexcept { return false; }

        static constexpr bool available() noexcept { return false; }

        static constexpr unsigned version() noexcept { return 0; }

        static constexpr std::size_t alignment() noexcept { return 0; }

        static constexpr bool requires_alignment() noexcept { return false; }

        static constexpr char const *name() noexcept { return "<none>"; }
    };
     */

    struct IndexFlatTraits {
        static constexpr bool supported() noexcept { return true; }

        static constexpr bool available() noexcept { return true; }

        static constexpr unsigned version() noexcept { return static_cast<int>(IndexType::IT_FLAT); }

        static constexpr std::size_t alignment() noexcept { return 0; }

        static constexpr bool requires_alignment() noexcept { return false; }

        static constexpr char const *name() noexcept { return "IndexFlat"; }
    };

    struct IndexFlatIPTraits {
        static constexpr bool supported() noexcept { return true; }

        static constexpr bool available() noexcept { return true; }

        static constexpr unsigned version() noexcept { return static_cast<int>(IndexType::IT_FLATIP); }

        static constexpr std::size_t alignment() noexcept { return 0; }

        static constexpr bool requires_alignment() noexcept { return false; }

        static constexpr char const *name() noexcept { return "IndexFlatIP"; }
    };

    struct IndexFlatL2Traits {
        static constexpr bool supported() noexcept { return true; }

        static constexpr bool available() noexcept { return true; }

        static constexpr unsigned version() noexcept { return static_cast<int>(IndexType::IT_FLATL2); }

        static constexpr std::size_t alignment() noexcept { return 0; }

        static constexpr bool requires_alignment() noexcept { return false; }

        static constexpr char const *name() noexcept { return "IndexFlatL2"; }
    };

    struct IndexLshTraits {
        static constexpr bool supported() noexcept { return true; }

        static constexpr bool available() noexcept { return true; }

        static constexpr unsigned version() noexcept { return static_cast<int>(IndexType::IT_LSH); }

        static constexpr std::size_t alignment() noexcept { return 0; }

        static constexpr bool requires_alignment() noexcept { return false; }

        static constexpr char const *name() noexcept { return "IndexLSH"; }
    };
    struct IndexIvfFlatTraits {
        static constexpr bool supported() noexcept { return true; }

        static constexpr bool available() noexcept { return true; }

        static constexpr unsigned version() noexcept { return static_cast<int>(IndexType::IT_IVFFLAT); }

        static constexpr std::size_t alignment() noexcept { return 0; }

        static constexpr bool requires_alignment() noexcept { return false; }

        static constexpr char const *name() noexcept { return "IndexIVFFlat"; }
    };

    struct IndexDvPQTraits {
        static constexpr bool supported() noexcept { return true; }

        static constexpr bool available() noexcept { return true; }

        static constexpr unsigned version() noexcept { return static_cast<int>(IndexType::INDEX_VAMANA_DISK); }

        static constexpr std::size_t alignment() noexcept { return 0; }

        static constexpr bool requires_alignment() noexcept { return false; }

        static constexpr char const *name() noexcept { return "IndexDVPQ"; }
    };

    struct IndexVamanaTraits {
        static constexpr bool supported() noexcept { return true; }

        static constexpr bool available() noexcept { return true; }

        static constexpr unsigned version() noexcept { return static_cast<int>(IndexType::INDEX_VAMANA); }

        static constexpr std::size_t alignment() noexcept { return 0; }

        static constexpr bool requires_alignment() noexcept { return false; }

        static constexpr char const *name() noexcept { return "IndexVamana"; }
    };

}  // namespace polaris

namespace polaris {
    using all_rt_index = collie::engine_list<IndexVamanaTraits,IndexLshTraits, IndexFlatL2Traits, IndexFlatIPTraits, IndexFlatTraits>;

    using all_static_index = collie::engine_list<IndexDvPQTraits>;

    using all_model_index = collie::engine_list<IndexIvfFlatTraits>;

    using all_index = collie::engine_join<all_model_index , all_rt_index>;

    template<typename I>
    void print_index_name(std::vector<std::string> &names) {
        names.push_back(I::name());
    }

    template<typename ...I>
    struct list_indexes;

    template<typename ...I>
    struct list_indexes<collie::engine_list<I...>> {
        static void run(std::vector<std::string> &names) {
            std::initializer_list<int>{(print_index_name<I>(names), 0)...};
        }
    };

    std::vector<std::string> list_rt_indexes() {
        std::vector<std::string> index_types;
        list_indexes<all_rt_index>::run(index_types);
        return index_types;
    }
}

