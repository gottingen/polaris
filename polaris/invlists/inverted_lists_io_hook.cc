/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <polaris/invlists/inverted_lists_io_hook.h>

#include <polaris/internal/polaris_assert.h>
#include <polaris/internal/io.h>
#include <polaris/internal/io_macros.h>

#include <polaris/invlists/block_inverted_lists.h>

#ifndef _MSC_VER

#include <polaris/invlists/on_disk_inverted_lists.h>

#endif // !_MSC_VER

namespace polaris {

    /**********************************************************
     * InvertedListIOHook's
     **********************************************************/

    InvertedListsIOHook::InvertedListsIOHook(
            const std::string &key,
            const std::string &classname)
            : key(key), classname(classname) {}

    namespace {

/// std::vector that deletes its contents
        struct IOHookTable : std::vector<InvertedListsIOHook *> {
            IOHookTable() {
#ifndef _MSC_VER
                push_back(new OnDiskInvertedListsIOHook());
#endif
                push_back(new BlockInvertedListsIOHook());
            }

            ~IOHookTable() {
                for (auto x: *this) {
                    delete x;
                }
            }
        };

        static IOHookTable InvertedListsIOHook_table;

    } // namespace

    InvertedListsIOHook *InvertedListsIOHook::lookup(int h) {
        for (const auto &callback: InvertedListsIOHook_table) {
            if (h == fourcc(callback->key)) {
                return callback;
            }
        }
        POLARIS_THROW_FMT(
                "read_InvertedLists: could not load ArrayInvertedLists as "
                "%08x (\"%s\")",
                h,
                fourcc_inv_printable(h).c_str());
    }

    InvertedListsIOHook *InvertedListsIOHook::lookup_classname(
            const std::string &classname) {
        for (const auto &callback: InvertedListsIOHook_table) {
            if (callback->classname == classname) {
                return callback;
            }
        }
        POLARIS_THROW_FMT(
                "read_InvertedLists: could not find classname %s",
                classname.c_str());
    }

    void InvertedListsIOHook::add_callback(InvertedListsIOHook *cb) {
        InvertedListsIOHook_table.push_back(cb);
    }

    void InvertedListsIOHook::print_callbacks() {
        printf("registered %zd InvertedListsIOHooks:\n",
               InvertedListsIOHook_table.size());
        for (const auto &cb: InvertedListsIOHook_table) {
            printf("%08x %s %s\n",
                   fourcc(cb->key.c_str()),
                   cb->key.c_str(),
                   cb->classname.c_str());
        }
    }

    InvertedLists *InvertedListsIOHook::read_ArrayInvertedLists(
            IOReader *,
            int,
            size_t,
            size_t,
            const std::vector<size_t> &) const {
        POLARIS_THROW_FMT("read to array not implemented for %s", classname.c_str());
    }

} // namespace polaris
