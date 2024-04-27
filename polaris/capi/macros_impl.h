/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Copyright 2004-present Facebook. All Rights Reserved.
// -*- c++ -*-

/// Utility macros for the C wrapper implementation.

#ifndef MACROS_IMPL_H
#define MACROS_IMPL_H

#include <polaris/internal/polaris_exception.h>
#include <iostream>
#include <stdexcept>
#include "error_impl.h"
#include <polaris/capi/faiss_c.h>

#ifdef NDEBUG
#define CATCH_AND_HANDLE                                                      \
    catch (polaris::FaissException & e) {                                       \
        faiss_last_exception = std::make_exception_ptr(e);                    \
        return -2;                                                            \
    }                                                                         \
    catch (std::exception & e) {                                              \
        faiss_last_exception = std::make_exception_ptr(e);                    \
        return -4;                                                            \
    }                                                                         \
    catch (...) {                                                             \
        faiss_last_exception =                                                \
                std::make_exception_ptr(std::runtime_error("Unknown error")); \
        return -1;                                                            \
    }                                                                         \
    return 0;
#else
#define CATCH_AND_HANDLE                                                      \
    catch (polaris::FaissException & e) {                                       \
        std::cerr << e.what() << '\n';                                        \
        faiss_last_exception = std::make_exception_ptr(e);                    \
        return -2;                                                            \
    }                                                                         \
    catch (std::exception & e) {                                              \
        std::cerr << e.what() << '\n';                                        \
        faiss_last_exception = std::make_exception_ptr(e);                    \
        return -4;                                                            \
    }                                                                         \
    catch (...) {                                                             \
        std::cerr << "Unrecognized exception!\n";                             \
        faiss_last_exception =                                                \
                std::make_exception_ptr(std::runtime_error("Unknown error")); \
        return -1;                                                            \
    }                                                                         \
    return 0;
#endif

#define DEFINE_GETTER(clazz, ty, name)                             \
    ty faiss_##clazz##_##name(const Faiss##clazz* obj) {           \
        return static_cast<ty>(                                    \
                reinterpret_cast<const polaris::clazz*>(obj)->name); \
    }

#define DEFINE_GETTER_SUBCLASS(clazz, parent, ty, name)                    \
    ty faiss_##clazz##_##name(const Faiss##clazz* obj) {                   \
        return static_cast<ty>(                                            \
                reinterpret_cast<const polaris::parent::clazz*>(obj)->name); \
    }

#define DEFINE_GETTER_PERMISSIVE(clazz, ty, name)                      \
    ty faiss_##clazz##_##name(const Faiss##clazz* obj) {               \
        return (ty)(reinterpret_cast<const polaris::clazz*>(obj)->name); \
    }

#define DEFINE_GETTER_SUBCLASS_PERMISSIVE(clazz, parent, ty, name)             \
    ty faiss_##clazz##_##name(const Faiss##clazz* obj) {                       \
        return (ty)(reinterpret_cast<const polaris::parent::clazz*>(obj)->name); \
    }

#define DEFINE_SETTER(clazz, ty, name)                           \
    void faiss_##clazz##_set_##name(Faiss##clazz* obj, ty val) { \
        reinterpret_cast<polaris::clazz*>(obj)->name = val;        \
    }

#define DEFINE_SETTER_STATIC(clazz, ty_to, ty_from, name)                     \
    void faiss_##clazz##_set_##name(Faiss##clazz* obj, ty_from val) {         \
        reinterpret_cast<polaris::clazz*>(obj)->name = static_cast<ty_to>(val); \
    }

#define DEFINE_DESTRUCTOR(clazz)                     \
    void faiss_##clazz##_free(Faiss##clazz* obj) {   \
        delete reinterpret_cast<polaris::clazz*>(obj); \
    }

#define DEFINE_INDEX_DOWNCAST(clazz)                                        \
    Faiss##clazz* faiss_##clazz##_cast(FaissIndex* index) {                 \
        return reinterpret_cast<Faiss##clazz*>(dynamic_cast<polaris::clazz*>( \
                reinterpret_cast<polaris::Index*>(index)));                   \
    }

#define DEFINE_SEARCH_PARAMETERS_DOWNCAST(clazz)                            \
    Faiss##clazz* faiss_##clazz##_cast(FaissSearchParameters* sp) {         \
        return reinterpret_cast<Faiss##clazz*>(dynamic_cast<polaris::clazz*>( \
                reinterpret_cast<polaris::SearchParameters*>(sp)));           \
    }

#endif
