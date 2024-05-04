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

#include <cstdint>
#include <cstddef>
#include <iostream>
#include <sstream>
#include <fstream>
#include <cstring>
#include <polaris/utility/polaris_exception.h>
#include <polaris/utility/serialize.h>

namespace polaris {

    template<class TYPE>
    class DynamicLengthVector {
    public:
        class iterator {
        public:
            iterator(void *ptr, uint32_t size) : i(static_cast<uint8_t *>(ptr)), elementSize(size) {}

            iterator operator+(size_t c) {
                return iterator(i + c * elementSize, elementSize);
            }

            iterator operator++(int c) {
                i += elementSize;
                return iterator(i, elementSize);
            }

            bool operator<(const iterator &it) const { return i < it.i; }

            TYPE &operator*() { return *reinterpret_cast<TYPE *>(i); }

            uint8_t *i;
            uint32_t elementSize;
        };

        DynamicLengthVector() : vector(0), vectorSize(0), allocatedSize(0), elementSize(0) {}

        ~DynamicLengthVector() { clear(); }

        void clear() {
            if (vector != 0) {
                delete[] vector;
            }
            vector = 0;
            vectorSize = 0;
            allocatedSize = 0;
        }

        TYPE &front() { return (*this).at(0); }

        TYPE &back() { return (*this).at(vectorSize - 1); }

        bool empty() { return vectorSize == 0; }

        iterator begin() {
            return iterator(vector, elementSize);
        }

        iterator end() {
            return begin() + vectorSize;
        }

        DynamicLengthVector &operator=(DynamicLengthVector<TYPE> &v) {
            std::cerr << "DynamicLengthVector cannot be copied." << std::endl;
            abort();
        }

        TYPE &at(size_t idx) {
            if (idx >= vectorSize) {
                std::stringstream msg;
                msg << "Vector: beyond the range. " << idx << ":" << vectorSize;
                POLARIS_THROW_EX(msg);
            }
            return *(begin() + idx);
        }

        TYPE &operator[](size_t idx) {
            return *(begin() + idx);
        }

        void copy(TYPE &dst, const TYPE &src) {
            std::memcpy(reinterpret_cast<unsigned char *>(&dst), reinterpret_cast<const unsigned char *>(&src), elementSize);
        }

        iterator erase(iterator b, iterator e) {
            iterator ret;
            e = end() < e ? end() : e;
            for (iterator i = b; i < e; i++) {
                ret = erase(i);
            }
            return ret;
        }

        iterator erase(iterator i) {
            iterator back = i;
            vectorSize--;
            iterator e = end();
            for (; i < e; i++) {
                copy(*i, *(i + 1));
            }
            return back;
        }

        void pop_back() {
            if (vectorSize > 0) {
                vectorSize--;
            }
        }

        iterator insert(iterator &i, const TYPE &data) {
            if (size() == 0) {
                push_back(data);
                return end();
            }
            off_t oft = i - begin();
            extend();
            i = begin() + oft;
            iterator b = begin();
            for (iterator ci = end(); ci > i && ci != b; ci--) {
                copy(*ci, *(ci - 1));
            }
            copy(*i, data);
            vectorSize++;
            return i + 1;
        }

        void push_back(const TYPE &data) {
            extend();
            vectorSize++;
            copy((*this).at(vectorSize - 1), data);
        }

        void reserve(size_t s) {
            if (s <= allocatedSize) {
                return;
            } else {
                uint8_t *newptr = new uint8_t[s * elementSize];
                uint8_t *dstptr = newptr;
                uint8_t *srcptr = vector;
                memcpy(dstptr, srcptr, vectorSize * elementSize);
                allocatedSize = s;
                if (vector != 0) {
                    delete[] vector;
                }
                vector = newptr;
            }
        }

        void resize(size_t s, TYPE v = TYPE()) {
            if (s > allocatedSize) {
                size_t asize = allocatedSize == 0 ? 1 : allocatedSize;
                while (asize < s) {
                    asize <<= 1;
                }
                reserve(asize);
                uint8_t *base = vector;
                for (size_t i = vectorSize; i < s; i++) {
                    copy(*reinterpret_cast<TYPE *>(base + i * elementSize), v);
                }
            }
            vectorSize = s;
        }

        void serializeAsText(std::ostream &os) {
            unsigned int s = size();
            os << s << " ";
            for (unsigned int i = 0; i < s; i++) {
                Serializer::writeAsText(os, (*this)[i]);
                os << " ";
            }
        }


        void deserializeAsText(std::istream &is) {
            clear();
            size_t s;
            Serializer::readAsText(is, s);
            resize(s);
            for (unsigned int i = 0; i < s; i++) {
                Serializer::readAsText(is, (*this)[i]);
            }
        }


        void serialize(std::ofstream &os) {
            uint32_t sz = size();
            polaris::Serializer::write(os, sz);
            os.write(reinterpret_cast<char *>(vector), size() * elementSize);
        }

        void deserialize(std::ifstream &is) {
            uint32_t sz;
            try {
                polaris::Serializer::read(is, sz);
            } catch (polaris::PolarisException &err) {
                std::stringstream msg;
                msg
                        << "DynamicLengthVector::deserialize: It might be caused by inconsistency of the valuable type of the vector. "
                        << err.what();
                POLARIS_THROW_EX(msg);
            }
            resize(sz);
            is.read(reinterpret_cast<char *>(vector), sz * elementSize);
        }

        size_t size() { return vectorSize; }

    public:
        void extend() {
            extend(vectorSize);
        }

        void extend(size_t idx) {
            if (idx >= allocatedSize) {
                uint64_t size = allocatedSize == 0 ? 1 : allocatedSize;
                do {
                    size <<= 1;
                } while (size <= idx);
                if (size > 0xffffffff) {
                    std::cerr << "Vector is too big. " << size << std::endl;
                    abort();
                }
                reserve(size);
            }
        }

        uint8_t *vector;
        uint32_t vectorSize;
        uint32_t allocatedSize;
        uint32_t elementSize;
    };
}  // namespace polaris
