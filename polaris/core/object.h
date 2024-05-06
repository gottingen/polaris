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

#include <polaris/core/common.h>
#include <polaris/utility/serialize.h>
#include <polaris/utility/polaris_assert.h>
#include <polaris/core/memory.h>

namespace polaris {

    class BaseObject {
    public:
        virtual uint8_t &operator[](size_t idx) const = 0;

        void serialize(std::ostream &os, const SpaceMeta &meta) {
            polaris::Serializer::write(os, (uint8_t *) &(*this)[0], meta.byte_size);
        }

        void deserialize(std::istream &is, const SpaceMeta &meta) {
            assert(&(*this)[0] != 0);
            polaris::Serializer::read(is, (uint8_t *) &(*this)[0], meta.byte_size);
            if (is.eof()) {
                std::stringstream msg;
                msg
                        << "ObjectSpace::BaseObject: Fatal Error! Read beyond the end of the object file. The object file is corrupted?"
                        << meta.byte_size;
                POLARIS_THROW_EX(msg);
            }
        }

        void serializeAsText(std::ostream &os, const SpaceMeta &meta) {
            auto t = meta.object_type;
            size_t dimension = meta.dimension;
            void *ref = (void *) &(*this)[0];
            switch (t) {
                case ObjectType::UINT8:
                    polaris::Serializer::writeAsText(os, (uint8_t *) ref, dimension);
                    break;
                case ObjectType::INT8:
                    polaris::Serializer::writeAsText(os, (int8_t *) ref, dimension);
                    break;
                case ObjectType::UINT16:
                    polaris::Serializer::writeAsText(os, (uint16_t *) ref, dimension);
                    break;
                case ObjectType::INT16:
                    polaris::Serializer::writeAsText(os, (int16_t *) ref, dimension);
                    break;
                case ObjectType::UINT32:
                    polaris::Serializer::writeAsText(os, (uint32_t *) ref, dimension);
                    break;
                case ObjectType::INT32:
                    polaris::Serializer::writeAsText(os, (int32_t *) ref, dimension);
                    break;
                case ObjectType::UINT64:
                    polaris::Serializer::writeAsText(os, (uint64_t *) ref, dimension);
                    break;
                case ObjectType::INT64:
                    polaris::Serializer::writeAsText(os, (int64_t *) ref, dimension);
                    break;

                case ObjectType::FLOAT:
                    polaris::Serializer::writeAsText(os, (float *) ref, dimension);
                    break;
                case ObjectType::FLOAT16:
                    polaris::Serializer::writeAsText(os, (float16 *) ref, dimension);
                    break;
                case ObjectType::DOUBLE:
                    polaris::Serializer::writeAsText(os, (double *) ref, dimension);
                    break;
                case ObjectType::BFLOAT16:
                    POLARIS_ASSERT_MSG(false, "Bfloat16 is not supported in serializeAsText");
                    break;
                default:
                    POLARIS_ASSERT_MSG(false, "Unknown object type");

            }
        }

        void deserializeAsText(std::ifstream &is, const SpaceMeta &meta) {
            auto t = meta.object_type;
            size_t dimension = meta.dimension;
            void *ref = (void *) &(*this)[0];
            assert(ref != 0);
            switch (t) {
                case ObjectType::UINT8:
                    polaris::Serializer::readAsText(is, (uint8_t *) ref, dimension);
                    break;
                case ObjectType::INT8:
                    polaris::Serializer::readAsText(is, (int8_t *) ref, dimension);
                    break;
                case ObjectType::UINT16:
                    polaris::Serializer::readAsText(is, (uint16_t *) ref, dimension);
                    break;
                case ObjectType::INT16:
                    polaris::Serializer::readAsText(is, (int16_t *) ref, dimension);
                    break;
                case ObjectType::UINT32:
                    polaris::Serializer::readAsText(is, (uint32_t *) ref, dimension);
                    break;
                case ObjectType::INT32:
                    polaris::Serializer::readAsText(is, (int32_t *) ref, dimension);
                    break;
                case ObjectType::UINT64:
                    polaris::Serializer::readAsText(is, (uint64_t *) ref, dimension);
                    break;
                case ObjectType::INT64:
                    polaris::Serializer::readAsText(is, (int64_t *) ref, dimension);
                    break;
                case ObjectType::FLOAT:
                    polaris::Serializer::readAsText(is, (float *) ref, dimension);
                    break;
                case ObjectType::FLOAT16:
                    polaris::Serializer::readAsText(is, (float16 *) ref, dimension);
                    break;
                case ObjectType::DOUBLE:
                    polaris::Serializer::readAsText(is, (double *) ref, dimension);
                    break;
                case ObjectType::BFLOAT16:
                    POLARIS_ASSERT_MSG(false, "Bfloat16 is not supported in deserializeAsText");
                    break;
                default:
                    POLARIS_ASSERT_MSG(false, "Unknown object type");
            }
        }

        template<typename T>
        void set(std::vector<T> &v, const SpaceMeta &meta) {
            static_assert(std::is_pod_v<T>, "T must be a POD type");
            auto t = meta.object_type;
            size_t dimension = meta.dimension;
            void *ref = (void *) &(*this)[0];
            if (ref == 0) {
                POLARIS_THROW_EX("BaseObject::set: vector is null");
            }
            auto bytesize = meta.byte_size;
            if (bytesize != v.size() * sizeof(T) || dimension != v.size()) {
                std::stringstream msg;
                msg << "BaseObject::set: size mismatch. [" << bytesize << "!=" << v.size() * sizeof(T) << "]";
                POLARIS_THROW_EX(msg);
            }
            std::memcpy(ref, v.data(), bytesize);
        }
    };

    class Object : public BaseObject {
    public:
        Object(const SpaceMeta *meta = nullptr) : vector(0) {
            if (!meta) {
                return;
            }
            size_t s = meta->byte_size;
            construct(s);
        }

        template<typename T>
        Object(std::vector<T> v, const SpaceMeta &meta):vector(0) {
            size_t s = meta.byte_size;
            construct(s);
            set(v, meta);
        }

        Object(size_t s) : vector(0) {
            assert(s != 0);
            construct(s);
        }

        virtual ~Object() { clear(); }

        void attach(void *ptr) { vector = static_cast<uint8_t *>(ptr); }

        void detach() { vector = 0; }

        void copy(Object &o, size_t s) {
            assert(vector != 0);
            for (size_t i = 0; i < s; i++) {
                vector[i] = o[i];
            }
        }

        uint8_t &operator[](size_t idx) const { return vector[idx]; }

        void *getPointer(size_t idx = 0) const { return vector + idx; }

        bool isEmpty() { return vector == 0; }

        static Object *allocate(const SpaceMeta &meta) { return new Object(&meta); }

    private:
        void clear() {
            if (vector != 0) {
                MemoryCache::alignedFree(vector);
            }
            vector = 0;
        }

        void construct(size_t s) {
            assert(vector == 0);
            size_t allocsize = ((s - 1) / 64 + 1) * 64;
            vector = static_cast<uint8_t *>(MemoryCache::alignedAlloc(allocsize));
            memset(vector, 0, allocsize);
        }

        uint8_t *vector;
    };
}  // namespace polaris

