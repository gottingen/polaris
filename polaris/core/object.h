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
#include <collie/utility/status.h>

namespace polaris {

    class BaseObject {
    public:
        virtual uint8_t &operator[](size_t idx) const = 0;

        void serialize(std::ostream &os, size_t byteSize) {
            polaris::Serializer::write(os, (uint8_t *) &(*this)[0], byteSize);
        }

        collie::Status deserialize(std::istream &is, size_t byteSize) {
            if(&(*this)[0] == nullptr) {
                return collie::Status::invalid_argument("BaseObject::deserialize: object is null");
            }
            polaris::Serializer::read(is, (uint8_t *) &(*this)[0], byteSize);
            if (is.eof()) {
                return collie::Status::io_error("ObjectSpace::BaseObject: Fatal Error! Read beyond the end of the object file. The object file is corrupted?");
            }
        }

        collie::Status serializeAsText(std::ostream &os, size_t dimension, ObjectType type) {

            void *ref = (void *) &(*this)[0];
            switch (type) {
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
                case ObjectType::DOUBLE:
                    polaris::Serializer::writeAsText(os, (double *) ref, dimension);
                    break;
                case ObjectType::FLOAT16:
                    polaris::Serializer::writeAsText(os, (float16 *) ref, dimension);
                    break;
                default:
                    return collie::Status::invalid_argument("BaseObject::serializeAsText: not supported data type");
            }
            return collie::Status::ok_status();
        }

        collie::Status deserializeAsText(std::ifstream &is, size_t dimension, ObjectType type) {
            void *ref = (void *) &(*this)[0];
            if(ref == nullptr) {
                return collie::Status::invalid_argument("BaseObject::deserializeAsText: object is null");
            }
            switch (type) {
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
                case ObjectType::DOUBLE:
                    polaris::Serializer::readAsText(is, (double *) ref, dimension);
                    break;
                case ObjectType::FLOAT16:
                    polaris::Serializer::readAsText(is, (float16 *) ref, dimension);
                    break;
                default:
                    return collie::Status::invalid_argument("BaseObject::deserializeAsText: not supported data type");
            }
            return collie::Status::ok_status();
        }

        template<typename T>
        collie::Status set(std::vector<T> &v, size_t dimension, ObjectType type) {
            void *ref = (void *) &(*this)[0];
            if (ref == 0) {
                return collie::Status::invalid_argument("BaseObject::set: vector is null");
            }
            switch (type) {
                case ObjectType::UINT8:
                    for (size_t d = 0; d < dimension; d++) {
                        *(static_cast<uint8_t *>(ref) + d) = v[d];
                    }
                    break;
                case ObjectType::INT8:
                    for (size_t d = 0; d < dimension; d++) {
                        *(static_cast<int8_t *>(ref) + d) = v[d];
                    }
                    break;
                case ObjectType::UINT16:
                    for (size_t d = 0; d < dimension; d++) {
                        *(static_cast<uint16_t *>(ref) + d) = v[d];
                    }
                    break;
                case ObjectType::INT16:
                    for (size_t d = 0; d < dimension; d++) {
                        *(static_cast<int16_t *>(ref) + d) = v[d];
                    }
                    break;
                case ObjectType::UINT32:
                    for (size_t d = 0; d < dimension; d++) {
                        *(static_cast<uint32_t *>(ref) + d) = v[d];
                    }
                    break;
                case ObjectType::INT32:
                    for (size_t d = 0; d < dimension; d++) {
                        *(static_cast<int32_t *>(ref) + d) = v[d];
                    }
                    break;
                case ObjectType::UINT64:
                    for (size_t d = 0; d < dimension; d++) {
                        *(static_cast<uint64_t *>(ref) + d) = v[d];
                    }
                    break;
                case ObjectType::INT64:
                    for (size_t d = 0; d < dimension; d++) {
                        *(static_cast<int64_t *>(ref) + d) = v[d];
                    }
                    break;
                case ObjectType::FLOAT:
                    for (size_t d = 0; d < dimension; d++) {
                        *(static_cast<float *>(ref) + d) = v[d];
                    }
                    break;
                case ObjectType::DOUBLE:
                    for (size_t d = 0; d < dimension; d++) {
                        *(static_cast<double *>(ref) + d) = v[d];
                    }
                    break;
                case ObjectType::FLOAT16:
                    for (size_t d = 0; d < dimension; d++) {
                        *(static_cast<float16 *>(ref) + d) = v[d];
                    }
                    break;
                default:
                    return collie::Status::invalid_argument("BaseObject::set: not supported data type");
            }
            return collie::Status::ok_status();
        }
    };

    class ObjectView {
    public:

        ObjectView() = default;
        explicit ObjectView(void *data) : _data(data) {}

        [[nodiscard]] const void *data() const { return _data; }

        [[nodiscard]] float l2_norm() const { return _l2_norm; }

        [[nodiscard]] float l2_norm_sq() const { return _l2_norm_sq; }

        ObjectView &set_data(const void *data) {
            _data = data;
            return *this;
        }

        ObjectView &set_l2_norm(float l2_norm) {
            _l2_norm = l2_norm;
            return *this;
        }

        ObjectView &set_l2_norm_sq(float l2_norm_sq) {
            _l2_norm_sq = l2_norm_sq;
            return *this;
        }

        explicit operator bool() const { return _data != nullptr; }


    private:
        const void *_data{nullptr};
        float _l2_norm{0.0f};
        float _l2_norm_sq{0.0f};
    };

    class Object : public BaseObject {
    public:
        Object(size_t bytes_size) : vector(nullptr) {
            construct(bytes_size);
        }

        template<typename T>
        Object(std::vector<T> v, size_t bytes_size, ObjectType type):vector(nullptr) {
            construct(bytes_size);
            set(v, bytes_size, type);
        }

        virtual ~Object() { clear(); }


        void copy(Object &o, size_t s) {
            assert(vector != 0);
            for (size_t i = 0; i < s; i++) {
                vector[i] = o[i];
            }
        }

        uint8_t &operator[](size_t idx) const { return vector[idx]; }

        void *data(size_t idx = 0) const { return vector + idx; }

        ObjectView get_view(size_t idx = 0) const {
            return ObjectView(data(idx));
        }

        bool is_empty() { return vector == nullptr; }

        static Object *allocate(size_t bytes_size) { return new Object(bytes_size); }

    private:
        void clear() {
            if (vector != nullptr) {
                MemoryCache::alignedFree(vector);
            }
            vector = nullptr;
        }

        void construct(size_t s) {
            assert(vector == nullptr);
            size_t allocsize = ((s - 1) / 64 + 1) * 64;
            vector = static_cast<uint8_t *>(MemoryCache::alignedAlloc(allocsize));
            memset(vector, 0, allocsize);
        }

        uint8_t *vector;
    };

}  // namespace polaris

