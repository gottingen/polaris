//
// Copyright (C) 2015 Yahoo Japan Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//

#pragma once

#include <polaris/core/common.h>
#include <polaris/core/memory.h>
#include <polaris/storage/repository.h>
#include <polaris/distance/comparator.h>
#include <polaris/core/array_view.h>

class ObjectSpace;

namespace polaris {

    class ObjectDistances : public std::vector<ObjectDistance> {
    public:
        ObjectDistances(polaris::ObjectSpace *os = 0) {}

        void serialize(std::ofstream &os, ObjectSpace *objspace = 0) {
            polaris::Serializer::write(os, (std::vector<ObjectDistance> &) *this);
        }

        void deserialize(std::ifstream &is, ObjectSpace *objspace = 0) {
            polaris::Serializer::read(is, (std::vector<ObjectDistance> &) *this);
        }

        void serializeAsText(std::ofstream &os, ObjectSpace *objspace = 0) {
            polaris::Serializer::writeAsText(os, size());
            os << " ";
            for (size_t i = 0; i < size(); i++) {
                (*this)[i].serializeAsText(os);
                os << " ";
            }
        }

        void deserializeAsText(std::ifstream &is, ObjectSpace *objspace = 0) {
            size_t s;
            polaris::Serializer::readAsText(is, s);
            resize(s);
            for (size_t i = 0; i < size(); i++) {
                (*this)[i].deserializeAsText(is);
            }
        }

        void moveFrom(std::priority_queue<ObjectDistance, std::vector<ObjectDistance>, std::less<ObjectDistance>> &pq) {
            this->clear();
            this->resize(pq.size());
            for (int i = pq.size() - 1; i >= 0; i--) {
                (*this)[i] = pq.top();
                pq.pop();
            }
            assert(pq.size() == 0);
        }

        void moveFrom(std::priority_queue<ObjectDistance, std::vector<ObjectDistance>, std::less<ObjectDistance>> &pq,
                      double (&f)(double)) {
            this->clear();
            this->resize(pq.size());
            for (int i = pq.size() - 1; i >= 0; i--) {
                (*this)[i] = pq.top();
                (*this)[i].distance = f((*this)[i].distance);
                pq.pop();
            }
            assert(pq.size() == 0);
        }

        void moveFrom(std::priority_queue<ObjectDistance, std::vector<ObjectDistance>, std::less<ObjectDistance>> &pq,
                      unsigned int id) {
            this->clear();
            if (pq.size() == 0) {
                return;
            }
            this->resize(id == 0 ? pq.size() : pq.size() - 1);
            int i = this->size() - 1;
            while (pq.size() != 0 && i >= 0) {
                if (pq.top().id != id) {
                    (*this)[i] = pq.top();
                    i--;
                }
                pq.pop();
            }
            if (pq.size() != 0 && pq.top().id != id) {
                std::cerr << "moveFrom: Fatal error: somethig wrong! " << pq.size() << ":" << this->size() << ":" << id
                          << ":" << pq.top().id << std::endl;
                assert(pq.size() == 0 || pq.top().id == id);
            }
        }

    };

    typedef ObjectDistances GraphNode;
    typedef Object PersistentObject;

    class ObjectRepository;

    class ObjectSpace {
    public:
        typedef std::priority_queue<ObjectDistance, std::vector<ObjectDistance>, std::less<ObjectDistance> > ResultSet;

        ObjectSpace(size_t d) : dimension(d), distanceType(MetricType::METRIC_NONE), comparator(0), normalization(false),
                                prefetchOffset(-1), prefetchSize(-1) {}

        virtual ~ObjectSpace() { if (comparator != 0) { delete comparator; }}

        virtual size_t insert(Object *obj) = 0;

        DistanceComparator &getComparator() { return *comparator; }

        virtual void serialize(const std::string &of) = 0;

        virtual void deserialize(const std::string &ifile) = 0;

        virtual void serializeAsText(const std::string &of) = 0;

        virtual void deserializeAsText(const std::string &of) = 0;

        virtual void readText(std::istream &is, size_t dataSize) = 0;

        virtual void appendText(std::istream &is, size_t dataSize) = 0;

        virtual void append(const float *data, size_t dataSize) = 0;

        virtual void append(const double *data, size_t dataSize) = 0;

        virtual void append(const uint8_t *data, size_t dataSize) = 0;

        virtual void append(const float16 *data, size_t dataSize) = 0;

        virtual void copy(Object &objecta, Object &objectb) = 0;

        virtual void linearSearch(Object &query, double radius, size_t size,
                                  ObjectSpace::ResultSet &results) = 0;

        virtual const std::type_info &getObjectType() = 0;

        virtual void show(std::ostream &os, Object &object) = 0;

        virtual size_t getSize() = 0;

        virtual size_t getSizeOfElement() = 0;

        virtual size_t getByteSizeOfObject() = 0;

        virtual Object *allocateNormalizedObject(const std::string &textLine, const std::string &sep) = 0;

        virtual Object *allocateNormalizedObject(const std::vector<double> &obj) = 0;

        virtual Object *allocateNormalizedObject(const std::vector<float> &obj) = 0;

        virtual Object *allocateNormalizedObject(const std::vector<float16> &obj) = 0;

        virtual Object *allocateNormalizedObject(const std::vector<uint8_t> &obj) = 0;

        virtual Object *allocateNormalizedObject(const float *obj, size_t size) = 0;

        virtual PersistentObject *allocateNormalizedPersistentObject(const std::vector<double> &obj) = 0;

        virtual PersistentObject *allocateNormalizedPersistentObject(const std::vector<float> &obj) = 0;

        virtual void deleteObject(Object *po) = 0;

        virtual Object *allocateObject() = 0;

        virtual void remove(size_t id) = 0;

        virtual ObjectRepository &getRepository() = 0;

        virtual void set_metric_type(MetricType t) = 0;

        virtual void *getObject(size_t idx) = 0;

        virtual void getObject(size_t idx, std::vector<float> &v) = 0;

        virtual std::vector<float> getObject(Object &object) = 0;

        virtual void getObjects(const std::vector<size_t> &idxs, std::vector<std::vector<float>> &vs) = 0;

        virtual float computeMaxMagnitude(ObjectID beginId) = 0;

        virtual void setMagnitude(float maxMag, polaris::Repository<void> &graphNodes, ObjectID beginId) = 0;

        MetricType getDistanceType() { return distanceType; }

        size_t getDimension() { return dimension; }

        size_t getPaddedDimension() { return ((dimension - 1) / 16 + 1) * 16; }

        template<typename T>
        void normalize(T *data, size_t dim) {
            float sum = 0.0;
            for (size_t i = 0; i < dim; i++) {
                sum += static_cast<float>(data[i]) * static_cast<float>(data[i]);
            }
            if (sum == 0.0) {
                for (size_t i = 0; i < dim; i++) {
                    if (static_cast<float>(data[i]) != 0.0) {
                        std::stringstream msg;
                        msg
                                << "ObjectSpace::normalize: Error! the sum of the object is zero for the cosine similarity, but not a zero vector. "
                                << i << ":" << static_cast<float>(data[i]);
                        POLARIS_THROW_EX(msg);
                    }
                }
                std::stringstream msg;
                msg << "ObjectSpace::normalize: Error! the object is an invalid zero vector for the cosine similarity.";
                POLARIS_THROW_EX(msg);
            }
            sum = sqrt(sum);
            for (size_t i = 0; i < dim; i++) {
                data[i] = static_cast<float>(data[i]) / sum;
            }
        }

        int32_t getPrefetchOffset() { return prefetchOffset; }

        int32_t setPrefetchOffset(int offset) {
            if (offset > 0) {
                prefetchOffset = offset;
            }
            if (prefetchOffset <= 0) {
                prefetchOffset = floor(300.0 / (static_cast<float>(getPaddedDimension()) + 30.0) + 1.0);
            }
            return prefetchOffset;
        }

        int32_t getPrefetchSize() { return prefetchSize; }

        int32_t setPrefetchSize(int size) {
            if (size > 0) {
                prefetchSize = size;
            }
            if (prefetchSize <= 0) {
                prefetchSize = getByteSizeOfObject();
            }
            return prefetchSize;
        }

        bool isNormalizedDistance() {
            return (getDistanceType() == MetricType::METRIC_NORMALIZED_ANGLE) ||
                   (getDistanceType() == MetricType::METRIC_NORMALIZED_COSINE) ||
                   (getDistanceType() == MetricType::METRIC_NORMALIZED_L2);
        }

        polaris::distance_t compareWithL1(polaris::Object &o1, polaris::Object &o2);

    protected:
        const size_t dimension;
        MetricType distanceType;
        DistanceComparator *comparator;
        bool normalization;
        int32_t prefetchOffset;
        int32_t prefetchSize;
    };

    class BaseObject {
    public:
        virtual uint8_t &operator[](size_t idx) const = 0;

        virtual ArrayView get_view(size_t idx = 0) const {
            POLARIS_THROW_EX("Object: objectspace is null");
        }

        void serialize(std::ostream &os, ObjectSpace *objectspace = 0) {
            if (objectspace == 0) {
                POLARIS_THROW_EX("Object: objectspace is null");
            }
            size_t byteSize = objectspace->getByteSizeOfObject();
            polaris::Serializer::write(os, (uint8_t *) &(*this)[0], byteSize);
        }

        void deserialize(std::istream &is, ObjectSpace *objectspace = 0) {
            if (objectspace == 0) {
                POLARIS_THROW_EX("Object: objectspace is null");
            }
            size_t byteSize = objectspace->getByteSizeOfObject();
            assert(&(*this)[0] != 0);
            polaris::Serializer::read(is, (uint8_t *) &(*this)[0], byteSize);
            if (is.eof()) {
                std::stringstream msg;
                msg
                        << "ObjectSpace::BaseObject: Fatal Error! Read beyond the end of the object file. The object file is corrupted?"
                        << byteSize;
                POLARIS_THROW_EX(msg);
            }
        }

        void serializeAsText(std::ostream &os, ObjectSpace *objectspace = 0) {
            if (objectspace == 0) {
                POLARIS_THROW_EX("Object: objectspace is null");
            }
            const std::type_info &t = objectspace->getObjectType();
            size_t dimension = objectspace->getDimension();
            void *ref = (void *) &(*this)[0];
            if (t == typeid(uint8_t)) {
                polaris::Serializer::writeAsText(os, (uint8_t *) ref, dimension);
            } else if (t == typeid(float)) {
                polaris::Serializer::writeAsText(os, (float *) ref, dimension);
            } else if (t == typeid(float16)) {
                polaris::Serializer::writeAsText(os, (float16 *) ref, dimension);
            } else if (t == typeid(double)) {
                polaris::Serializer::writeAsText(os, (double *) ref, dimension);
            } else if (t == typeid(uint16_t)) {
                polaris::Serializer::writeAsText(os, (uint16_t *) ref, dimension);
            } else if (t == typeid(uint32_t)) {
                polaris::Serializer::writeAsText(os, (uint32_t *) ref, dimension);
            } else {
                std::cerr << "Object::serializeAsText: not supported data type. [" << t.name() << "]" << std::endl;
                assert(0);
            }
        }

        void deserializeAsText(std::ifstream &is, ObjectSpace *objectspace = 0) {
            if (objectspace == 0) {
                POLARIS_THROW_EX("Object: objectspace is null");
            }
            const std::type_info &t = objectspace->getObjectType();
            size_t dimension = objectspace->getDimension();
            void *ref = (void *) &(*this)[0];
            assert(ref != 0);
            if (t == typeid(uint8_t)) {
                polaris::Serializer::readAsText(is, (uint8_t *) ref, dimension);
            } else if (t == typeid(float)) {
                polaris::Serializer::readAsText(is, (float *) ref, dimension);
            } else if (t == typeid(float16)) {
                polaris::Serializer::readAsText(is, (float16 *) ref, dimension);
            } else if (t == typeid(double)) {
                polaris::Serializer::readAsText(is, (double *) ref, dimension);
            } else if (t == typeid(uint16_t)) {
                polaris::Serializer::readAsText(is, (uint16_t *) ref, dimension);
            } else if (t == typeid(uint32_t)) {
                polaris::Serializer::readAsText(is, (uint32_t *) ref, dimension);
            } else {
                std::cerr << "Object::deserializeAsText: not supported data type. [" << t.name() << "]" << std::endl;
                assert(0);
            }
        }

        template<typename T>
        void set(std::vector<T> &v, ObjectSpace &objectspace) {
            const std::type_info &t = objectspace.getObjectType();
            size_t dimension = objectspace.getDimension();
            void *ref = (void *) &(*this)[0];
            if (ref == 0) {
                POLARIS_THROW_EX("BaseObject::set: vector is null");
            }
            if (t == typeid(uint8_t)) {
                for (size_t d = 0; d < dimension; d++) {
                    *(static_cast<uint8_t *>(ref) + d) = v[d];
                }
            } else if (t == typeid(float)) {
                for (size_t d = 0; d < dimension; d++) {
                    *(static_cast<float *>(ref) + d) = v[d];
                }
            } else if (t == typeid(float16)) {
                for (size_t d = 0; d < dimension; d++) {
                    *(static_cast<float16 *>(ref) + d) = v[d];
                }
            } else if (t == typeid(double)) {
                for (size_t d = 0; d < dimension; d++) {
                    *(static_cast<double *>(ref) + d) = v[d];
                }
            } else if (t == typeid(uint16_t)) {
                for (size_t d = 0; d < dimension; d++) {
                    *(static_cast<uint16_t *>(ref) + d) = v[d];
                }
            } else if (t == typeid(uint32_t)) {
                for (size_t d = 0; d < dimension; d++) {
                    *(static_cast<uint32_t *>(ref) + d) = v[d];
                }
            } else {
                std::stringstream msg;
                msg << "BaseObject::set: not supported data type. [" << t.name() << "]";
                POLARIS_THROW_EX(msg);
            }
        }
    };

    class Object : public BaseObject {
    public:
        Object(polaris::ObjectSpace *os = 0) : vector(0) {
            if (os == 0) {
                return;
            }
            size_t s = os->getByteSizeOfObject();
            construct(s);
        }

        template<typename T>
        Object(std::vector<T> v, polaris::ObjectSpace &os):vector(0) {
            size_t s = os.getByteSizeOfObject();
            construct(s);
            set(v, os);
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

        ArrayView get_view(size_t idx = 0) const {
            ArrayView v;
            v.set_data(getPointer(idx));
            return v;
        }

        bool isEmpty() { return vector == 0; }

        static Object *allocate(ObjectSpace &objectspace) { return new Object(&objectspace); }

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

}

