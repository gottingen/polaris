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


#ifdef _OPENMP

#include <omp.h>

#else
#warning "*** OMP is *NOT* available! ***"
#endif

#include <polaris/utility/common.h>
#include <polaris/graph/ngt/object_space.h>
#include <polaris/graph/ngt/object_repository.h>
#include <polaris/distance/primitive_comparator.h>

class ObjectSpace;

namespace polaris {

    template<typename OBJECT_TYPE, typename COMPARE_TYPE>
    class ObjectSpaceRepository : public ObjectSpace, public ObjectRepository {
    public:

        class ComparatorL1 : public Comparator {
        public:
            ComparatorL1(size_t d) : Comparator(d) {}

            double operator()(Object &objecta, Object &objectb) {
                return PrimitiveComparator::compareL1((OBJECT_TYPE *) &objecta[0], (OBJECT_TYPE *) &objectb[0],
                                                      dimension);
            }
        };

        class ComparatorL2 : public Comparator {
        public:
            ComparatorL2(size_t d) : Comparator(d) {}

            double operator()(Object &objecta, Object &objectb) {
                return PrimitiveComparator::compareL2((OBJECT_TYPE *) &objecta[0], (OBJECT_TYPE *) &objectb[0],
                                                      dimension);
            }
        };

        class ComparatorNormalizedL2 : public Comparator {
        public:
            ComparatorNormalizedL2(size_t d) : Comparator(d) {}

            double operator()(Object &objecta, Object &objectb) {
                return PrimitiveComparator::compareNormalizedL2((OBJECT_TYPE *) &objecta[0],
                                                                (OBJECT_TYPE *) &objectb[0], dimension);
            }
        };

        class ComparatorHammingDistance : public Comparator {
        public:
            ComparatorHammingDistance(size_t d) : Comparator(d) {}

            double operator()(Object &objecta, Object &objectb) {
                return PrimitiveComparator::compareHammingDistance((OBJECT_TYPE *) &objecta[0],
                                                                   (OBJECT_TYPE *) &objectb[0], dimension);
            }
        };

        class ComparatorJaccardDistance : public Comparator {
        public:
            ComparatorJaccardDistance(size_t d) : Comparator(d) {}

            double operator()(Object &objecta, Object &objectb) {
                return PrimitiveComparator::compareJaccardDistance((OBJECT_TYPE *) &objecta[0],
                                                                   (OBJECT_TYPE *) &objectb[0], dimension);
            }
        };

        class ComparatorSparseJaccardDistance : public Comparator {
        public:
            ComparatorSparseJaccardDistance(size_t d) : Comparator(d) {}

            double operator()(Object &objecta, Object &objectb) {
                return PrimitiveComparator::compareSparseJaccardDistance((OBJECT_TYPE *) &objecta[0],
                                                                         (OBJECT_TYPE *) &objectb[0], dimension);
            }
        };

        class ComparatorAngleDistance : public Comparator {
        public:
            ComparatorAngleDistance(size_t d) : Comparator(d) {}

            double operator()(Object &objecta, Object &objectb) {
                return PrimitiveComparator::compareAngleDistance((OBJECT_TYPE *) &objecta[0],
                                                                 (OBJECT_TYPE *) &objectb[0], dimension);
            }
        };

        class ComparatorNormalizedAngleDistance : public Comparator {
        public:
            ComparatorNormalizedAngleDistance(size_t d) : Comparator(d) {}

            double operator()(Object &objecta, Object &objectb) {
                return PrimitiveComparator::compareNormalizedAngleDistance((OBJECT_TYPE *) &objecta[0],
                                                                           (OBJECT_TYPE *) &objectb[0], dimension);
            }
        };

        class ComparatorCosineSimilarity : public Comparator {
        public:
            ComparatorCosineSimilarity(size_t d) : Comparator(d) {}

            double operator()(Object &objecta, Object &objectb) {
                return PrimitiveComparator::compareCosineSimilarity((OBJECT_TYPE *) &objecta[0],
                                                                    (OBJECT_TYPE *) &objectb[0], dimension);
            }
        };

        class ComparatorNormalizedCosineSimilarity : public Comparator {
        public:
            ComparatorNormalizedCosineSimilarity(size_t d) : Comparator(d) {}

            double operator()(Object &objecta, Object &objectb) {
                return PrimitiveComparator::compareNormalizedCosineSimilarity((OBJECT_TYPE *) &objecta[0],
                                                                              (OBJECT_TYPE *) &objectb[0], dimension);
            }
        };

        class ComparatorPoincareDistance : public Comparator {  // added by Nyapicom
        public:
            ComparatorPoincareDistance(size_t d) : Comparator(d) {}

            double operator()(Object &objecta, Object &objectb) {
                return PrimitiveComparator::comparePoincareDistance((OBJECT_TYPE *) &objecta[0],
                                                                    (OBJECT_TYPE *) &objectb[0], dimension);
            }
        };

        class ComparatorLorentzDistance : public Comparator {  // added by Nyapicom
        public:
            ComparatorLorentzDistance(size_t d) : Comparator(d) {}

            double operator()(Object &objecta, Object &objectb) {
                return PrimitiveComparator::compareLorentzDistance((OBJECT_TYPE *) &objecta[0],
                                                                   (OBJECT_TYPE *) &objectb[0], dimension);
            }
        };

        class ComparatorInnerProduct : public Comparator {
        public:
            ComparatorInnerProduct(size_t d) : Comparator(d) {}

            double operator()(Object &objecta, Object &objectb) {
                return PrimitiveComparator::compareDotProduct((OBJECT_TYPE *) &objecta[0], (OBJECT_TYPE *) &objectb[0],
                                                              dimension);
            }
        };

        ObjectSpaceRepository(size_t d, const std::type_info &ot, DistanceType t) : ObjectSpace(d),
                                                                                    ObjectRepository(d, ot) {
            size_t objectSize = 0;
            if (ot == typeid(uint8_t)) {
                objectSize = sizeof(uint8_t);
            } else if (ot == typeid(float)) {
                objectSize = sizeof(float);
            } else if (ot == typeid(float16)) {
                objectSize = sizeof(float16);
            } else if (ot == typeid(bfloat16)) {
                objectSize = sizeof(bfloat16);
            } else {
                std::stringstream msg;
                msg << "ObjectSpace::constructor: Not supported type. " << ot.name();
                POLARIS_THROW_EX(msg);
            }
            setLength(objectSize * d);
            setPaddedLength(objectSize * ObjectSpace::getPaddedDimension());
            setDistanceType(t);
        }


        void copy(Object &objecta, Object &objectb) {
            objecta.copy(objectb, getByteSizeOfObject());
        }

        void setDistanceType(DistanceType t) {
            if (comparator != 0) {
                delete comparator;
            }
            assert(ObjectSpace::dimension != 0);
            distanceType = t;
            switch (distanceType) {
                case DistanceTypeL1:
                    comparator = new ObjectSpaceRepository::ComparatorL1(ObjectSpace::getPaddedDimension());
                    break;
                case DistanceTypeL2:
                    comparator = new ObjectSpaceRepository::ComparatorL2(ObjectSpace::getPaddedDimension());
                    break;
                case DistanceTypeNormalizedL2:
                    comparator = new ObjectSpaceRepository::ComparatorNormalizedL2(ObjectSpace::getPaddedDimension());
                    normalization = true;
                    break;
                case DistanceTypeHamming:
                    comparator = new ObjectSpaceRepository::ComparatorHammingDistance(
                            ObjectSpace::getPaddedDimension());
                    break;
                case DistanceTypeJaccard:
                    comparator = new ObjectSpaceRepository::ComparatorJaccardDistance(
                            ObjectSpace::getPaddedDimension());
                    break;
                case DistanceTypeSparseJaccard:
                    comparator = new ObjectSpaceRepository::ComparatorSparseJaccardDistance(
                            ObjectSpace::getPaddedDimension());
                    setSparse();
                    break;
                case DistanceTypeAngle:
                    comparator = new ObjectSpaceRepository::ComparatorAngleDistance(ObjectSpace::getPaddedDimension());
                    break;
                case DistanceTypeCosine:
                    comparator = new ObjectSpaceRepository::ComparatorCosineSimilarity(
                            ObjectSpace::getPaddedDimension());
                    break;
                case DistanceTypePoincare:  // added by Nyapicom
                    comparator = new ObjectSpaceRepository::ComparatorPoincareDistance(
                            ObjectSpace::getPaddedDimension());
                    break;
                case DistanceTypeLorentz:  // added by Nyapicom
                    comparator = new ObjectSpaceRepository::ComparatorLorentzDistance(
                            ObjectSpace::getPaddedDimension());
                    break;
                case DistanceTypeNormalizedAngle:
                    comparator = new ObjectSpaceRepository::ComparatorNormalizedAngleDistance(
                            ObjectSpace::getPaddedDimension());
                    normalization = true;
                    break;
                case DistanceTypeNormalizedCosine:
                    comparator = new ObjectSpaceRepository::ComparatorNormalizedCosineSimilarity(
                            ObjectSpace::getPaddedDimension());
                    normalization = true;
                    break;
                case DistanceTypeInnerProduct:
                    comparator = new ObjectSpaceRepository::ComparatorL2(ObjectSpace::getPaddedDimension());
                    setInnerProduct();
                    break;
                default:
                    std::stringstream msg;
                    msg << "polaris::ObjectSpaceRepository: The distance type is invalid. " << distanceType;
                    POLARIS_THROW_EX(msg);
            }
        }


        void serialize(const std::string &ofile) { ObjectRepository::serialize(ofile, this); }

        void deserialize(const std::string &ifile) { ObjectRepository::deserialize(ifile, this); }

        void serializeAsText(const std::string &ofile) { ObjectRepository::serializeAsText(ofile, this); }

        void deserializeAsText(const std::string &ifile) { ObjectRepository::deserializeAsText(ifile, this); }

        void readText(std::istream &is, size_t dataSize) { ObjectRepository::readText(is, dataSize); }

        void appendText(std::istream &is, size_t dataSize) { ObjectRepository::appendText(is, dataSize); }

        void append(const float *data, size_t dataSize) { ObjectRepository::append(data, dataSize); }

        void append(const double *data, size_t dataSize) { ObjectRepository::append(data, dataSize); }

        void append(const uint8_t *data, size_t dataSize) { ObjectRepository::append(data, dataSize); }


        void append(const float16 *data, size_t dataSize) { ObjectRepository::append(data, dataSize); }


        size_t insert(Object *obj) { return ObjectRepository::insert(obj); }

        void remove(size_t id) { ObjectRepository::remove(id); }

        void linearSearch(Object &query, double radius, size_t size, ObjectSpace::ResultSet &results) {
            if (!results.empty()) {
                POLARIS_THROW_EX("lenearSearch: results is not empty");
            }
            size_t byteSizeOfObject = getByteSizeOfObject();
            const size_t prefetchOffset = getPrefetchOffset();
            ObjectRepository &rep = *this;
            for (size_t idx = 0; idx < rep.size(); idx++) {
                if (idx + prefetchOffset < rep.size() && rep[idx + prefetchOffset] != 0) {
                    MemoryCache::prefetch(
                            (unsigned char *) &(*static_cast<PersistentObject *>(rep[idx + prefetchOffset]))[0],
                            byteSizeOfObject);
                }
                if (rep[idx] == 0) {
                    continue;
                }
                Distance d = (*comparator)((Object &) query, (Object &) *rep[idx]);
                if (radius < 0.0 || d <= radius) {
                    polaris::ObjectDistance obj(idx, d);
                    results.push(obj);
                    if (results.size() > size) {
                        results.pop();
                    }
                }
            }
            return;
        }

        float computeMaxMagnitude(polaris::ObjectID beginID = 1) {
            float maxMag = 0.0;
            ObjectRepository &rep = *this;
            auto nOfThreads = omp_get_max_threads();
            std::vector<float> maxm(nOfThreads, 0.0);
#pragma omp parallel for
            for (size_t idx = beginID; idx < rep.size(); idx++) {
                if (rep[idx] == 0) {
                    continue;
                }
                auto thdID = omp_get_thread_num();
                auto object = getObject(*rep[idx]);
                double mag = 0.0;
                for (size_t i = 0; i < object.size() - 1; i++) {
                    mag += object[i] * object[i];
                }
                if (mag > maxm[thdID]) {
                    maxm[thdID] = mag;
                }
            }
            for (int ti = 0; ti < nOfThreads; ti++) {
                if (maxm[ti] > maxMag) {
                    maxMag = maxm[ti];
                }
            }
            return maxMag;
        }

        void setMagnitude(float maxMag, polaris::Repository<void> &graphNodes, polaris::ObjectID beginID = 1) {
            ObjectRepository &rep = *this;
#pragma omp parallel for
            for (size_t idx = beginID; idx < rep.size(); idx++) {
                if (rep[idx] == 0) {
                    continue;
                }
                if (idx < graphNodes.size() && graphNodes[idx] != 0) {
                    continue;
                }

                auto object = getObject(*rep[idx]);

                double mag = 0.0;
                for (size_t i = 0; i < object.size() - 1; i++) {
                    mag += object[i] * object[i];
                }
                auto v = maxMag - static_cast<float>(mag);
                if (v < 0.0) {
                    std::cerr << "Warning! magnitude is larger than the current max magnitude. " << idx << ":" << v
                              << ":" << maxMag << ":" << static_cast<float>(mag) << std::endl;
                    v = 0.0;
                }
                object.back() = sqrt(v);
                setObject(*rep[idx], object);
            }
        }


        void *getObject(size_t idx) {
            if (isEmpty(idx)) {
                std::stringstream msg;
                msg
                        << "polaris::ObjectSpaceRepository: The specified ID is out of the range. The object ID should be greater than zero. "
                        << idx << ":" << ObjectRepository::size() << ".";
                POLARIS_THROW_EX(msg);
            }
            PersistentObject &obj = *(*this)[idx];
            return reinterpret_cast<OBJECT_TYPE *>(&obj[0]);
        }

        void getObject(size_t idx, std::vector<float> &v) {
            OBJECT_TYPE *obj = static_cast<OBJECT_TYPE *>(getObject(idx));
            size_t dim = getDimension();
            v.resize(dim);
            for (size_t i = 0; i < dim; i++) {
                v[i] = static_cast<float>(obj[i]);
            }
        }

        std::vector<float> getObject(Object &object) {
            std::vector<float> v;
            OBJECT_TYPE *obj = static_cast<OBJECT_TYPE *>(object.getPointer());
            size_t dim = getDimension();
            v.resize(dim);
            for (size_t i = 0; i < dim; i++) {
                v[i] = static_cast<float>(obj[i]);
            }
            return v;
        }

        void getObjects(const std::vector<size_t> &idxs, std::vector<std::vector<float>> &vs) {
            vs.resize(idxs.size());
            auto v = vs.begin();
            for (auto idx = idxs.begin(); idx != idxs.end(); idx++, v++) {
                getObject(*idx, *v);
            }
        }

        void normalize(Object &object) {
            OBJECT_TYPE *obj = (OBJECT_TYPE *) &object[0];
            ObjectSpace::normalize(obj, ObjectSpace::dimension);
        }

        Object *allocateObject() { return ObjectRepository::allocateObject(); }

        void deleteObject(Object *po) { ObjectRepository::deleteObject(po); }

        Object *allocateNormalizedObject(const std::string &textLine, const std::string &sep) {
            Object *allocatedObject = ObjectRepository::allocateObject(textLine, sep);
            if (normalization) {
                normalize(*allocatedObject);
            }
            return allocatedObject;
        }

        Object *allocateNormalizedObject(const std::vector<double> &obj) {
            Object *allocatedObject = ObjectRepository::allocateObject(obj);
            if (normalization) {
                normalize(*allocatedObject);
            }
            return allocatedObject;
        }

        Object *allocateNormalizedObject(const std::vector<float> &obj) {
            Object *allocatedObject = ObjectRepository::allocateObject(obj);
            if (normalization) {
                normalize(*allocatedObject);
            }
            return allocatedObject;
        }


        Object *allocateNormalizedObject(const std::vector<float16> &obj) {
            Object *allocatedObject = ObjectRepository::allocateObject(obj);
            if (normalization) {
                normalize(*allocatedObject);
            }
            return allocatedObject;
        }


        Object *allocateNormalizedObject(const std::vector<uint8_t> &obj) {
            Object *allocatedObject = ObjectRepository::allocateObject(obj);
            if (normalization) {
                normalize(*allocatedObject);
            }
            return allocatedObject;
        }

        Object *allocateNormalizedObject(const float *obj, size_t size) {
            Object *allocatedObject = ObjectRepository::allocateObject(obj, size);
            if (normalization) {
                normalize(*allocatedObject);
            }
            return allocatedObject;
        }

        PersistentObject *allocateNormalizedPersistentObject(const std::vector<double> &obj) {
            PersistentObject *allocatedObject = ObjectRepository::allocatePersistentObject(obj);
            if (normalization) {
                normalize(*allocatedObject);
            }
            return allocatedObject;
        }

        PersistentObject *allocateNormalizedPersistentObject(const std::vector<float> &obj) {
            PersistentObject *allocatedObject = ObjectRepository::allocatePersistentObject(obj);
            if (normalization) {
                normalize(*allocatedObject);
            }
            return allocatedObject;
        }

        PersistentObject *allocateNormalizedPersistentObject(const std::vector<float16> &obj) {
            PersistentObject *allocatedObject = ObjectRepository::allocatePersistentObject(obj);
            if (normalization) {
                normalize(*allocatedObject);
            }
            return allocatedObject;
        }


        PersistentObject *allocateNormalizedPersistentObject(const std::vector<uint8_t> &obj) {
            PersistentObject *allocatedObject = ObjectRepository::allocatePersistentObject(obj);
            if (normalization) {
                normalize(*allocatedObject);
            }
            return allocatedObject;
        }

        size_t getSize() { return ObjectRepository::size(); }

        size_t getSizeOfElement() { return sizeof(OBJECT_TYPE); }

        const std::type_info &getObjectType() { return typeid(OBJECT_TYPE); };

        size_t getByteSizeOfObject() { return getByteSize(); }

        ObjectRepository &getRepository() { return *this; };

        void show(std::ostream &os, Object &object) {
            const std::type_info &t = getObjectType();
            if (t == typeid(uint8_t)) {
                unsigned char *optr = static_cast<unsigned char *>(&object[0]);
                for (size_t i = 0; i < getDimension(); i++) {
                    os << (int) optr[i] << " ";
                }
            } else if (t == typeid(float)) {
                float *optr = reinterpret_cast<float *>(&object[0]);
                for (size_t i = 0; i < getDimension(); i++) {
                    os << optr[i] << " ";
                }
            } else if (t == typeid(float16)) {
                float16 *optr = reinterpret_cast<float16 *>(&object[0]);
                for (size_t i = 0; i < getDimension(); i++) {
                    os << optr[i] << " ";
                }
            } else {
                os << " not implement for the type.";
            }
        }

    };

} // namespace polaris

