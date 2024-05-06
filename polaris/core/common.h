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

#include <string>
#include <vector>
#include <queue>
#include <map>
#include <set>
#include <iostream>
#include <sstream>
#include <fstream>
#include <cassert>
#include <cstdlib>
#include <cmath>
#include <cfloat>
#include <climits>
#include <iomanip>
#include <algorithm>
#include <typeinfo>


#include <polaris/core/defines.h>
#include <polaris/core/metric_type.h>
#include <polaris/utility/shared_memory_allocator.h>
#include <polaris/utility/half.hpp>
#include <polaris/utility/bfloat.h>
#include <polaris/utility/polaris_exception.h>
#include <polaris/utility/serialize.h>
#include <polaris/distance/object_distance.h>

#define ADVANCED_USE_REMOVED_LIST
#define    SHARED_REMOVED_LIST

namespace polaris {
    typedef unsigned int ObjectID;
    typedef float distance_t;
    typedef half_float::half float16;
    typedef polaris::bfloat16 bfloat16;

    enum ObjectType {
        ObjectTypeNone = 0,
        UINT8 = 1,
        INT8 = 2,
        UINT16 = 3,
        INT16 = 4,
        UINT32 = 5,
        INT32 = 6,
        UINT64 = 7,
        INT64 = 8,
        FLOAT = 9,
        FLOAT16 = 10,
        DOUBLE = 11,
        BFLOAT16 = 12
    };

    struct SpaceMeta {
        ObjectType object_type;
        size_t dimension;
        size_t byte_size;
        MetricType metric_type;
    };

#pragma pack()

    class Object;

    class ObjectDistances;

    class Container {
    public:
        Container(Object &o, ObjectID i) : object(o), id(i) {}

        Container(Container &c) : object(c.object), id(c.id) {}

        bool isEmptyObject() { return &object == 0; }

        Object &object;
        ObjectID id;
    };

    class SearchContainer : public polaris::Container {
    public:
        SearchContainer(Object &f, ObjectID i) : Container(f, i) { initialize(); }

        SearchContainer(Object &f) : Container(f, 0) { initialize(); }

        SearchContainer(SearchContainer &sc) : Container(sc) { *this = sc; }

        SearchContainer(SearchContainer &sc, Object &f) : Container(f, sc.id) { *this = sc; }

        SearchContainer() : Container(*reinterpret_cast<Object *>(0), 0) { initialize(); }

        SearchContainer &operator=(SearchContainer &sc) {
            size = sc.size;
            radius = sc.radius;
            explorationCoefficient = sc.explorationCoefficient;
            result = sc.result;
            distanceComputationCount = sc.distanceComputationCount;
            edgeSize = sc.edgeSize;
            workingResult = sc.workingResult;
            useAllNodesInLeaf = sc.useAllNodesInLeaf;
            expectedAccuracy = sc.expectedAccuracy;
            visitCount = sc.visitCount;
            return *this;
        }

        virtual ~SearchContainer() {}

        virtual void initialize() {
            size = 10;
            radius = FLT_MAX;
            explorationCoefficient = 1.1;
            result = 0;
            edgeSize = -1;    // dynamically prune the edges during search. -1 means following the index property. 0 means using all edges.
            useAllNodesInLeaf = false;
            expectedAccuracy = -1.0;
        }

        void setSize(size_t s) { size = s; }

        void setResults(ObjectDistances *r) { result = r; }

        void setRadius(distance_t r) { radius = r; }

        void setEpsilon(float e) { explorationCoefficient = e + 1.0; }

        void setEdgeSize(int e) { edgeSize = e; }

        void setExpectedAccuracy(float a) { expectedAccuracy = a; }

        inline bool resultIsAvailable() { return result != 0; }

        ObjectDistances &getResult() {
            if (result == 0) {
                POLARIS_THROW_EX("Inner error: results is not set");
            }
            return *result;
        }

        ResultPriorityQueue &getWorkingResult() { return workingResult; }


        size_t size;
        distance_t radius;
        float explorationCoefficient;
        int edgeSize;
        size_t distanceComputationCount;
        ResultPriorityQueue workingResult;
        bool useAllNodesInLeaf;
        size_t visitCount;
        float expectedAccuracy;
    private:
        ObjectDistances *result;
    };


    class QueryContainer {
    public:
        template<typename QTYPE>
        QueryContainer(const std::vector<QTYPE> &q):query(0) { setQuery(q); }

        ~QueryContainer() { deleteQuery(); }

        template<typename QTYPE>
        void setQuery(const std::vector<QTYPE> &q) {
            if (query != 0) {
                deleteQuery();
            }
            queryType = &typeid(QTYPE);
            if (*queryType != typeid(float) &&
                *queryType != typeid(float16) &&
                *queryType != typeid(double) &&
                *queryType != typeid(uint8_t)) {
                query = 0;
                queryType = 0;
                dimension = 0;
                std::stringstream msg;
                msg << "polaris::SearchQuery: Invalid query type!";
                POLARIS_THROW_EX(msg);
            }
            query = new std::vector<QTYPE>(q);
            dimension = q.size();
#ifdef NGT_REFINEMENT
            refinementExpansion = 0.0;
#endif
        }

        void *getQuery() { return query; }

        size_t getDimension() { return dimension; }

        const std::type_info &getQueryType() { return *queryType; }

        void pushBack(float v) {
            if (*queryType == typeid(float)) {
                static_cast<std::vector<float> *>(query)->push_back(v);
            } else if (*queryType == typeid(double)) {
                static_cast<std::vector<double> *>(query)->push_back(v);
            } else if (*queryType == typeid(uint8_t)) {
                static_cast<std::vector<uint8_t> *>(query)->push_back(v);
            } else if (*queryType == typeid(float16)) {
                static_cast<std::vector<float16> *>(query)->push_back(static_cast<float16>(v));
            }
        }

#ifdef NGT_REFINEMENT
        void setRefinementExpansion(float re) { refinementExpansion = re; }
        float getRefinementExpansion() { return refinementExpansion; }
#endif
    private:
        void deleteQuery() {
            if (query == 0) {
                return;
            }
            if (*queryType == typeid(float)) {
                delete static_cast<std::vector<float> *>(query);
            } else if (*queryType == typeid(double)) {
                delete static_cast<std::vector<double> *>(query);
            } else if (*queryType == typeid(uint8_t)) {
                delete static_cast<std::vector<uint8_t> *>(query);
            } else if (*queryType == typeid(float16)) {
                delete static_cast<std::vector<float16> *>(query);
            }
            query = 0;
            queryType = 0;
        }

        void *query;
        const std::type_info *queryType;
        size_t dimension;
#ifdef NGT_REFINEMENT
        float			refinementExpansion;
#endif
    };

    class SearchQuery : public polaris::QueryContainer, public polaris::SearchContainer {
    public:
        template<typename QTYPE>
        SearchQuery(const std::vector<QTYPE> &q):polaris::QueryContainer(q) {}
    };

    class InsertContainer : public Container {
    public:
        InsertContainer(Object &f, ObjectID i) : Container(f, i) {}
    };

    class Comparator {
    public:
        Comparator(size_t d) : dimension(d) {}

        virtual double operator()(Object &objecta, Object &objectb) = 0;

        size_t dimension;

        virtual ~Comparator() {}
    };

} // namespace polaris

