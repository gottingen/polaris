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
#include <polaris/utility/property_set.h>
#include <polaris/distance/object_distance.h>
#include <collie/strings/match.h>

#define ADVANCED_USE_REMOVED_LIST
#define    SHARED_REMOVED_LIST

namespace polaris {
    typedef unsigned int ObjectID;
    typedef float distance_t;
    typedef uint64_t vid_t;
    typedef uint32_t localid_t;
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

    enum DatabaseType {
        DatabaseTypeNone = 0,
        Memory = 1,
        MemoryMappedFile = 2,
        SSD = 3
    };

    inline uint32_t polaris_type_to_size(ObjectType type) {
        switch (type) {
            case UINT8:
                return sizeof(uint8_t);
            case INT8:
                return sizeof(int8_t);
            case UINT16:
                return sizeof(uint16_t);
            case INT16:
                return sizeof(int16_t);
            case UINT32:
                return sizeof(uint32_t);
            case INT32:
                return sizeof(int32_t);
            case UINT64:
                return sizeof(uint64_t);
            case INT64:
                return sizeof(int64_t);
            case FLOAT:
                return sizeof(float);
            case FLOAT16:
                return sizeof(float16);
            case DOUBLE:
                return sizeof(double);
            case BFLOAT16:
                return sizeof(bfloat16);
            default:
                return 0;
        }
    }

    inline std::string polaris_type_to_string(ObjectType type) {
        switch (type) {
            case UINT8:
                return "UINT8";
            case INT8:
                return "INT8";
            case UINT16:
                return "UINT16";
            case INT16:
                return "INT16";
            case UINT32:
                return "UINT32";
            case INT32:
                return "INT32";
            case UINT64:
                return "UINT64";
            case INT64:
                return "INT64";
            case FLOAT:
                return "FLOAT";
            case FLOAT16:
                return "FLOAT16";
            case DOUBLE:
                return "DOUBLE";
            case BFLOAT16:
                return "BFLOAT16";
            default:
                return "UNKNOWN";
        }
    }

    inline ObjectType string_to_polaris_type(const std::string &type) {
        if (collie::str_equals_ignore_case(type, "UINT8")) {
            return UINT8;
        } else if (collie::str_equals_ignore_case(type, "INT8")) {
            return INT8;
        } else if (collie::str_equals_ignore_case(type, "UINT16")) {
            return UINT16;
        } else if (collie::str_equals_ignore_case(type, "INT16")) {
            return INT16;
        } else if (collie::str_equals_ignore_case(type, "UINT32")) {
            return UINT32;
        } else if (collie::str_equals_ignore_case(type, "INT32")) {
            return INT32;
        } else if (collie::str_equals_ignore_case(type, "UINT64")) {
            return UINT64;
        } else if (collie::str_equals_ignore_case(type, "INT64")) {
            return INT64;
        } else if (collie::str_equals_ignore_case(type, "FLOAT")) {
            return FLOAT;
        } else if (collie::str_equals_ignore_case(type, "FLOAT16")) {
            return FLOAT16;
        } else if (collie::str_equals_ignore_case(type, "DOUBLE")) {
            return DOUBLE;
        } else if (collie::str_equals_ignore_case(type, "BFLOAT16")) {
            return BFLOAT16;
        } else {
            return ObjectTypeNone;
        }
    }

    template<typename T>
    inline polaris::ObjectType polaris_type_to_name() = delete;

    template<>
    inline polaris::ObjectType polaris_type_to_name<float>() {
        return polaris::ObjectType::FLOAT;
    }

    template<>
    inline polaris::ObjectType polaris_type_to_name<uint8_t>() {
        return polaris::ObjectType::UINT8;
    }

    template<>
    inline polaris::ObjectType polaris_type_to_name<int8_t>() {
        return polaris::ObjectType::INT8;
    }

    template<>
    inline polaris::ObjectType polaris_type_to_name<uint16_t>() {
        return polaris::ObjectType::UINT16;
    }

    template<>
    inline polaris::ObjectType polaris_type_to_name<int16_t>() {
        return polaris::ObjectType::INT16;
    }

    template<>
    inline polaris::ObjectType polaris_type_to_name<uint32_t>() {
        return polaris::ObjectType::UINT32;
    }

    template<>
    inline polaris::ObjectType polaris_type_to_name<int32_t>() {
        return polaris::ObjectType::INT32;
    }

    template<>
    inline polaris::ObjectType polaris_type_to_name<uint64_t>() {
        return polaris::ObjectType::UINT64;
    }

    template<>
    inline polaris::ObjectType polaris_type_to_name<int64_t>() {
        return polaris::ObjectType::INT64;
    }

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

    enum ObjectAlignment {
        ObjectAlignmentNone = 0,
        ObjectAlignmentTrue = 1,
        ObjectAlignmentFalse = 2
    };

    enum GraphType {
        GraphTypeNone = 0,
        GraphTypeANNG = 1,
        GraphTypeKNNG = 2,
        GraphTypeBKNNG = 3,
        GraphTypeONNG = 4,
        GraphTypeIANNG = 5,    // Improved ANNG
        GraphTypeDNNG = 6,
        GraphTypeRANNG = 7,
        GraphTypeRIANNG = 8,
        GraphTypeHNSW = 9,
        GraphTypeVAMANA = 10
    };

    enum SeedType {
        SeedTypeNone = 0,
        SeedTypeRandomNodes = 1,
        SeedTypeFixedNodes = 2,
        SeedTypeFirstNode = 3,
        SeedTypeAllLeafNodes = 4
    };

    enum IndexType {
        INDEX_NONE = 0,
        // INDEX NGT Graph And Tree
        INDEX_NGT_GRAPH_AND_TREE,
        // INDEX NGT Graph
        INDEX_NGT_GRAPH,
        INDEX_HNSW_FLAT,
        IT_FLAT,
        IT_FLATIP,
        IT_FLATL2,
        IT_LSH,
        IT_IVFFLAT,
        INDEX_VAMANA_DISK,
        INDEX_VAMANA,
        INDEX_HNSW
    };

    struct PropertySerializer {
        static const std::string OBJECT_TYPE;
        static const std::string DISTANCE_TYPE;
        static const std::string DATABASE_TYPE;
        static const std::string OBJECT_ALIGNMENT;
        static const std::string GRAPH_TYPE;
        static const std::string SEED_TYPE;
        static const std::string INDEX_TYPE;
        static const std::string DIMENSION;

        [[nodiscard]] static turbo::Status
        object_type_export(PropertySet &ps, ObjectType type, std::set<ObjectType> *allow_set = nullptr);

        static turbo::ResultStatus<ObjectType>
        object_type_import(const PropertySet &ps, std::set<ObjectType> *allow_set = nullptr);

        [[nodiscard]] static turbo::Status
        distance_type_export(PropertySet &ps, MetricType diss, std::set<MetricType> *allow_set = nullptr);

        static turbo::ResultStatus<MetricType>
        distance_type_import(const PropertySet &ps, std::set<MetricType> *allow_set = nullptr);

        [[nodiscard]] static turbo::Status
        database_type_export(PropertySet &ps, DatabaseType type, std::set<DatabaseType> *allow_set = nullptr);

        static turbo::ResultStatus<DatabaseType>
        database_type_import(const PropertySet &ps, std::set<DatabaseType> *allow_set = nullptr);

        [[nodiscard]] static turbo::Status object_alignment_export(PropertySet &ps, ObjectAlignment type);
        static turbo::ResultStatus<ObjectAlignment> object_alignment_import(const PropertySet &ps);

        [[nodiscard]] static turbo::Status graph_type_export(PropertySet &ps, GraphType type);
        static turbo::ResultStatus<GraphType> graph_type_import(const PropertySet &ps);

        [[nodiscard]] static turbo::Status seed_type_export(PropertySet &ps, SeedType type);
        static turbo::ResultStatus<SeedType> seed_type_import(const PropertySet &ps);

        [[nodiscard]] static turbo::Status index_type_export(PropertySet &ps, IndexType type);
        static turbo::ResultStatus<IndexType> index_type_import(const PropertySet &ps);

        static void dimension_export(PropertySet &ps, size_t dimension);
        static turbo::ResultStatus<size_t> dimension_import(const PropertySet &ps);
    };

} // namespace polaris

