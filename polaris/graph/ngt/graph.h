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

#include <bitset>

#include <polaris/core/defines.h>
#include <polaris/core/common.h>
#include <polaris/graph/ngt/object_space_repository.h>
#include <polaris/utility/hash_based_boolean_set.h>
#include <polaris/utility/boolean_set.h>
#include <polaris/utility/property_set.h>

#ifndef NGT_GRAPH_CHECK_VECTOR
#include <unordered_set>
#endif

#ifdef NGT_GRAPH_UNCHECK_STACK
#include <stack>
#endif

#ifndef NGT_EXPLORATION_COEFFICIENT
#define NGT_EXPLORATION_COEFFICIENT		1.1
#endif

#ifndef NGT_INSERTION_EXPLORATION_COEFFICIENT
#define NGT_INSERTION_EXPLORATION_COEFFICIENT	1.1
#endif

#ifndef NGT_TRUNCATION_THRESHOLD
#define    NGT_TRUNCATION_THRESHOLD        50
#endif

#ifndef NGT_SEED_SIZE
#define    NGT_SEED_SIZE                10
#endif

#ifndef NGT_CREATION_EDGE_SIZE
#define NGT_CREATION_EDGE_SIZE			10
#endif

namespace polaris {
    class Property;

    typedef GraphNode GRAPH_NODE;

    class GraphRepository : public Repository<GRAPH_NODE> {

    public:
        typedef Repository<GRAPH_NODE> VECTOR;

        GraphRepository() : prevsize(0) {
            initialize();
        }

        virtual ~GraphRepository() {
            destruct();
        }

        void initialize() {
            destruct();
            prevsize = new vector<unsigned short>;
        }

        void destruct() {
            deleteAll();
            if (prevsize != 0) {
                delete prevsize;
                prevsize = 0;
            }
        }

        void insert(ObjectID id, ObjectDistances &objects) {
            GRAPH_NODE *r = allocate();
            *r = objects;
            try {
                put(id, r);
            } catch (polaris::PolarisException &exp) {
                delete r;
                throw exp;
            }
            if (id >= prevsize->size()) {
                prevsize->resize(id + 1, 0);
            } else {
                (*prevsize)[id] = 0;
            }
            return;
        }

        inline GRAPH_NODE *get(ObjectID fid, size_t &minsize) {
            GRAPH_NODE *rs = VECTOR::get(fid);
            minsize = (*prevsize)[fid];
            return rs;
        }

        void serialize(std::ofstream &os) {
            VECTOR::serialize(os);
            Serializer::write(os, *prevsize);
        }

        void deserialize(std::ifstream &is) {
            VECTOR::deserialize(is);
            Serializer::read(is, *prevsize);
        }

        void show() {
            for (size_t i = 0; i < this->size(); i++) {
                std::cout << "Show graph " << i << " ";
                if ((*this)[i] == 0) {
                    std::cout << std::endl;
                    continue;
                }
                for (size_t j = 0; j < (*this)[i]->size(); j++) {
                    std::cout << (*this)[i]->at(j).id << ":" << (*this)[i]->at(j).distance << " ";
                }
                std::cout << std::endl;
            }
        }

    public:
        std::vector<unsigned short> *prevsize;
    };

#ifdef NGT_GRAPH_READ_ONLY_GRAPH

    class ReadOnlyGraphNode : public std::vector<std::pair<uint32_t, PersistentObject *>> {
        typedef std::vector<std::pair<uint32_t, PersistentObject *>> PARENT;
    public:
        ReadOnlyGraphNode() : reservedSize(0), usedSize(0) {}

        void reserve(size_t s) {
            PARENT::resize(s);
            for (size_t i = reservedSize; i < s; i++) {
                (*this)[i].first = 0;
            }
            reservedSize = s;
        }

        void push_back(std::pair<uint32_t, PersistentObject *> node) {
            (*this)[usedSize] = node;
            usedSize++;
        }

        size_t size() { return usedSize; }

        void resize(size_t s) {
            if (s <= usedSize) {
                for (size_t i = s; i < usedSize; i++) {
                    (*this)[i].first = 0;
                }
                usedSize = s;
            } else {
                std::cerr << "ReadOnlyGraphNode: Not implemented" << std::endl;
                abort();
            }
        }

        size_t reservedSize;
        size_t usedSize;
    };

    class SearchGraphRepository : public std::vector<ReadOnlyGraphNode> {
    public:
        SearchGraphRepository() {}

        bool isEmpty(size_t idx) { return (*this)[idx].empty(); }

        void deserialize(std::ifstream &is, ObjectRepository &objectRepository) {
            if (!is.is_open()) {
                POLARIS_THROW_EX("polaris::SearchGraph: Not open the specified stream yet.");
            }
            clear();
            size_t s;
            polaris::Serializer::read(is, s);
            resize(s);
            for (size_t id = 0; id < s; id++) {
                char type;
                polaris::Serializer::read(is, type);
                switch (type) {
                    case '-':
                        break;
                    case '+': {
                        ObjectDistances node;
                        node.deserialize(is);
                        ReadOnlyGraphNode &searchNode = at(id);
                        searchNode.reserve(node.size());
                        for (auto ni = node.begin(); ni != node.end(); ni++) {
                            searchNode.push_back(
                                    std::pair<uint32_t, Object *>((*ni).id, objectRepository.get((*ni).id)));
                        }
                    }
                        break;
                    default: {
                        assert(type == '-' || type == '+');
                        break;
                    }
                }
            }
        }

    };

#endif // NGT_GRAPH_READ_ONLY_GRAPH

    class NeighborhoodGraph {
    public:

#ifdef NGT_GRAPH_READ_ONLY_GRAPH

        class Search {
        public:
            static void
            (*getMethod(MetricType dtype, polaris::ObjectType otype, size_t size))(
                    polaris::NeighborhoodGraph &, polaris::SearchContainer &, polaris::ObjectDistances &) {
                if (size < 5000000) {
                    switch (otype) {
                        case polaris::ObjectType::FLOAT:
                            switch (dtype) {
                                case polaris::MetricType::METRIC_NORMALIZED_COSINE :
                                    return normalizedCosineSimilarityFloat;
                                case polaris::MetricType::METRIC_COSINE :
                                    return cosineSimilarityFloat;
                                case polaris::MetricType::METRIC_NORMALIZED_ANGLE :
                                    return normalizedAngleFloat;
                                case polaris::MetricType::METRIC_ANGLE :
                                    return angleFloat;
                                case polaris::MetricType::METRIC_NORMALIZED_L2 :
                                    return normalizedL2Float;
                                case polaris::MetricType::METRIC_L2 :
                                    return l2Float;
                                case polaris::MetricType::METRIC_L1 :
                                    return l1Float;
                                case polaris::MetricType::METRIC_SPARSE_JACCARD :
                                    return sparseJaccardFloat;
                                case polaris::MetricType::METRIC_POINCARE :
                                    return poincareFloat;  // added by Nyapicom
                                case polaris::MetricType::METRIC_LORENTZ :
                                    return lorentzFloat;  // added by Nyapicom
                                default:
                                    return l2Float;
                            }
                            break;
                        case polaris::ObjectType::UINT8:
                            switch (dtype) {
                                case polaris::MetricType::METRIC_HAMMING :
                                    return hammingUint8;
                                case polaris::MetricType::METRIC_JACCARD :
                                    return jaccardUint8;
                                case polaris::MetricType::METRIC_L2 :
                                    return l2Uint8;
                                case polaris::MetricType::METRIC_L1 :
                                    return l1Uint8;
                                default :
                                    return l2Uint8;
                            }
                            break;
                        case polaris::ObjectType::FLOAT16:
                            switch (dtype) {
                                case polaris::MetricType::METRIC_NORMALIZED_COSINE :
                                    return normalizedCosineSimilarityFloat16;
                                case polaris::MetricType::METRIC_COSINE :
                                    return cosineSimilarityFloat16;
                                case polaris::MetricType::METRIC_NORMALIZED_ANGLE :
                                    return normalizedAngleFloat16;
                                case polaris::MetricType::METRIC_ANGLE :
                                    return angleFloat16;
                                case polaris::MetricType::METRIC_NORMALIZED_L2 :
                                    return normalizedL2Float16;
                                case polaris::MetricType::METRIC_L2 :
                                    return l2Float16;
                                case polaris::MetricType::METRIC_L1 :
                                    return l1Float16;
                                case polaris::MetricType::METRIC_SPARSE_JACCARD :
                                    return sparseJaccardFloat16;
                                case polaris::MetricType::METRIC_POINCARE :
                                    return poincareFloat16;  // added by Nyapicom
                                case polaris::MetricType::METRIC_LORENTZ :
                                    return lorentzFloat16;  // added by Nyapicom
                                default:
                                    return l2Float16;
                            }
                            break;
                        default:
                            POLARIS_THROW_EX("polaris::Graph::Search: Not supported object type.");
                            break;
                    }
                    return l1Uint8;
                } else {
                    switch (otype) {
                        case polaris::ObjectType::FLOAT:
                            switch (dtype) {
                                case polaris::MetricType::METRIC_NORMALIZED_COSINE :
                                    return normalizedCosineSimilarityFloatForLargeDataset;
                                case polaris::MetricType::METRIC_COSINE :
                                    return cosineSimilarityFloatForLargeDataset;
                                case polaris::MetricType::METRIC_NORMALIZED_ANGLE :
                                    return normalizedAngleFloatForLargeDataset;
                                case polaris::MetricType::METRIC_ANGLE :
                                    return angleFloatForLargeDataset;
                                case polaris::MetricType::METRIC_NORMALIZED_L2 :
                                    return normalizedL2FloatForLargeDataset;
                                case polaris::MetricType::METRIC_L2 :
                                    return l2FloatForLargeDataset;
                                case polaris::MetricType::METRIC_L1 :
                                    return l1FloatForLargeDataset;
                                case polaris::MetricType::METRIC_SPARSE_JACCARD :
                                    return sparseJaccardFloatForLargeDataset;
                                case polaris::MetricType::METRIC_POINCARE :
                                    return poincareFloatForLargeDataset;
                                case polaris::MetricType::METRIC_LORENTZ :
                                    return lorentzFloatForLargeDataset;
                                default:
                                    return l2FloatForLargeDataset;
                            }
                            break;
                        case polaris::ObjectType::UINT8:
                            switch (dtype) {
                                case polaris::MetricType::METRIC_HAMMING :
                                    return hammingUint8ForLargeDataset;
                                case polaris::MetricType::METRIC_JACCARD :
                                    return jaccardUint8ForLargeDataset;
                                case polaris::MetricType::METRIC_L2 :
                                    return l2Uint8ForLargeDataset;
                                case polaris::MetricType::METRIC_L1 :
                                    return l1Uint8ForLargeDataset;
                                default :
                                    return l2Uint8ForLargeDataset;
                            }
                            break;
                        case polaris::ObjectType::FLOAT16:
                            switch (dtype) {
                                case polaris::MetricType::METRIC_NORMALIZED_COSINE :
                                    return normalizedCosineSimilarityFloat16ForLargeDataset;
                                case polaris::MetricType::METRIC_COSINE :
                                    return cosineSimilarityFloat16ForLargeDataset;
                                case polaris::MetricType::METRIC_NORMALIZED_ANGLE :
                                    return normalizedAngleFloat16ForLargeDataset;
                                case polaris::MetricType::METRIC_ANGLE :
                                    return angleFloat16ForLargeDataset;
                                case polaris::MetricType::METRIC_NORMALIZED_L2 :
                                    return normalizedL2Float16ForLargeDataset;
                                case polaris::MetricType::METRIC_L2 :
                                    return l2Float16ForLargeDataset;
                                case polaris::MetricType::METRIC_L1 :
                                    return l1Float16ForLargeDataset;
                                case polaris::MetricType::METRIC_SPARSE_JACCARD :
                                    return sparseJaccardFloat16ForLargeDataset;
                                case polaris::MetricType::METRIC_POINCARE :
                                    return poincareFloat16ForLargeDataset;
                                case polaris::MetricType::METRIC_LORENTZ :
                                    return lorentzFloat16ForLargeDataset;
                                default:
                                    return l2Float16ForLargeDataset;
                            }
                        default:
                            POLARIS_THROW_EX("polaris::Graph::Search: Not supported object type.");
                            break;
                    }
                    return l1Uint8ForLargeDataset;
                }
            }

            static void l1Uint8(NeighborhoodGraph &graph, polaris::SearchContainer &sc, ObjectDistances &seeds);

            static void l2Uint8(NeighborhoodGraph &graph, polaris::SearchContainer &sc, ObjectDistances &seeds);

            static void l1Float(NeighborhoodGraph &graph, polaris::SearchContainer &sc, ObjectDistances &seeds);

            static void l2Float(NeighborhoodGraph &graph, polaris::SearchContainer &sc, ObjectDistances &seeds);

            static void hammingUint8(NeighborhoodGraph &graph, polaris::SearchContainer &sc, ObjectDistances &seeds);

            static void jaccardUint8(NeighborhoodGraph &graph, polaris::SearchContainer &sc, ObjectDistances &seeds);

            static void sparseJaccardFloat(NeighborhoodGraph &graph, polaris::SearchContainer &sc, ObjectDistances &seeds);

            static void
            cosineSimilarityFloat(NeighborhoodGraph &graph, polaris::SearchContainer &sc, ObjectDistances &seeds);

            static void angleFloat(NeighborhoodGraph &graph, polaris::SearchContainer &sc, ObjectDistances &seeds);

            static void
            normalizedCosineSimilarityFloat(NeighborhoodGraph &graph, polaris::SearchContainer &sc, ObjectDistances &seeds);

            static void
            normalizedAngleFloat(NeighborhoodGraph &graph, polaris::SearchContainer &sc, ObjectDistances &seeds);

            static void normalizedL2Float(NeighborhoodGraph &graph, polaris::SearchContainer &sc, ObjectDistances &seeds);

            static void poincareFloat(NeighborhoodGraph &graph, polaris::SearchContainer &sc,
                                      ObjectDistances &seeds);  // added by Nyapicom
            static void lorentzFloat(NeighborhoodGraph &graph, polaris::SearchContainer &sc,
                                     ObjectDistances &seeds);  // added by Nyapicom

            static void l1Float16(NeighborhoodGraph &graph, polaris::SearchContainer &sc, ObjectDistances &seeds);

            static void l2Float16(NeighborhoodGraph &graph, polaris::SearchContainer &sc, ObjectDistances &seeds);

            static void
            sparseJaccardFloat16(NeighborhoodGraph &graph, polaris::SearchContainer &sc, ObjectDistances &seeds);

            static void
            cosineSimilarityFloat16(NeighborhoodGraph &graph, polaris::SearchContainer &sc, ObjectDistances &seeds);

            static void angleFloat16(NeighborhoodGraph &graph, polaris::SearchContainer &sc, ObjectDistances &seeds);

            static void normalizedCosineSimilarityFloat16(NeighborhoodGraph &graph, polaris::SearchContainer &sc,
                                                          ObjectDistances &seeds);

            static void
            normalizedAngleFloat16(NeighborhoodGraph &graph, polaris::SearchContainer &sc, ObjectDistances &seeds);

            static void normalizedL2Float16(NeighborhoodGraph &graph, polaris::SearchContainer &sc, ObjectDistances &seeds);

            static void poincareFloat16(NeighborhoodGraph &graph, polaris::SearchContainer &sc,
                                        ObjectDistances &seeds);  // added by Nyapicom
            static void lorentzFloat16(NeighborhoodGraph &graph, polaris::SearchContainer &sc,
                                       ObjectDistances &seeds);  // added by Nyapicom

            static void
            l1Uint8ForLargeDataset(NeighborhoodGraph &graph, polaris::SearchContainer &sc, ObjectDistances &seeds);

            static void
            l2Uint8ForLargeDataset(NeighborhoodGraph &graph, polaris::SearchContainer &sc, ObjectDistances &seeds);

            static void
            l1FloatForLargeDataset(NeighborhoodGraph &graph, polaris::SearchContainer &sc, ObjectDistances &seeds);

            static void
            l2FloatForLargeDataset(NeighborhoodGraph &graph, polaris::SearchContainer &sc, ObjectDistances &seeds);

            static void
            hammingUint8ForLargeDataset(NeighborhoodGraph &graph, polaris::SearchContainer &sc, ObjectDistances &seeds);

            static void
            jaccardUint8ForLargeDataset(NeighborhoodGraph &graph, polaris::SearchContainer &sc, ObjectDistances &seeds);

            static void sparseJaccardFloatForLargeDataset(NeighborhoodGraph &graph, polaris::SearchContainer &sc,
                                                          ObjectDistances &seeds);

            static void cosineSimilarityFloatForLargeDataset(NeighborhoodGraph &graph, polaris::SearchContainer &sc,
                                                             ObjectDistances &seeds);

            static void
            angleFloatForLargeDataset(NeighborhoodGraph &graph, polaris::SearchContainer &sc, ObjectDistances &seeds);

            static void
            normalizedCosineSimilarityFloatForLargeDataset(NeighborhoodGraph &graph, polaris::SearchContainer &sc,
                                                           ObjectDistances &seeds);

            static void normalizedAngleFloatForLargeDataset(NeighborhoodGraph &graph, polaris::SearchContainer &sc,
                                                            ObjectDistances &seeds);

            static void normalizedL2FloatForLargeDataset(NeighborhoodGraph &graph, polaris::SearchContainer &sc,
                                                         ObjectDistances &seeds);

            static void
            poincareFloatForLargeDataset(NeighborhoodGraph &graph, polaris::SearchContainer &sc, ObjectDistances &seeds);

            static void
            lorentzFloatForLargeDataset(NeighborhoodGraph &graph, polaris::SearchContainer &sc, ObjectDistances &seeds);

            static void
            l1Float16ForLargeDataset(NeighborhoodGraph &graph, polaris::SearchContainer &sc, ObjectDistances &seeds);

            static void
            l2Float16ForLargeDataset(NeighborhoodGraph &graph, polaris::SearchContainer &sc, ObjectDistances &seeds);

            static void sparseJaccardFloat16ForLargeDataset(NeighborhoodGraph &graph, polaris::SearchContainer &sc,
                                                            ObjectDistances &seeds);

            static void cosineSimilarityFloat16ForLargeDataset(NeighborhoodGraph &graph, polaris::SearchContainer &sc,
                                                               ObjectDistances &seeds);

            static void
            angleFloat16ForLargeDataset(NeighborhoodGraph &graph, polaris::SearchContainer &sc, ObjectDistances &seeds);

            static void
            normalizedCosineSimilarityFloat16ForLargeDataset(NeighborhoodGraph &graph, polaris::SearchContainer &sc,
                                                             ObjectDistances &seeds);

            static void normalizedAngleFloat16ForLargeDataset(NeighborhoodGraph &graph, polaris::SearchContainer &sc,
                                                              ObjectDistances &seeds);

            static void normalizedL2Float16ForLargeDataset(NeighborhoodGraph &graph, polaris::SearchContainer &sc,
                                                           ObjectDistances &seeds);

            static void
            poincareFloat16ForLargeDataset(NeighborhoodGraph &graph, polaris::SearchContainer &sc, ObjectDistances &seeds);

            static void
            lorentzFloat16ForLargeDataset(NeighborhoodGraph &graph, polaris::SearchContainer &sc, ObjectDistances &seeds);

        };

#endif

        class Property {
        public:
            Property() { setDefault(); }

            void setDefault() {
                truncationThreshold = 0;
                edgeSizeForCreation = NGT_CREATION_EDGE_SIZE;
                edgeSizeForSearch = 0;
                edgeSizeLimitForCreation = 5;
                insertionRadiusCoefficient = NGT_INSERTION_EXPLORATION_COEFFICIENT;
                seedSize = NGT_SEED_SIZE;
                seedType = SeedTypeNone;
                truncationThreadPoolSize = 8;
                batchSizeForCreation = 200;
                graphType = GraphTypeANNG;
                dynamicEdgeSizeBase = 30;
                dynamicEdgeSizeRate = 20;
                buildTimeLimit = 0.0;
                outgoingEdge = 10;
                incomingEdge = 80;
            }

            void clear() {
                truncationThreshold = -1;
                edgeSizeForCreation = -1;
                edgeSizeForSearch = -1;
                edgeSizeLimitForCreation = -1;
                insertionRadiusCoefficient = -1;
                seedSize = -1;
                seedType = SeedTypeNone;
                truncationThreadPoolSize = -1;
                batchSizeForCreation = -1;
                graphType = GraphTypeNone;
                dynamicEdgeSizeBase = -1;
                dynamicEdgeSizeRate = -1;
                buildTimeLimit = -1;
                outgoingEdge = -1;
                incomingEdge = -1;
            }

            void set(polaris::Property &prop);

            void get(polaris::Property &prop);

            void exportProperty(polaris::PropertySet &p) {
                p.set("IncrimentalEdgeSizeLimitForTruncation", truncationThreshold);
                p.set("EdgeSizeForCreation", edgeSizeForCreation);
                p.set("EdgeSizeForSearch", edgeSizeForSearch);
                p.set("EdgeSizeLimitForCreation", edgeSizeLimitForCreation);
                assert(insertionRadiusCoefficient >= 1.0);
                p.set("EpsilonForCreation", insertionRadiusCoefficient - 1.0);
                p.set("BatchSizeForCreation", batchSizeForCreation);
                p.set("SeedSize", seedSize);
                p.set("TruncationThreadPoolSize", truncationThreadPoolSize);
                p.set("DynamicEdgeSizeBase", dynamicEdgeSizeBase);
                p.set("DynamicEdgeSizeRate", dynamicEdgeSizeRate);
                p.set("BuildTimeLimit", buildTimeLimit);
                p.set("OutgoingEdge", outgoingEdge);
                p.set("IncomingEdge", incomingEdge);
                auto rs = PropertySerializer::graph_type_export(p, graphType);
                if (!rs.ok()) {
                    std::cerr << "Graph::exportProperty: Fatal error! Invalid Graph Type. " << graphType << std::endl;
                    abort();
                }
                rs = PropertySerializer::seed_type_export(p, seedType);
                if (!rs.ok()) {
                    std::cerr << "Graph::exportProperty: Fatal error! Invalid Seed Type. " << seedType << std::endl;
                    abort();
                }
            }

            void importProperty(polaris::PropertySet &p) {
                setDefault();
                truncationThreshold = p.getl("IncrimentalEdgeSizeLimitForTruncation", truncationThreshold);
                edgeSizeForCreation = p.getl("EdgeSizeForCreation", edgeSizeForCreation);
                edgeSizeForSearch = p.getl("EdgeSizeForSearch", edgeSizeForSearch);
                edgeSizeLimitForCreation = p.getl("EdgeSizeLimitForCreation", edgeSizeLimitForCreation);
                insertionRadiusCoefficient = p.getf("EpsilonForCreation", insertionRadiusCoefficient);
                insertionRadiusCoefficient += 1.0;
                batchSizeForCreation = p.getl("BatchSizeForCreation", batchSizeForCreation);
                seedSize = p.getl("SeedSize", seedSize);
                truncationThreadPoolSize = p.getl("TruncationThreadPoolSize", truncationThreadPoolSize);
                dynamicEdgeSizeBase = p.getl("DynamicEdgeSizeBase", dynamicEdgeSizeBase);
                dynamicEdgeSizeRate = p.getl("DynamicEdgeSizeRate", dynamicEdgeSizeRate);
                buildTimeLimit = p.getf("BuildTimeLimit", buildTimeLimit);
                outgoingEdge = p.getl("OutgoingEdge", outgoingEdge);
                incomingEdge = p.getl("IncomingEdge", incomingEdge);
                auto grs = PropertySerializer::graph_type_import(p);
                if (!grs.ok()) {
                    std::cerr << "Graph::importProperty: Fatal error! Invalid Graph Type. " << grs.status().message() << std::endl;
                    abort();
                }
                graphType = (GraphType) grs.value();
                auto srs = PropertySerializer::seed_type_import(p);
                if (!srs.ok()) {
                    std::cerr << "Graph::importProperty: Fatal error! Invalid Seed Type. " << srs.status().message() << std::endl;
                    abort();
                }
                seedType = (SeedType) srs.value();
                auto it = p.find("SeedType");
            }

            friend std::ostream &operator<<(std::ostream &os, const Property &p) {
                os << "truncationThreshold=" << p.truncationThreshold << std::endl;
                os << "edgeSizeForCreation=" << p.edgeSizeForCreation << std::endl;
                os << "edgeSizeForSearch=" << p.edgeSizeForSearch << std::endl;
                os << "edgeSizeLimitForCreation=" << p.edgeSizeLimitForCreation << std::endl;
                os << "insertionRadiusCoefficient=" << p.insertionRadiusCoefficient << std::endl;
                os << "insertionRadiusCoefficient=" << p.insertionRadiusCoefficient << std::endl;
                os << "seedSize=" << p.seedSize << std::endl;
                os << "seedType=" << p.seedType << std::endl;
                os << "truncationThreadPoolSize=" << p.truncationThreadPoolSize << std::endl;
                os << "batchSizeForCreation=" << p.batchSizeForCreation << std::endl;
                os << "graphType=" << p.graphType << std::endl;
                os << "dynamicEdgeSizeBase=" << p.dynamicEdgeSizeBase << std::endl;
                os << "dynamicEdgeSizeRate=" << p.dynamicEdgeSizeRate << std::endl;
                os << "outgoingEdge=" << p.outgoingEdge << std::endl;
                os << "incomingEdge=" << p.incomingEdge << std::endl;
                return os;
            }

            int16_t truncationThreshold;
            int16_t edgeSizeForCreation;
            int16_t edgeSizeForSearch;
            int16_t edgeSizeLimitForCreation;
            double insertionRadiusCoefficient;
            int16_t seedSize;
            SeedType seedType;
            int16_t truncationThreadPoolSize;
            int16_t batchSizeForCreation;
            GraphType graphType;
            int16_t dynamicEdgeSizeBase;
            int16_t dynamicEdgeSizeRate;
            float buildTimeLimit;
            int16_t outgoingEdge;
            int16_t incomingEdge;
        };

        NeighborhoodGraph() : objectSpace(0) {
            property.truncationThreshold = NGT_TRUNCATION_THRESHOLD;
            // initialize random to generate random seeds
#ifdef NGT_DISABLE_SRAND_FOR_RANDOM
            struct timeval randTime;
            gettimeofday(&randTime, 0);
            srand(randTime.tv_usec);
#endif
        }

        inline GraphNode *getNode(ObjectID fid, size_t &minsize) { return repository.get(fid, minsize); }

        inline GraphNode *getNode(ObjectID fid) { return repository.VECTOR::get(fid); }

        void insertNode(ObjectID id, ObjectDistances &objects) {
            switch (property.graphType) {
                case GraphTypeANNG:
                case GraphTypeRANNG:
                    insertANNGNode(id, objects);
                    break;
                case GraphTypeIANNG:
                case GraphTypeRIANNG:
                    insertIANNGNode(id, objects);
                    break;
                case GraphTypeONNG:
                    insertONNGNode(id, objects);
                    break;
                case GraphTypeKNNG:
                    insertKNNGNode(id, objects);
                    break;
                case GraphTypeBKNNG:
                    insertBKNNGNode(id, objects);
                    break;
                case GraphTypeNone:
                    POLARIS_THROW_EX("polaris::insertNode: GraphType is not specified.");
                    break;
                default:
                    POLARIS_THROW_EX("polaris::insertNode: GraphType is invalid.");
                    break;
            }
        }

        void insertBKNNGNode(ObjectID id, ObjectDistances &results) {
            if (repository.isEmpty(id)) {
                repository.insert(id, results);
            } else {
                GraphNode &rs = *getNode(id);
                for (ObjectDistances::iterator ri = results.begin(); ri != results.end(); ri++) {
                    rs.push_back((*ri));
                }
                std::sort(rs.begin(), rs.end());
                ObjectID prev = 0;
                for (GraphNode::iterator ri = rs.begin(); ri != rs.end();) {
                    if (prev == (*ri).id) {
                        ri = rs.erase(ri);
                        continue;
                    }
                    prev = (*ri).id;
                    ri++;
                }
            }
            for (ObjectDistances::iterator ri = results.begin(); ri != results.end(); ri++) {
                assert(id != (*ri).id);
                addBKNNGEdge((*ri).id, id, (*ri).distance);
            }
            return;
        }

        void insertKNNGNode(ObjectID id, ObjectDistances &results) {
            repository.insert(id, results);
        }

        void insertANNGNode(ObjectID id, ObjectDistances &results) {
            repository.insert(id, results);
            std::queue<ObjectID> truncateQueue;
            for (ObjectDistances::iterator ri = results.begin(); ri != results.end(); ri++) {
                assert(id != (*ri).id);
                if (addEdge((*ri).id, id, (*ri).distance)) {
                    truncateQueue.push((*ri).id);
                }
            }
            while (!truncateQueue.empty()) {
                ObjectID tid = truncateQueue.front();
                truncateEdges(tid);
                truncateQueue.pop();
            }
            return;
        }

        void insertIANNGNode(ObjectID id, ObjectDistances &results) {
            repository.insert(id, results);
            size_t nOfEdges = std::max(property.incomingEdge, property.outgoingEdge);
            nOfEdges = nOfEdges != 0 ? nOfEdges : property.edgeSizeForCreation;
            for (ObjectDistances::iterator ri = results.begin(); ri != results.end(); ri++) {
                assert(id != (*ri).id);
                addEdgeWithDeletion((*ri).id, id, (*ri).distance, nOfEdges);
            }
            return;
        }

        void deleteShortcutEdges(ObjectDistances &srcNode) {
            size_t removeCount = 0;
            std::vector<bool> removedEdge(srcNode.size(), false);
            for (uint32_t rank = 1; rank < srcNode.size(); rank++) {
                auto dstNodeID = srcNode[rank].id;
                auto dstNodeDistance = srcNode[rank].distance;
#ifdef NGT_SHORTCUT_REDUCTION_WITH_ANGLE
                auto dstNodeDistance2 = dstNodeDistance * dstNodeDistance;
#endif
                bool found = false;
                for (size_t sni = 0; sni < srcNode.size() && sni < rank; sni++) {
                    if (removedEdge[sni]) continue;
                    auto pathNodeID = srcNode[sni].id;
#ifdef NGT_SHORTCUT_REDUCTION_WITH_ANGLE
                    auto srcNodeDistance2 = srcNode[sni].distance * srcNode[sni].distance;
#else
                    if (srcNode[sni].distance >= dstNodeDistance) continue;
#endif
                    polaris::GraphNode &pathNode = *getNode(pathNodeID);
                    for (size_t pni = 0; pni < pathNode.size(); pni++) {
                        auto nodeID = pathNode[pni].id;
                        if (nodeID != dstNodeID) continue;
#ifdef NGT_SHORTCUT_REDUCTION_WITH_ANGLE
                        auto pathNodeDistance2 = pathNode[pni].distance * pathNode[pni].distance;
                        auto v1 = srcNodeDistance2 + pathNodeDistance2 - dstNodeDistance2;
                        auto v2 = 2.0 * srcNode[sni].distance * pathNode[pni].distance;
                        if (cosAlpha >= range) {
                      break;
                        }
#else
                        if (pathNode[pni].distance >= dstNodeDistance) break;
#ifdef NGT_SHORTCUT_REDUCTION_WITH_ADDITIONAL_CONDITION
                        if (srcNode[sni].distance + pathNode[pni].distance >= dstNodeDistance * range) break;
#endif
#endif
                        found = true;
                        removeCount++;
                        break;
                    }
                    if (found) {
                        removedEdge[rank] = true;
                    }
                }
            }
            {
                size_t idx = 0;
                for (auto e = srcNode.begin(); e != srcNode.end(); idx++) {
                    if (removedEdge[idx]) {
                        e = srcNode.erase(e);
                    } else {
                        ++e;
                    }
                }
            }
        }

        void insertONNGNode(ObjectID id, ObjectDistances &results) {
            if (property.truncationThreshold != 0) {
                std::stringstream msg;
                msg << "polaris::insertONNGNode: truncation should be disabled!" << std::endl;
                POLARIS_THROW_EX(msg);
            }
            int count = 0;
            for (ObjectDistances::iterator ri = results.begin(); ri != results.end(); ri++, count++) {
                assert(id != (*ri).id);
                if (count >= property.incomingEdge) {
                    break;
                }
                addEdge((*ri).id, id, (*ri).distance);
            }
            if (static_cast<int>(results.size()) > property.outgoingEdge) {
                results.resize(property.outgoingEdge);
            }
            repository.insert(id, results);
        }

        void removeEdgesReliably(ObjectID id);

        int truncateEdgesOptimally(ObjectID id, GraphNode &results, size_t truncationSize);

        int truncateEdges(ObjectID id) {
            GraphNode &results = *getNode(id);
            if (results.size() == 0) {
                return -1;
            }

            size_t truncationSize = NGT_TRUNCATION_THRESHOLD;
            if (truncationSize < (size_t) property.edgeSizeForCreation) {
                truncationSize = property.edgeSizeForCreation;
            }
            return truncateEdgesOptimally(id, results, truncationSize);
        }

        // setup edgeSize
        inline size_t getEdgeSize(polaris::SearchContainer &sc) {
            int64_t esize = sc.edgeSize == -1 ? property.edgeSizeForSearch : sc.edgeSize;
            size_t edgeSize = INT_MAX;

            if (esize == 0) {
                edgeSize = INT_MAX;
            } else if (esize > 0) {
                edgeSize = esize;
            } else if (esize == -2) {
                double add = pow(10,
                                 (sc.explorationCoefficient - 1.0) * static_cast<float>(property.dynamicEdgeSizeRate));
                edgeSize = add >= static_cast<double>(INT_MAX) ? INT_MAX : property.dynamicEdgeSizeBase + add;
            } else {
                std::stringstream msg;
                msg << "polaris::getEdgeSize: Invalid edge size parameters " << sc.edgeSize << ":"
                    << property.edgeSizeForSearch;
                POLARIS_THROW_EX(msg);
            }
            return edgeSize;
        }

        void search(polaris::SearchContainer &sc, ObjectDistances &seeds);

#ifdef NGT_GRAPH_READ_ONLY_GRAPH

        template<typename COMPARATOR, typename CHECK_LIST>
        void searchReadOnlyGraph(polaris::SearchContainer &sc, ObjectDistances &seeds);

#endif

        void removeEdge(ObjectID fid, ObjectID rmid) {
            GraphNode &rs = *getNode(fid);
            for (GraphNode::iterator ri = rs.begin(); ri != rs.end(); ri++) {
                if ((*ri).id == rmid) {
                    rs.erase(ri);
                    break;
                }
            }
        }

        void removeEdge(GraphNode &node, ObjectDistance &edge) {
            GraphNode::iterator ni = std::lower_bound(node.begin(), node.end(), edge);
            if (ni != node.end() && (*ni).id == edge.id) {
                node.erase(ni);
                return;
            }
            if (ni == node.end()) {
                std::stringstream msg;
                msg << "polaris::removeEdge: Cannot found " << edge.id;
                POLARIS_THROW_EX(msg);
            } else {
                std::stringstream msg;
                msg << "polaris::removeEdge: Cannot found " << (*ni).id << ":" << edge.id;
                POLARIS_THROW_EX(msg);
            }
        }

        void
        removeNode(ObjectID id) {
            repository.erase(id);
        }

        class BooleanVector : public std::vector<bool> {
        public:
            inline BooleanVector(size_t s) : std::vector<bool>(s, false) {}

            inline void insert(size_t i) { std::vector<bool>::operator[](i) = true; }
        };

#ifdef NGT_GRAPH_VECTOR_RESULT
        typedef ObjectDistances ResultSet;
#else
        typedef std::priority_queue<ObjectDistance, std::vector<ObjectDistance>, std::less<ObjectDistance> > ResultSet;
#endif

#if defined(NGT_GRAPH_CHECK_BOOLEANSET)
        typedef BooleanSet DistanceCheckedSet;
#elif defined(NGT_GRAPH_CHECK_VECTOR)
        typedef BooleanVector DistanceCheckedSet;
#elif defined(NGT_GRAPH_CHECK_HASH_BASED_BOOLEAN_SET)
        typedef HashBasedBooleanSet DistanceCheckedSet;
#else
        class DistanceCheckedSet : public unordered_set<ObjectID> {
        public:
      bool operator[](ObjectID id) { return find(id) != end(); }
        };
#endif

        typedef HashBasedBooleanSet DistanceCheckedSetForLargeDataset;

        class NodeWithPosition : public ObjectDistance {
        public:
            NodeWithPosition(uint32_t p = 0) : position(p) {}

            NodeWithPosition(ObjectDistance &o) : ObjectDistance(o), position(0) {}

            NodeWithPosition &operator=(const NodeWithPosition &n) {
                ObjectDistance::operator=(static_cast<const ObjectDistance &>(n));
                position = n.position;
                assert(id != 0);
                return *this;
            }

            uint32_t position;
        };

#ifdef NGT_GRAPH_UNCHECK_STACK
        typedef std::stack<ObjectDistance> UncheckedSet;
#else
#ifdef NGT_GRAPH_BETTER_FIRST_RESTORE
        typedef std::priority_queue<NodeWithPosition, std::vector<NodeWithPosition>, std::greater<NodeWithPosition> > UncheckedSet;
#else
        typedef std::priority_queue<ObjectDistance, std::vector<ObjectDistance>, std::greater<ObjectDistance> > UncheckedSet;
#endif
#endif

        void setupDistances(polaris::SearchContainer &sc, ObjectDistances &seeds);

        void setupDistances(polaris::SearchContainer &sc, ObjectDistances &seeds,
                            double (&comparator)(const void *, const void *, size_t));

        void setupSeeds(SearchContainer &sc, ObjectDistances &seeds, ResultSet &results,
                        UncheckedSet &unchecked, DistanceCheckedSet &distanceChecked);

#if !defined(NGT_GRAPH_CHECK_HASH_BASED_BOOLEAN_SET)

        void setupSeeds(SearchContainer &sc, ObjectDistances &seeds, ResultSet &results,
                        UncheckedSet &unchecked, DistanceCheckedSetForLargeDataset &distanceChecked);

#endif


        int getEdgeSize() { return property.edgeSizeForCreation; }

        ObjectRepository &getObjectRepository() { return objectSpace->getRepository(); }

        ObjectSpace &getObjectSpace() { return *objectSpace; }

#ifdef NGT_REFINEMENT
        ObjectSpace &getRefinementObjectSpace() { return *refinementObjectSpace; }
#endif

        void deleteInMemory() {
            for (std::vector<polaris::GraphNode *>::iterator i = repository.begin(); i != repository.end(); i++) {
                if ((*i) != 0) {
                    delete (*i);
                }
            }
            repository.clear();
        }


    protected:
        void
        addBKNNGEdge(ObjectID target, ObjectID addID, distance_t addDistance) {
            if (repository.isEmpty(target)) {
                ObjectDistances objs;
                objs.push_back(ObjectDistance(addID, addDistance));
                repository.insert(target, objs);
                return;
            }
            addEdge(target, addID, addDistance, false);
        }

    public:
        void addEdge(GraphNode &node, ObjectID addID, distance_t addDistance, bool identityCheck = true) {
            ObjectDistance obj(addID, addDistance);
            GraphNode::iterator ni = std::lower_bound(node.begin(), node.end(), obj);
            if ((ni != node.end()) && ((*ni).id == addID)) {
                if (identityCheck) {
                    std::stringstream msg;
                    msg << "polaris::addEdge: already existed! " << (*ni).id << ":" << addID;
                    POLARIS_THROW_EX(msg);
                }
                return;
            }
            node.insert(ni, obj);
        }

        // identityCheck is checking whether the same edge has already added to the node.
        // return whether truncation is needed that means the node has too many edges.
        bool addEdge(ObjectID target, ObjectID addID, distance_t addDistance, bool identityCheck = true) {
            size_t minsize = 0;
            GraphNode &node = property.truncationThreshold == 0 ? *getNode(target) : *getNode(target, minsize);
            try {
                addEdge(node, addID, addDistance, identityCheck);
            } catch (polaris::PolarisException &err) {
                std::stringstream msg;
                msg << " Cannot add the edge. " << target << "->" << addID << ". " << err.what();
                POLARIS_THROW_EX(msg);
            }
            if ((size_t) property.truncationThreshold != 0 && node.size() - minsize >
                                                              (size_t) property.truncationThreshold) {
                return true;
            }
            return false;
        }

        void addEdgeWithDeletion(ObjectID target, ObjectID addID, distance_t addDistance, size_t kEdge,
                                 bool identityCheck = true) {
            GraphNode &node = *getNode(target);
            try {
                while (node.size() >= kEdge && node.back().distance > addDistance) {
                    removeEdge(node, node.back());
                }
            } catch (polaris::PolarisException &exp) {
                std::stringstream msg;
                msg << "addEdge: Cannot remove. (b) " << target << "," << addID << "," << node[kEdge - 1].distance;
                msg << ":" << exp.what();
                POLARIS_THROW_EX(msg.str());
            }
            if (node.size() < kEdge) {
                addEdge(node, addID, addDistance, identityCheck);
                if (node.capacity() > kEdge) {
                    node.shrink_to_fit();
                }
            }
        }


#ifdef NGT_GRAPH_READ_ONLY_GRAPH

        void loadSearchGraph(const std::string &database) {
            std::ifstream isg(database + "/grp");
            NeighborhoodGraph::searchRepository.deserialize(isg, NeighborhoodGraph::getObjectRepository());
        }

#endif

    public:

        GraphRepository repository;
        ObjectSpace *objectSpace;
#ifdef NGT_REFINEMENT
        ObjectSpace	*refinementObjectSpace;
#endif
#ifdef NGT_GRAPH_READ_ONLY_GRAPH
        SearchGraphRepository searchRepository;
#endif

        NeighborhoodGraph::Property property;

    }; // NeighborhoodGraph

} // NGT

