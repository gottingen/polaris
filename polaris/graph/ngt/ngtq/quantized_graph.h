//
// Copyright (C) 2020 Yahoo Japan Corporation
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

#include <polaris/graph/ngt/index.h>
#include <polaris/graph/ngt/ngtq/quantizer.h>

#ifdef NGTQ_QBG

#define GLOBAL_SIZE    1


namespace QBG {
    class BuildParameters;
};

namespace NGTQG {
    class SearchContainer : public polaris::SearchContainer {
    public:
        SearchContainer(SearchContainer &sc, polaris::Object &f) : polaris::SearchContainer(sc, f),
                                                               resultExpansion(sc.resultExpansion) {}

        SearchContainer() : resultExpansion(0.0) {}

        void setResultExpansion(float re) { resultExpansion = re; }

        float resultExpansion;
    };

    class SearchQuery : public polaris::QueryContainer, public NGTQG::SearchContainer {
    public:
        template<typename QTYPE>
        SearchQuery(const std::vector<QTYPE> &q): polaris::QueryContainer(q) {}
    };

    class QuantizedNode {
    public:
        ~QuantizedNode() {
            clear();
        }

        void clear() {
            ids.clear();
            delete[] static_cast<uint8_t *>(objects);
            objects = 0;
        }

        uint32_t subspaceID;
        std::vector<uint32_t> ids;
        void *objects;
    };

    typedef QuantizedNode RearrangedQuantizedObjectSet;

    class QuantizedGraphRepository : public std::vector<QuantizedNode> {
        typedef std::vector<QuantizedNode> PARENT;
    public:
        QuantizedGraphRepository(NGTQ::NgtqIndex &quantizedIndex) : numOfSubspaces(
                quantizedIndex.getQuantizer().property.localDivisionNo) {}

        ~QuantizedGraphRepository() {}

        void *get(size_t id) {
            return PARENT::at(id).objects;
        }

        std::vector<uint32_t> &getIDs(size_t id) {
            return PARENT::at(id).ids;
        }

        void construct(polaris::NgtIndex &ngtindex, NGTQ::NgtqIndex &quantizedIndex, size_t maxNoOfEdges) {
            polaris::GraphAndTreeIndex &index = static_cast<polaris::GraphAndTreeIndex &>(ngtindex.getIndex());
            polaris::NeighborhoodGraph &graph = static_cast<polaris::NeighborhoodGraph &>(index);
            polaris::GraphRepository &graphRepository = graph.repository;
            construct(graphRepository, quantizedIndex, maxNoOfEdges);
        }

        void construct(polaris::GraphRepository &graphRepository, NGTQ::NgtqIndex &quantizedIndex, size_t maxNoOfEdges) {
            NGTQ::InvertedIndexEntry<uint16_t> invertedIndexObjects(numOfSubspaces);
            quantizedIndex.getQuantizer().extractInvertedIndexObject(invertedIndexObjects);
            quantizedIndex.getQuantizer().eraseInvertedIndexObject();


            std::cerr << "graph repository size=" << graphRepository.size() << std::endl;
            PARENT::resize(graphRepository.size());

            for (size_t id = 1; id < graphRepository.size(); id++) {
                if ((graphRepository.size() > 100) && ((id % ((graphRepository.size() - 1) / 100)) == 0)) {
                    std::cerr << "# of processed objects=" << id << "/" << (graphRepository.size() - 1)
                              << "(" << id * 100 / (graphRepository.size() - 1) << "%)" << std::endl;
                }
                polaris::GraphNode &node = *graphRepository.VECTOR::get(id);
                size_t numOfEdges = node.size() < maxNoOfEdges ? node.size() : maxNoOfEdges;
                (*this)[id].ids.reserve(numOfEdges);
                NGTQ::QuantizedObjectProcessingStream quantizedStream(quantizedIndex.getQuantizer().divisionNo,
                                                                      numOfEdges);
                for (auto i = node.begin(); i != node.end(); i++) {
                    if (distance(node.begin(), i) >= static_cast<int64_t>(numOfEdges)) {
                        break;
                    }
                    if ((*i).id == 0) {
                        std::cerr << "something strange" << std::endl;
                        abort();
                        continue;
                    }
                    (*this)[id].ids.push_back((*i).id);
                    for (size_t idx = 0; idx < numOfSubspaces; idx++) {
                        size_t dataNo = distance(node.begin(), i);
                        if (invertedIndexObjects[(*i).id].localID[idx] < 1 ||
                            invertedIndexObjects[(*i).id].localID[idx] > 16) {
                            std::stringstream msg;
                            msg << "Fatal inner error! Invalid local centroid ID. ID=" << (*i).id << ":"
                                << invertedIndexObjects[(*i).id].localID[idx];
                            POLARIS_THROW_EX(msg);
                        }
                        quantizedStream.arrangeQuantizedObject(dataNo, idx,
                                                               invertedIndexObjects[(*i).id].localID[idx] - 1);
                    }
                }


                (*this)[id].objects = quantizedStream.compressIntoUint4();
            }
        }

        void serialize(std::ofstream &os, polaris::ObjectSpace *objspace = 0) {
            NGTQ::QuantizedObjectProcessingStream quantizedObjectProcessingStream(numOfSubspaces);
            uint64_t n = numOfSubspaces;
            polaris::Serializer::write(os, n);
            n = PARENT::size();
            polaris::Serializer::write(os, n);
            for (auto i = PARENT::begin(); i != PARENT::end(); ++i) {
                uint32_t sid = (*i).subspaceID;
                polaris::Serializer::write(os, sid);
                polaris::Serializer::write(os, (*i).ids);
                size_t streamSize = quantizedObjectProcessingStream.getUint4StreamSize((*i).ids.size());
                polaris::Serializer::write(os, static_cast<uint8_t *>((*i).objects), streamSize);
            }
        }

        void deserialize(std::ifstream &is, polaris::ObjectSpace *objectspace = 0) {
            try {
                NGTQ::QuantizedObjectProcessingStream quantizedObjectProcessingStream(numOfSubspaces);
                uint64_t n;
                polaris::Serializer::read(is, n);
                numOfSubspaces = n;
                polaris::Serializer::read(is, n);
                PARENT::resize(n);
                for (auto i = PARENT::begin(); i != PARENT::end(); ++i) {
                    uint32_t sid;
                    polaris::Serializer::read(is, sid);
                    (*i).subspaceID = sid;
                    polaris::Serializer::read(is, (*i).ids);
                    size_t streamSize = quantizedObjectProcessingStream.getUint4StreamSize((*i).ids.size());
                    uint8_t *objectStream = new uint8_t[streamSize];
                    polaris::Serializer::read(is, objectStream, streamSize);
                    (*i).objects = objectStream;
                }
            } catch (polaris::PolarisException &err) {
                std::stringstream msg;
                msg << "QuantizedGraph::deserialize: Fatal error. " << err.what();
                POLARIS_THROW_EX(msg);
            }
        }

        bool stat(const string &path) {
            struct stat st;
            const std::string p(path + "/grp");
            return ::stat(p.c_str(), &st) == 0;
        }

        void save(const string &path) {
            if (PARENT::size() == 0) {
                return;
            }
            const std::string p(path + "/grp");
            std::ofstream os(p);
            serialize(os);
        }

        void load(const string &path) {
            const std::string p(path + "/grp");
            std::ifstream is(p);
            deserialize(is);
        }

        size_t numOfSubspaces;
    };


    class NgtqgIndex : public polaris::NgtIndex {
    public:
        NgtqgIndex(const std::string &indexPath, size_t maxNoOfEdges = 128, bool rdOnly = false) :
                polaris::NgtIndex(indexPath, rdOnly, polaris::NgtIndex::OpenTypeNone),
                readOnly(rdOnly),
                path(indexPath),
                quantizedIndex(indexPath + "/qg", rdOnly),
                quantizedGraph(quantizedIndex) {
            {
                struct stat st;
                std::string qgpath(path + "/qg/grp");
                if (stat(qgpath.c_str(), &st) == 0) {
                    quantizedGraph.load(path + "/qg");
                } else {
                    if (readOnly) {
                        std::cerr << "No quantized graph. Construct it temporarily." << std::endl;
                    }
                    quantizedGraph.construct(*this, quantizedIndex, maxNoOfEdges);
                }
            }
        }

        void save() {
            quantizedGraph.save(path + "/qg");
        }


        void
        searchQuantizedGraph(polaris::NeighborhoodGraph &graph, NGTQG::SearchContainer &sc, polaris::ObjectDistances &seeds) {
            size_t sizeBackup = sc.size;
            if (sc.resultExpansion > 1.0) {
                sc.size *= sc.resultExpansion;
            }

            NGTQ::Quantizer &quantizer = quantizedIndex.getQuantizer();
            NGTQ::QuantizedObjectDistance &quantizedObjectDistance = quantizer.getQuantizedObjectDistance();


#ifdef NGTQ_QBG
            NGTQ::QuantizedObjectDistance::DistanceLookupTableUint8 cache[GLOBAL_SIZE];
#else
            NGTQ::QuantizedObjectDistance::DistanceLookupTableUint8 cache[GLOBAL_SIZE + 1];
#endif
            auto rotatedQuery = graph.getObjectSpace().getObject(sc.object);
            if (quantizer.property.dimension > rotatedQuery.size()) {
                rotatedQuery.resize(quantizer.property.dimension);
            }
#ifndef NGTQG_NO_ROTATION
            quantizedObjectDistance.rotation->mul(rotatedQuery.data());
#endif
#ifdef NGTQ_QBG
            for (int i = 0; i < GLOBAL_SIZE; i++) {
#else
                for (int i = 1; i < GLOBAL_SIZE + 1; i++) {
#endif
                quantizedObjectDistance.initialize(cache[i]);
                quantizedObjectDistance.createDistanceLookup(rotatedQuery.data(), i, cache[i]);
            }
            if (sc.explorationCoefficient == 0.0) {
                sc.explorationCoefficient = NGT_EXPLORATION_COEFFICIENT;
            }
            polaris::NeighborhoodGraph::UncheckedSet unchecked;
            polaris::NeighborhoodGraph::DistanceCheckedSet distanceChecked(
                    polaris::NgtIndex::getObjectSpace().getRepository().size());
            polaris::NeighborhoodGraph::ResultSet results;

            graph.setupDistances(sc, seeds, polaris::PrimitiveComparator::L2Float::compare);
            graph.setupSeeds(sc, seeds, results, unchecked, distanceChecked);
            auto specifiedRadius = sc.radius;
            polaris::distance_t explorationRadius = sc.explorationCoefficient * sc.radius;
            polaris::ObjectDistance result;
            polaris::ObjectDistance target;

            while (!unchecked.empty()) {
                target = unchecked.top();
                unchecked.pop();
                if (target.distance > explorationRadius) {
                    break;
                }
                auto &neighborIDs = quantizedGraph.getIDs(target.id);
                size_t neighborSize = neighborIDs.size();
                float ds[neighborSize + NGTQ_SIMD_BLOCK_SIZE];

#ifdef NGTQG_PREFETCH
                {
                    uint8_t *lid = static_cast<uint8_t *>(quantizedGraph.get(target.id));
                    size_t size = ((neighborSize - 1) / (NGTQ_SIMD_BLOCK_SIZE * NGTQ_BATCH_SIZE) + 1) *
                                  (NGTQ_SIMD_BLOCK_SIZE * NGTQ_BATCH_SIZE);
                    size /= 2;
                    size *= quantizedIndex.getQuantizer().divisionNo;

                    polaris::MemoryCache::prefetch(lid, size);
                }
#endif
#ifdef NGTQ_QBG
                quantizedObjectDistance(quantizedGraph.get(target.id), ds, neighborSize, cache[0]);
#else
                quantizedObjectDistance(quantizedGraph.get(target.id), ds, neighborSize, cache[1]);
#endif
                for (size_t idx = 0; idx < neighborSize; idx++) {
                    polaris::distance_t distance = ds[idx];
                    auto objid = neighborIDs[idx];
                    if (distance <= explorationRadius) {
                        bool checked = distanceChecked[objid];
                        if (checked) {
                            continue;
                        }
#ifdef NGT_VISIT_COUNT
                        sc.visitCount++;
#endif
                        distanceChecked.insert(objid);
                        result.set(objid, distance);
                        unchecked.push(result);
                        if (distance <= sc.radius) {
                            results.push(result);
                            if (results.size() >= sc.size) {
                                if (results.size() > sc.size) {
                                    results.pop();
                                }
                                sc.radius = results.top().distance;
                                explorationRadius = sc.explorationCoefficient * sc.radius;
                            }
                        }
                    }
                }
            }

            if (sc.resultIsAvailable()) {
                polaris::ObjectDistances &qresults = sc.getResult();
                qresults.moveFrom(results);
                if (sc.resultExpansion >= 1.0) {
                    {
                        polaris::ObjectRepository &objectRepository = polaris::NgtIndex::getObjectSpace().getRepository();
                        auto &comparator = polaris::NgtIndex::getObjectSpace().getComparator();
                        for (auto i = qresults.begin(); i != qresults.end(); ++i) {
#ifdef NGTQG_PREFETCH
                            if (static_cast<size_t>(distance(qresults.begin(), i + 10)) < qresults.size()) {
                                polaris::Object &o = *objectRepository[(*(i + 10)).id];
                                _mm_prefetch(&o[0], _MM_HINT_T0);
                            }
#endif
                            polaris::Object &obj = *objectRepository[(*i).id];
                            (*i).distance = comparator(sc.object.get_view(), obj.get_view());
                        }
                        std::sort(qresults.begin(), qresults.end());
                        if (specifiedRadius < std::numeric_limits<float>::max()) {
                            auto pos = std::upper_bound(qresults.begin(), qresults.end(),
                                                        polaris::ObjectDistance(0, specifiedRadius));
                            qresults.resize(distance(qresults.begin(), pos));
                        }
                    }
                    sc.size = sizeBackup;
                    if (sc.size < qresults.size()) {
                        qresults.resize(sc.size);
                    }
                }
            } else {
                if (sc.resultExpansion >= 1.0) {
                    {
                        polaris::ObjectRepository &objectRepository = polaris::NgtIndex::getObjectSpace().getRepository();
                        auto &comparator = polaris::NgtIndex::getObjectSpace().getComparator();
                        while (!sc.workingResult.empty()) { sc.workingResult.pop(); }
                        while (!results.empty()) {
                            polaris::ObjectDistance obj = results.top();
                            obj.distance = comparator(sc.object.get_view(), objectRepository.get(obj.id)->get_view());
                            results.pop();
                            sc.workingResult.push(obj);
                        }
                        sc.size = sizeBackup;
                        while (sc.workingResult.size() > sc.size) { sc.workingResult.pop(); }
                        if (specifiedRadius < std::numeric_limits<float>::max()) {
                            while (sc.workingResult.top().distance > specifiedRadius) { sc.workingResult.pop(); }
                        }
                    }
                } else {
                    sc.workingResult = std::move(results);
                }
            }

        }


        void search(polaris::GraphIndex &index, NGTQG::SearchContainer &sc, polaris::ObjectDistances &seeds) {
            if (sc.size == 0) {
                while (!sc.workingResult.empty()) sc.workingResult.pop();
                return;
            }
            if (seeds.size() == 0) {
                index.getSeedsFromGraph(index.getObjectSpace().getRepository(), seeds);
            }
            if (sc.expectedAccuracy > 0.0) {
                sc.setEpsilon(getEpsilonFromExpectedAccuracy(sc.expectedAccuracy));
            }
            try {
                searchQuantizedGraph(static_cast<polaris::NeighborhoodGraph &>(index), sc, seeds);
            } catch (polaris::PolarisException &err) {
                std::cerr << err.what() << std::endl;
                polaris::PolarisException e(err);
                throw e;
            }
        }

        void search(NGTQG::SearchQuery &sq) {
            polaris::GraphAndTreeIndex &index = static_cast<polaris::GraphAndTreeIndex &>(getIndex());
            polaris::Object *query = NgtqgIndex::allocateQuery(sq);
            try {
                NGTQG::SearchContainer sc(sq, *query);
                sc.distanceComputationCount = 0;
                sc.visitCount = 0;
                polaris::ObjectDistances seeds;
                if (!readOnly) {
                    try {
                        index.getSeedsFromTree(sc, seeds);
                    } catch (...) {}
                }
                NGTQG::NgtqgIndex::search(static_cast<polaris::GraphIndex &>(index), sc, seeds);
                sq.workingResult = std::move(sc.workingResult);
                sq.distanceComputationCount = sc.distanceComputationCount;
                sq.visitCount = sc.visitCount;
            } catch (polaris::PolarisException &err) {
                deleteObject(query);
                throw err;
            }
            deleteObject(query);
        }

        static size_t getNumberOfSubvectors(size_t dimension, size_t dimensionOfSubvector) {
            if (dimensionOfSubvector == 0) {
                dimensionOfSubvector = dimension > 400 ? 2 : 1;
                dimensionOfSubvector = (dimension % dimensionOfSubvector == 0) ? dimensionOfSubvector : 1;
            }
            if (dimension % dimensionOfSubvector != 0) {
                stringstream msg;
                msg << "Quantizer::getNumOfSubvectors: dimensionOfSubvector is invalid. " << dimension << " : "
                    << dimensionOfSubvector << std::endl;
                POLARIS_THROW_EX(msg);
            }
            return dimension / dimensionOfSubvector;
        }

        static void buildQuantizedGraph(const std::string indexPath, size_t maxNumOfEdges = 128) {
            const std::string qgPath(indexPath + "/qg");
            NGTQ::NgtqIndex quantizedIndex(qgPath, false);
            NGTQG::QuantizedGraphRepository quantizedGraph(quantizedIndex);
            {
                struct stat st;
                std::string qgGraphPath(qgPath + "/grp");
                if (stat(qgGraphPath.c_str(), &st) == 0) {
                    std::cerr << "already exists" << std::endl;
                    abort();
                } else {
                    polaris::GraphRepository graph;
                    polaris::GraphIndex::loadGraph(indexPath, graph);
                    quantizedGraph.construct(graph, quantizedIndex, maxNumOfEdges);
                    quantizedGraph.save(qgPath);
                }
            }

            std::cerr << "Quantized graph is completed." << std::endl;
            std::cerr << "  vmsize=" << polaris::Common::getProcessVmSizeStr() << std::endl;
            std::cerr << "  peak vmsize=" << polaris::Common::getProcessVmPeakStr() << std::endl;
        }

#ifndef NGTQ_QBG
        static void buildQuantizedObjects(const std::string quantizedIndexPath, polaris::ObjectSpace &objectSpace, bool insertion = false) {
          NGTQ::NgtqIndex	quantizedIndex(quantizedIndexPath);
          NGTQ::Quantizer	&quantizer = quantizedIndex.getQuantizer();

          {
        std::vector<float> meanObject(objectSpace.getDimension(), 0);
        quantizedIndex.getQuantizer().globalCodebookIndex.insert(meanObject);
        quantizedIndex.getQuantizer().globalCodebookIndex.createIndex(8);
          }

          vector<pair<polaris::Object*, size_t> > objects;
          for (size_t id = 1; id < objectSpace.getRepository().size(); id++) {
        if (id % 100000 == 0) {
          std::cerr << "Processed " << id << " objects." << std::endl;
        }
        std::vector<float> object;
        try {
          objectSpace.getObject(id, object);
        } catch(...) {
          std::cerr << "NGTQG::buildQuantizedObjects: Warning! Cannot get the object. " << id << std::endl;
          continue;
        }
        quantizer.insert(object, objects, id);
          }
          if (objects.size() > 0) {
        quantizer.insert(objects);
          }

          quantizedIndex.save();
          quantizedIndex.close();
        }
#endif

#ifdef NGTQ_QBG

        static void
        createQuantizedGraphFrame(const std::string quantizedIndexPath, size_t dimension, size_t pseudoDimension,
                                  size_t dimensionOfSubvector) {
#else
            static void createQuantizedGraphFrame(const std::string quantizedIndexPath, size_t dimension, size_t dimensionOfSubvector) {
#endif
            NGTQ::Property property;
            polaris::Property globalProperty;
            polaris::Property localProperty;

            property.threadSize = 24;
            property.globalRange = 0;
            property.localRange = 0;
            property.dataType = NGTQ::DataTypeFloat;
            property.distanceType =  polaris::MetricType::METRIC_L2;
            property.singleLocalCodebook = false;
            property.batchSize = 1000;
            property.centroidCreationMode = NGTQ::CentroidCreationModeStatic;
#ifdef NGTQ_QBG
            property.localCentroidCreationMode = NGTQ::CentroidCreationModeStatic;
#else
            property.localCentroidCreationMode = NGTQ::CentroidCreationModeDynamicKmeans;
#endif

            property.globalCentroidLimit = 1;
            property.localCentroidLimit = 16;
            property.localClusteringSampleCoefficient = 100;
#ifdef NGTQ_QBG
            property.genuineDimension = dimension;
            if (pseudoDimension == 0) {
                property.dimension = dimension;
            } else {
                property.dimension = pseudoDimension;
            }
#else
            property.dimension = dimension;
#endif
            property.localDivisionNo = NGTQG::NgtqgIndex::getNumberOfSubvectors(property.dimension, dimensionOfSubvector);
            globalProperty.edgeSizeForCreation = 10;
            globalProperty.edgeSizeForSearch = 40;
            globalProperty.indexType = polaris::Property::GraphAndTree;
            globalProperty.insertionRadiusCoefficient = 1.1;

            localProperty.indexType = polaris::Property::GraphAndTree;
            localProperty.insertionRadiusCoefficient = 1.1;

            NGTQ::NgtqIndex::create(quantizedIndexPath, property, globalProperty, localProperty);

        }

#ifdef NGTQ_QBG

        static void create(const std::string indexPath, QBG::BuildParameters &buildParameters);

#endif

        static void create(const std::string indexPath, size_t dimensionOfSubvector, size_t pseudoDimension) {
            polaris::NgtIndex index(indexPath);
            std::string quantizedIndexPath = indexPath + "/qg";
            struct stat st;
            if (stat(quantizedIndexPath.c_str(), &st) == 0) {
                std::stringstream msg;
                msg << "QuantizedGraph::create: Quantized graph is already existed. " << indexPath;
                POLARIS_THROW_EX(msg);
            }
            polaris::Property ngtProperty;
            index.getProperty(ngtProperty);
            //NGTQG::Command::CreateParameters createParameters(args, property.dimension);
#ifdef NGTQ_QBG
            int align = 16;
            if (pseudoDimension == 0) {
                pseudoDimension = ((ngtProperty.dimension - 1) / align + 1) * align;
            }
            if (ngtProperty.dimension > static_cast<int>(pseudoDimension)) {
                std::stringstream msg;
                msg
                        << "QuantizedGraph::quantize: the specified pseudo dimension is smaller than the genuine dimension. "
                        << ngtProperty.dimension << ":" << pseudoDimension << std::endl;
                POLARIS_THROW_EX(msg);
            }
            if (pseudoDimension % align != 0) {
                std::stringstream msg;
                msg << "QuantizedGraph::quantize: the specified pseudo dimension should be a multiple of " << align
                    << ". "
                    << pseudoDimension << std::endl;
                POLARIS_THROW_EX(msg);
            }
            createQuantizedGraphFrame(quantizedIndexPath, ngtProperty.dimension, pseudoDimension, dimensionOfSubvector);
#else
            createQuantizedGraphFrame(quantizedIndexPath, ngtProperty.dimension, dimensionOfSubvector);
#endif
            return;
        }

        static void append(const std::string indexPath, QBG::BuildParameters &buildParameters);

#ifdef NGTQ_QBG

        static void
        quantize(const std::string indexPath, size_t dimensionOfSubvector, size_t maxNumOfEdges, bool verbose = false);

        static void realign(const std::string indexPath, size_t maxNumOfEdges, bool verbose = false) {
            polaris::StdOstreamRedirector redirector(!verbose);
            redirector.begin();
            {
                std::string quantizedIndexPath = indexPath + "/qg";
                struct stat st;
                if (stat(quantizedIndexPath.c_str(), &st) != 0) {
                    std::stringstream msg;
                    msg << "QuantizedGraph::quantize: Quantized graph is already existed. " << quantizedIndexPath;
                    POLARIS_THROW_EX(msg);
                }
                if (maxNumOfEdges == 0) {
                    POLARIS_THROW_EX("QuantizedGraph::quantize: The maximum number of edges is zero.");
                }
                buildQuantizedGraph(indexPath, maxNumOfEdges);
            }
            redirector.end();
        }

#else
        static void quantize(const std::string indexPath, float dimensionOfSubvector, size_t maxNumOfEdges, bool verbose = false) {
          polaris::StdOstreamRedirector redirector(!verbose);
          redirector.begin();
          {
        polaris::NgtIndex	index(indexPath);
        polaris::ObjectSpace &objectSpace = index.getObjectSpace();
        std::string quantizedIndexPath = indexPath + "/qg";
        struct stat st;
        if (stat(quantizedIndexPath.c_str(), &st) != 0) {
          polaris::Property ngtProperty;
          index.getProperty(ngtProperty);
          createQuantizedGraphFrame(quantizedIndexPath, ngtProperty.dimension, dimensionOfSubvector);
          buildQuantizedObjects(quantizedIndexPath, objectSpace);
          if (maxNumOfEdges != 0) {
            buildQuantizedGraph(indexPath, maxNumOfEdges);
          }
        }
          }
          redirector.end();
        }
#endif

        const bool readOnly;
        const std::string path;
        NGTQ::NgtqIndex quantizedIndex;
        NGTQ::NgtqIndex blobIndex;

        QuantizedGraphRepository quantizedGraph;

    };

}

#endif
