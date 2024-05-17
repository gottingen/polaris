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
#include <map>
#include <set>
#include <bitset>
#include <iomanip>
#include <unordered_set>
#include <thread>

#include <sys/time.h>
#include <sys/stat.h>
#include <stdint.h>
#include <polaris/core/common.h>
#include <polaris/core/parameter/ngt_parameters.h>
#include <polaris/utility/property_set.h>
#include <polaris/graph/ngt/tree.h>
#include <polaris/graph/ngt/thread.h>
#include <polaris/graph/ngt/graph.h>


namespace polaris {

    class NgtIndex {
    public:
        enum OpenType {
            OpenTypeNone = 0x00,
            OpenTypeGraphDisabled = 0x01,
            OpenTypeTreeDisabled = 0x02,
            OpenTypeObjectDisabled = 0x04
        };

        class InsertionOrder : public std::vector<uint32_t> {
        public:
            InsertionOrder() : nOfNeighboringNodes(50), epsilon(0.1), nOfThreads(0), indegreeOrder(false) {}

            ObjectID getID(ObjectID id) {
                if (id > size()) {
                    std::stringstream msg;
                    msg << "InsertionOrder::getID: Invalid ID. " << size() << ":" << id;
                    POLARIS_THROW_EX(msg);
                }
                return at(id - 1);
            }

            size_t nOfNeighboringNodes;
            float epsilon;
            size_t nOfThreads;
            bool indegreeOrder;
        };

        class InsertionResult {
        public:
            InsertionResult() : id(0), identical(false), distance(0.0) {}

            InsertionResult(size_t i, bool tf, distance_t d) : id(i), identical(tf), distance(d) {}

            size_t id;
            bool identical;
            distance_t distance; // the distance between the centroid and the inserted object.
        };

        class AccuracyTable {
        public:
            AccuracyTable() {};

            AccuracyTable(std::vector<std::pair<float, double>> &t) { set(t); }

            AccuracyTable(std::string str) { set(str); }

            void set(std::vector<std::pair<float, double>> &t) { table = t; }

            void set(std::string str) {
                std::vector<std::string> tokens;
                Common::tokenize(str, tokens, ",");
                if (tokens.size() < 2) {
                    return;
                }
                for (auto i = tokens.begin(); i != tokens.end(); ++i) {
                    std::vector<std::string> ts;
                    Common::tokenize(*i, ts, ":");
                    if (ts.size() != 2) {
                        std::stringstream msg;
                        msg << "AccuracyTable: Invalid accuracy table string " << *i << ":" << str;
                        POLARIS_THROW_EX(msg);
                    }
                    table.push_back(std::make_pair(Common::strtod(ts[0]), Common::strtod(ts[1])));
                }
            }

            float getEpsilon(double accuracy) {
                if (table.size() <= 2) {
                    std::stringstream msg;
                    msg << "AccuracyTable: The accuracy table is not set yet. The table size=" << table.size();
                    POLARIS_THROW_EX(msg);
                }
                if (accuracy > 1.0) {
                    accuracy = 1.0;
                }
                std::pair<float, double> lower, upper;
                {
                    auto i = table.begin();
                    for (; i != table.end(); ++i) {
                        if ((*i).second >= accuracy) {
                            break;
                        }
                    }
                    if (table.end() == i) {
                        i -= 2;
                    } else if (table.begin() != i) {
                        i--;
                    }
                    lower = *i++;
                    upper = *i;
                }
                float e = lower.first +
                          (upper.first - lower.first) * (accuracy - lower.second) / (upper.second - lower.second);
                if (e < -0.9) {
                    e = -0.9;
                }
                return e;
            }

            std::string getString() {
                std::stringstream str;
                for (auto i = table.begin(); i != table.end(); ++i) {
                    str << (*i).first << ":" << (*i).second;
                    if (i + 1 != table.end()) {
                        str << ",";
                    }
                }
                return str.str();
            }

            std::vector<std::pair<float, double>> table;
        };

        NgtIndex() : index(0) {
#if defined(NGT_AVX2)
            if (!CpuInfo::isAVX2()) {
                std::stringstream msg;
                msg
                        << "polaris::NgtIndex: Fatal Error!. Despite that this NGT library is built with AVX2, this CPU doesn't support AVX2. This CPU supoorts "
                        << CpuInfo::getSupportedSimdTypes();
                POLARIS_THROW_EX(msg);
            }
#elif defined(NGT_AVX512)
            if (!CpuInfo::isAVX512()) {
          std::stringstream msg;
          msg << "polaris::NgtIndex: Fatal Error!. Despite that this NGT library is built with AVX512, this CPU doesn't support AVX512. This CPU supoorts " << CpuInfo::getSupportedSimdTypes();
          POLARIS_THROW_EX(msg);
            }
#endif
        }

        NgtIndex(polaris::NgtParameters &prop);

        NgtIndex(const std::string &database, bool rdOnly = false, NgtIndex::OpenType openType = NgtIndex::OpenTypeNone) : index(
                0), redirect(false) { open(database, rdOnly, openType); }

        NgtIndex(const std::string &database, polaris::NgtParameters &prop) : index(0), redirect(false) { open(database, prop); }

        virtual ~NgtIndex() { close(); }

        void open(const std::string &database, polaris::NgtParameters &prop) {
            open(database);
            setProperty(prop);
        }

        void open(const std::string &database, bool rdOnly = false, NgtIndex::OpenType openType = OpenTypeNone);

        void close() {
            if (index != 0) {
                delete index;
                index = 0;
            }
            path.clear();
        }

        void save() {
            if (path.empty()) {
                POLARIS_THROW_EX("polaris::NgtIndex::saveIndex: path is empty");
            }
            saveIndex(path);
        }

        void save(std::string indexPath) {
            saveIndex(indexPath);
        }

        static void mkdir(const std::string &dir) {
            if (::mkdir(dir.c_str(), S_IRWXU | S_IRGRP | S_IXGRP | S_IROTH | S_IXOTH) != 0) {
                std::stringstream msg;
                msg << "polaris::NgtIndex::mkdir: Cannot make the specified directory. " << dir;
                POLARIS_THROW_EX(msg);
            }
        }

        static void create(const std::string &database, polaris::NgtParameters &prop, bool redirect = false) {
            createGraphAndTree(database, prop, redirect);
        }

        static void createGraphAndTree(const std::string &database, polaris::NgtParameters &prop, const std::string &dataFile,
                                       size_t dataSize = 0, bool redirect = false);

        static void createGraphAndTree(const std::string &database, polaris::NgtParameters &prop,
                                       bool redirect = false) { createGraphAndTree(database, prop, "", redirect); }

        static void
        createGraph(const std::string &database, polaris::NgtParameters &prop, const std::string &dataFile, size_t dataSize = 0,
                    bool redirect = false);

        template<typename T>
        size_t insert(const std::vector<T> &object);

        template<typename T>
        size_t append(const std::vector<T> &object);

        template<typename T>
        void update(ObjectID id, const std::vector<T> &object);

#ifdef NGT_REFINEMENT
        template<typename T> size_t appendToRefinement(const std::vector<T> &object);
        template<typename T> void updateToRefinement(ObjectID id, const std::vector<T> &object);
#endif

        static void
        append(const std::string &database, const std::string &dataFile, size_t threadSize, size_t dataSize);

        static void append(const std::string &database, const float *data, size_t dataSize, size_t threadSize);

        static void remove(const std::string &database, std::vector<ObjectID> &objects, bool force = false);

        static void exportIndex(const std::string &database, const std::string &file);

        static void importIndex(const std::string &database, const std::string &file);

        virtual void load(const std::string &ifile, size_t dataSize) { getIndex().load(ifile, dataSize); }

        virtual void append(const std::string &ifile, size_t dataSize) { getIndex().append(ifile, dataSize); }

        virtual void append(const float *data, size_t dataSize) {
            StdOstreamRedirector redirector(redirect);
            redirector.begin();
            try {
                getIndex().append(data, dataSize);
            } catch (polaris::PolarisException &err) {
                redirector.end();
                throw err;
            }
            redirector.end();
        }

        virtual void append(const double *data, size_t dataSize) {
            StdOstreamRedirector redirector(redirect);
            redirector.begin();
            try {
                getIndex().append(data, dataSize);
            } catch (polaris::PolarisException &err) {
                redirector.end();
                throw err;
            }
            redirector.end();
        }

        virtual void append(const uint8_t *data, size_t dataSize) {
            StdOstreamRedirector redirector(redirect);
            redirector.begin();
            try {
                getIndex().append(data, dataSize);
            } catch (polaris::PolarisException &err) {
                redirector.end();
                throw err;
            }
            redirector.end();
        }

        virtual void append(const float16 *data, size_t dataSize) {
            StdOstreamRedirector redirector(redirect);
            redirector.begin();
            try {
                getIndex().append(data, dataSize);
            } catch (polaris::PolarisException &err) {
                redirector.end();
                throw err;
            }
            redirector.end();
        }

        virtual size_t getNumberOfObjects() { return getIndex().getNumberOfObjects(); }

        virtual size_t getNumberOfIndexedObjects() { return getIndex().getNumberOfIndexedObjects(); }

        virtual size_t getObjectRepositorySize() { return getIndex().getObjectRepositorySize(); }

        virtual size_t getGraphRepositorySize() { return getIndex().getGraphRepositorySize(); }

        void createIndex(size_t threadNumber = 0, size_t sizeOfRepository = 0);

        virtual void createIndexWithInsertionOrder(InsertionOrder &insertionOrder, size_t threadNumber = 0,
                                                   size_t sizeOfRepository = 0) {
            StdOstreamRedirector redirector(redirect);
            redirector.begin();
            try {
                getIndex().createIndexWithInsertionOrder(insertionOrder, threadNumber, sizeOfRepository);
            } catch (polaris::PolarisException &err) {
                redirector.end();
                throw err;
            }
            redirector.end();
        }

        virtual void saveIndex(const std::string &ofile) { getIndex().saveIndex(ofile); }

        virtual void loadIndex(const std::string &ofile) { getIndex().loadIndex(ofile); }

        virtual Object *allocateObject(const std::string &textLine, const std::string &sep) {
            return getIndex().allocateObject(textLine, sep);
        }

        virtual Object *allocateObject(const std::vector<double> &obj) { return getIndex().allocateObject(obj); }

        virtual Object *allocateObject(const std::vector<float> &obj) { return getIndex().allocateObject(obj); }

        virtual Object *allocateObject(const std::vector<uint8_t> &obj) { return getIndex().allocateObject(obj); }

        virtual Object *allocateObject(const std::vector<float16> &obj) { return getIndex().allocateObject(obj); }

        virtual Object *allocateObject(const float *obj, size_t size) { return getIndex().allocateObject(obj, size); }

        virtual size_t getSizeOfElement() { return getIndex().getSizeOfElement(); }

        virtual void setProperty(polaris::NgtParameters &prop) { getIndex().setProperty(prop); }

        virtual void getProperty(polaris::NgtParameters &prop) { getIndex().getProperty(prop); }

        virtual void deleteObject(Object *po) { getIndex().deleteObject(po); }

        virtual void linearSearch(polaris::SearchContainer &sc) { getIndex().linearSearch(sc); }

        virtual void linearSearch(polaris::SearchQuery &sc) { getIndex().linearSearch(sc); }

        virtual void search(polaris::SearchContainer &sc) { getIndex().search(sc); }

        virtual void search(polaris::SearchQuery &sc) { getIndex().search(sc); }

        virtual void search(polaris::SearchContainer &sc, ObjectDistances &seeds) { getIndex().search(sc, seeds); }

        virtual void getSeeds(polaris::SearchContainer &sc, ObjectDistances &seeds, size_t n) {
            getIndex().getSeeds(sc, seeds, n);
        }

        virtual void remove(ObjectID id, bool force = false) { getIndex().remove(id, force); }

        virtual void exportIndex(const std::string &file) { getIndex().exportIndex(file); }

        virtual void importIndex(const std::string &file) { getIndex().importIndex(file); }

        virtual bool verify(std::vector<uint8_t> &status, bool info = false, char mode = '-') {
            return getIndex().verify(status, info, mode);
        }

        virtual ObjectSpace &getObjectSpace() { return getIndex().getObjectSpace(); }

#ifdef NGT_REFINEMENT
        virtual ObjectSpace &getRefinementObjectSpace() { return getIndex().getRefinementObjectSpace(); }
#endif

        virtual size_t getSharedMemorySize(std::ostream &os,
                                           SharedMemoryAllocator::GetMemorySizeType t = SharedMemoryAllocator::GetTotalMemorySize) {
            size_t osize = 0;
            os << "object=" << osize << std::endl;
            size_t isize = getIndex().getSharedMemorySize(os, t);
            return osize + isize;
        }

        float getEpsilonFromExpectedAccuracy(double accuracy);

        void searchUsingOnlyGraph(polaris::SearchContainer &sc);

        void searchUsingOnlyGraph(polaris::SearchQuery &searchQuery);

        std::vector<float> makeSparseObject(std::vector<uint32_t> &object);

        NgtIndex &getIndex() {
            if (index == 0) {
                POLARIS_THROW_EX("polaris::NgtIndex::getIndex: NgtIndex is unavailable.");
            }
            return *index;
        }

        void enableLog() { redirect = false; }

        void disableLog() { redirect = true; }

        void extractInsertionOrder(InsertionOrder &insertionOrder);

        static void destroy(const std::string &path) {
            std::remove(std::string(path + "/grp").c_str());
            std::remove(std::string(path + "/tre").c_str());
            std::remove(std::string(path + "/obj").c_str());
            std::remove(std::string(path + "/prf").c_str());
            std::remove(path.c_str());
        }

        static void version(std::ostream &os);

        static std::string getVersion();

        std::string getPath() { return path; }

        size_t getDimension();

    protected:
        Object *allocateQuery(polaris::QueryContainer &queryContainer) {
            auto *vec = queryContainer.getQuery();
            if (vec == 0) {
                std::stringstream msg;
                msg << "polaris::NgtIndex::allocateObject: Object is not set. ";
                POLARIS_THROW_EX(msg);
            }
            Object *object = 0;
            auto &objectType = queryContainer.getQueryType();
            if (objectType == typeid(float)) {
                object = allocateObject(*static_cast<std::vector<float> *>(vec));
            } else if (objectType == typeid(double)) {
                object = allocateObject(*static_cast<std::vector<double> *>(vec));
            } else if (objectType == typeid(uint8_t)) {
                object = allocateObject(*static_cast<std::vector<uint8_t> *>(vec));
            } else if (objectType == typeid(float16)) {
                object = allocateObject(*static_cast<std::vector<float16> *>(vec));
            } else {
                std::stringstream msg;
                msg << "polaris::NgtIndex::allocateObject: Unavailable object type.";
                POLARIS_THROW_EX(msg);
            }
            return object;
        }

        static void loadAndCreateIndex(NgtIndex &index, const std::string &database, const std::string &dataFile,
                                       size_t threadSize, size_t dataSize);

        NgtIndex *index;
        std::string path;
        bool redirect;
    };

    class GraphIndex : public NgtIndex,
                       public NeighborhoodGraph {
    public:

        GraphIndex(const std::string &database, bool rdOnly = false, NgtIndex::OpenType openType = NgtIndex::OpenTypeNone);

        GraphIndex(polaris::NgtParameters &prop) : readOnly(false) {
            initialize(prop);
        }

        void initialize(polaris::NgtParameters &prop) {
            constructObjectSpace(prop);
            setProperty(prop);
        }


        virtual ~GraphIndex() {
            destructObjectSpace();
        }

        void constructObjectSpace(polaris::NgtParameters &prop);

        void destructObjectSpace() {
#ifdef NGT_REFINEMENT
            if (refinementObjectSpace != 0) {
          auto *os = (ObjectSpaceRepository<float, double>*)refinementObjectSpace;
          os->deleteAll();
          delete os;
              refinementObjectSpace = 0;
            }
#endif
            if (objectSpace == 0) {
                return;
            }
            if (property.object_type == polaris::ObjectType::FLOAT) {
                ObjectSpaceRepository<float, double> *os = (ObjectSpaceRepository<float, double> *) objectSpace;
                os->deleteAll();
                delete os;
            } else if (property.object_type == polaris::ObjectType::UINT8) {
                ObjectSpaceRepository<unsigned char, int> *os = (ObjectSpaceRepository<unsigned char, int> *) objectSpace;
                os->deleteAll();
                delete os;
            } else if (property.object_type == polaris::ObjectType::FLOAT16) {
                ObjectSpaceRepository<float16, float> *os = (ObjectSpaceRepository<float16, float> *) objectSpace;
                os->deleteAll();
                delete os;
            } else {
                std::cerr << "Cannot find Object Type in the property. " << property.object_type << std::endl;
                return;
            }
            objectSpace = 0;
        }

        virtual void load(const std::string &ifile, size_t dataSize = 0) {
            if (ifile.empty()) {
                return;
            }
            std::istream *is;
            std::ifstream *ifs = 0;
            if (ifile == "-") {
                is = &std::cin;
            } else {
                ifs = new std::ifstream;
                ifs->std::ifstream::open(ifile);
                if (!(*ifs)) {
                    std::stringstream msg;
                    msg << "NgtIndex::load: Cannot open the specified file. " << ifile;
                    POLARIS_THROW_EX(msg);
                }
                is = ifs;
            }
            try {
                objectSpace->readText(*is, dataSize);
            } catch (polaris::PolarisException &err) {
                if (ifile != "-") {
                    delete ifs;
                }
                throw (err);
            }
            if (ifile != "-") {
                delete ifs;
            }
        }

        virtual void append(const std::string &ifile, size_t dataSize = 0) {
            if (ifile.empty()) {
                return;
            }
            std::istream *is;
            std::ifstream *ifs = 0;
            if (ifile == "-") {
                is = &std::cin;
            } else {
                ifs = new std::ifstream;
                ifs->std::ifstream::open(ifile);
                if (!(*ifs)) {
                    std::stringstream msg;
                    msg << "NgtIndex::load: Cannot open the specified file. " << ifile;
                    POLARIS_THROW_EX(msg);
                }
                is = ifs;
            }
            try {
                objectSpace->appendText(*is, dataSize);
            } catch (polaris::PolarisException &err) {
                if (ifile != "-") {
                    delete ifs;
                }
                throw (err);
            }
            if (ifile != "-") {
                delete ifs;
            }
        }

        virtual void append(const float *data, size_t dataSize) { objectSpace->append(data, dataSize); }

        virtual void append(const double *data, size_t dataSize) { objectSpace->append(data, dataSize); }

        virtual void append(const uint8_t *data, size_t dataSize) { objectSpace->append(data, dataSize); }

        virtual void append(const float16 *data, size_t dataSize) { objectSpace->append(data, dataSize); }

        void saveObjectRepository(const std::string &ofile) {
            try {
                mkdir(ofile);
            } catch (...) {}
            if (objectSpace != 0) {
                objectSpace->serialize(ofile + "/obj");
            } else {
                std::cerr << "saveIndex::Warning! ObjectSpace is null. continue saving..." << std::endl;
            }
#ifdef NGT_REFINEMENT
            if (refinementObjectSpace != 0) {
          refinementObjectSpace->serialize(ofile + "/robj");
            }
#endif
        }

        void saveGraph(const std::string &ofile) {
            std::string fname = ofile + "/grp";
            std::ofstream osg(fname);
            if (!osg.is_open()) {
                std::stringstream msg;
                msg << "saveIndex:: Cannot open. " << fname;
                POLARIS_THROW_EX(msg);
            }
            repository.serialize(osg);
        }

        virtual void saveIndex(const std::string &ofile) {
            saveObjectRepository(ofile);
            saveGraph(ofile);
            saveProperty(ofile);
        }

        void saveProperty(const std::string &file);

        void exportProperty(const std::string &file);

        static void loadGraph(const std::string &ifile, polaris::GraphRepository &graph);

        virtual void loadIndex(const std::string &ifile, bool readOnly, polaris::NgtIndex::OpenType openType);

        virtual void exportIndex(const std::string &ofile) {
            try {
                mkdir(ofile);
            } catch (...) {
                std::stringstream msg;
                msg << "exportIndex:: Cannot make the directory. " << ofile;
                POLARIS_THROW_EX(msg);
            }
            objectSpace->serializeAsText(ofile + "/obj");
            std::ofstream osg(ofile + "/grp");
            repository.serializeAsText(osg);
            exportProperty(ofile);
        }

        virtual void importIndex(const std::string &ifile) {
            objectSpace->deserializeAsText(ifile + "/obj");
            std::string fname = ifile + "/grp";
            std::ifstream isg(fname);
            if (!isg.is_open()) {
                std::stringstream msg;
                msg << "importIndex:: Cannot open. " << fname;
                POLARIS_THROW_EX(msg);
            }
            repository.deserializeAsText(isg);
        }

        void linearSearch(polaris::SearchContainer &sc) {
            ObjectSpace::ResultSet results;
            objectSpace->linearSearch(sc.object, sc.radius, sc.size, results);
            ObjectDistances &qresults = sc.getResult();
            qresults.moveFrom(results);
        }

        void linearSearch(polaris::SearchQuery &searchQuery) {
            Object *query = NgtIndex::allocateQuery(searchQuery);
            try {
                polaris::SearchContainer sc(searchQuery, *query);
                GraphIndex::linearSearch(sc);
                searchQuery.distanceComputationCount = sc.distanceComputationCount;
                searchQuery.visitCount = sc.visitCount;
            } catch (polaris::PolarisException &err) {
                deleteObject(query);
                throw err;
            }
            deleteObject(query);
        }

        // GraphIndex
        virtual void search(polaris::SearchContainer &sc) {
            sc.distanceComputationCount = 0;
            sc.visitCount = 0;
            ObjectDistances seeds;
            search(sc, seeds);
        }

        // GraphIndex
        void search(polaris::SearchQuery &searchQuery) {
            Object *query = NgtIndex::allocateQuery(searchQuery);
            try {
                polaris::SearchContainer sc(searchQuery, *query);
#ifdef NGT_REFINEMENT
                auto expansion = searchQuery.getRefinementExpansion();
                if (expansion < 1.0) {
                  GraphIndex::search(sc);
                  searchQuery.workingResult = std::move(sc.workingResult);
                } else {
                  size_t poffset = 12;
                  size_t psize = 64;
                  auto size = sc.size;
                  sc.size *= expansion;
                  try {
                    GraphIndex::search(sc);
                  } catch(polaris::PolarisException &err) {
                    sc.size = size;
                    throw err;
                  }
                  auto &ros = getRefinementObjectSpace();
                  auto &rrepo = ros.getRepository();
                  polaris::Object *robject = 0;
                  if (searchQuery.getQueryType() == typeid(float)) {
                    auto &v = *static_cast<std::vector<float>*>(searchQuery.getQuery());
                    robject = ros.allocateNormalizedObject(v);
                  } else if (searchQuery.getQueryType() == typeid(uint8_t)) {
                    auto &v = *static_cast<std::vector<uint8_t>*>(searchQuery.getQuery());
                    robject = ros.allocateNormalizedObject(v);
                  } else if (searchQuery.getQueryType() == typeid(float16)) {
                    auto &v = *static_cast<std::vector<float16>*>(searchQuery.getQuery());
                    robject = ros.allocateNormalizedObject(v);
                  } else {
                    std::stringstream msg;
                    msg << "Invalid query object type.";
                    POLARIS_THROW_EX(msg);
                  }
                  sc.size = size;
                  auto &comparator = getRefinementObjectSpace().getComparator();
                  if (sc.resultIsAvailable()) {
                    auto &results = sc.getResult();
                    for (auto &r : results) {
                      r.distance = comparator(*robject, *rrepo.get(r.id));
                    }
                    std::sort(results.begin(), results.end());
                    results.resize(size);
                  } else {
                    ObjectDistances rs;
                    rs.resize(sc.workingResult.size());
                        size_t counter = 0;
                        while (!sc.workingResult.empty()) {
                          if (counter < poffset) {
                            auto *ptr = rrepo.get(sc.workingResult.top().id)->getPointer();
                    MemoryCache::prefetch(static_cast<uint8_t*>(ptr), psize);
                          }
                          rs[counter++].id = sc.workingResult.top().id;
                          sc.workingResult.pop();
                        }
                        for (size_t idx = 0; idx < rs.size(); idx++) {
                          if (idx + poffset < rs.size()) {
                            auto *ptr = rrepo.get(rs[idx + poffset].id)->getPointer();
                    MemoryCache::prefetch(static_cast<uint8_t*>(ptr), psize);
                          }
                          auto &r = rs[idx];
                          r.distance = comparator(*robject, *rrepo.get(r.id));
                      searchQuery.workingResult.emplace(r);
                        }
                    while (searchQuery.workingResult.size() > sc.size) { searchQuery.workingResult.pop(); }
                  }
                  ros.deleteObject(robject);
                }
#else
                GraphIndex::search(sc);
                searchQuery.workingResult = std::move(sc.workingResult);
#endif
                searchQuery.distanceComputationCount = sc.distanceComputationCount;
                searchQuery.visitCount = sc.visitCount;
            } catch (polaris::PolarisException &err) {
                deleteObject(query);
                throw err;
            }
            deleteObject(query);
        }

        void getSeeds(polaris::SearchContainer &sc, ObjectDistances &seeds, size_t n) {
            getRandomSeeds(repository, seeds, n);
            setupDistances(sc, seeds);
            std::sort(seeds.begin(), seeds.end());
            if (seeds.size() > n) {
                seeds.resize(n);
            }
        }

        // get randomly nodes as seeds.
        template<class REPOSITORY>
        void getRandomSeeds(REPOSITORY &repo, ObjectDistances &seeds, size_t seedSize) {
            // clear all distances to find the same object as a randomized object.
            for (ObjectDistances::iterator i = seeds.begin(); i != seeds.end(); i++) {
                (*i).distance = 0.0;
            }
            size_t repositorySize = repo.size();
            repositorySize = repositorySize == 0 ? 0 : repositorySize - 1; // Because the head of repository is a dummy.
            seedSize = seedSize > repositorySize ? repositorySize : seedSize;
            std::vector<ObjectID> deteted;
            size_t emptyCount = 0;
            while (seedSize > seeds.size()) {
                double random = ((double) rand() + 1.0) / ((double) RAND_MAX + 2.0);
                size_t idx = floor(repositorySize * random) + 1;
                if (repo.isEmpty(idx)) {
                    emptyCount++;
                    if (emptyCount > repositorySize) {
                        break;
                    }
                    continue;
                }
                ObjectDistance obj(idx, 0.0);
                if (find(seeds.begin(), seeds.end(), obj) != seeds.end()) {
                    continue;
                }
                seeds.push_back(obj);
            }
        }

        void remove(const ObjectID id, bool force) {
            if (!NeighborhoodGraph::repository.isEmpty(id)) {
                removeEdgesReliably(id);
            }
            try {
                getObjectRepository().remove(id);
            } catch (polaris::PolarisException &err) {
                std::cerr << "polaris::GraphIndex::remove:: cannot remove from feature. id=" << id << " " << err.what()
                          << std::endl;
                throw err;
            }
        }

        virtual void searchForNNGInsertion(Object &po, ObjectDistances &result) {
            polaris::SearchContainer sc(po);
            sc.setResults(&result);
            sc.size = NeighborhoodGraph::property.edgeSizeForCreation;
            sc.radius = FLT_MAX;
            sc.explorationCoefficient = NeighborhoodGraph::property.insertionRadiusCoefficient;
            try {
                GraphIndex::search(sc);
            } catch (polaris::PolarisException &err) {
                throw err;
            }
            if (static_cast<int>(result.size()) < NeighborhoodGraph::property.edgeSizeForCreation &&
                result.size() < repository.size()) {
                if (sc.edgeSize != 0) {
                    sc.edgeSize = 0;    // not prune edges.
                    try {
                        GraphIndex::search(sc);
                    } catch (polaris::PolarisException &err) {
                        throw err;
                    }
                }
            }
        }

        void searchForKNNGInsertion(Object &po, ObjectID id, ObjectDistances &result) {
            double radius = FLT_MAX;
            size_t size = NeighborhoodGraph::property.edgeSizeForCreation;
            if (id > 0) {
                size = NeighborhoodGraph::property.edgeSizeForCreation + 1;
            }
            ObjectSpace::ResultSet rs;
            objectSpace->linearSearch(po, radius, size, rs);
            result.moveFrom(rs, id);
            if ((size_t) NeighborhoodGraph::property.edgeSizeForCreation != result.size()) {
                std::cerr << "searchForKNNGInsert::Warning! inconsistency of the sizes. ID=" << id
                          << " " << NeighborhoodGraph::property.edgeSizeForCreation << ":" << result.size()
                          << std::endl;
                for (size_t i = 0; i < result.size(); i++) {
                    std::cerr << result[i].id << ":" << result[i].distance << " ";
                }
                std::cerr << std::endl;
            }
        }

        virtual void insert(
                ObjectID id
        ) {
            ObjectRepository &fr = objectSpace->getRepository();
            if (fr[id] == 0) {
                std::cerr << "NGTIndex::insert empty " << id << std::endl;
                return;
            }
            Object &po = *fr[id];
            ObjectDistances rs;
            if (NeighborhoodGraph::property.graphType == GraphType::GraphTypeANNG ||
                NeighborhoodGraph::property.graphType == GraphType::GraphTypeIANNG ||
                NeighborhoodGraph::property.graphType == GraphType::GraphTypeRANNG) {
                searchForNNGInsertion(po, rs);
            } else {
                searchForKNNGInsertion(po, id, rs);
            }
            insertNode(id, rs);
        }

        virtual void createIndexWithSingleThread();

        virtual void createIndexWithInsertionOrder(InsertionOrder &insertionOrder, size_t threadNumber = 0,
                                                   size_t sizeOfRepository = 0);

        void checkGraph() {
            GraphRepository &repo = repository;
            ObjectRepository &fr = objectSpace->getRepository();
            for (size_t id = 0; id < fr.size(); id++) {
                if (repo[id] == 0) {
                    std::cerr << id << " empty" << std::endl;
                    continue;
                }
                if ((id % 10000) == 0) {
                    std::cerr << "checkGraph: Processed size=" << id << std::endl;
                }
                Object &po = *fr[id];
                GraphNode *objects = getNode(id);

                ObjectDistances rs;
                NeighborhoodGraph::property.edgeSizeForCreation = objects->size() + 1;
                searchForNNGInsertion(po, rs);

                if (rs.size() != objects->size()) {
                    std::cerr << "Cannot get the specified number of the results. " << rs.size() << ":"
                              << objects->size() << std::endl;
                }
                size_t count = 0;
                ObjectDistances::iterator rsi = rs.begin();
                for (GraphNode::iterator ri = objects->begin();
                     ri != objects->end() && rsi != rs.end();) {
                    if ((*ri).distance == (*rsi).distance && (*ri).id == (*rsi).id) {
                        count++;
                        ri++;
                        rsi++;
                    } else if ((*ri).distance < (*rsi).distance) {
                        ri++;
                    } else {
                        rsi++;
                    }
                }
                if (count != objects->size()) {
                    std::cerr << "id=" << id << " identities=" << count << " " << objects->size() << " " << rs.size()
                              << std::endl;
                }
            }
        }

        virtual bool verify(std::vector<uint8_t> &status, bool info) {
            bool valid = true;
            std::cerr << "Started verifying graph and objects" << std::endl;
            GraphRepository &repo = repository;
            ObjectRepository &fr = objectSpace->getRepository();
            if (repo.size() != fr.size()) {
                if (info) {
                    std::cerr << "Warning! # of nodes is different from # of objects. " << repo.size() << ":"
                              << fr.size() << std::endl;
                }
            }
            status.clear();
            status.resize(fr.size(), 0);
            for (size_t id = 1; id < fr.size(); id++) {
                status[id] |= repo[id] != 0 ? 0x02 : 0x00;
                status[id] |= fr[id] != 0 ? 0x01 : 0x00;
            }
            for (size_t id = 1; id < fr.size(); id++) {
                if (fr[id] == 0) {
                    if (id < repo.size() && repo[id] != 0) {
                        std::cerr << "Error! The node exists in the graph, but the object does not exist. " << id
                                  << std::endl;
                        valid = false;
                    }
                }
                if (fr[id] != 0 && repo[id] == 0) {
                    std::cerr << "Error. No." << id << " is not registerd in the graph." << std::endl;
                    valid = false;
                }
                if ((id % 1000000) == 0) {
                    std::cerr << "  verified " << id << " entries." << std::endl;
                }
                if (fr[id] != 0) {
                    try {
                        Object *po = fr[id];
                        if (po == 0) {
                            std::cerr << "Error! Cannot get the object. " << id << std::endl;
                            valid = false;
                            continue;
                        }
                    } catch (polaris::PolarisException &err) {
                        std::cerr << "Error! Cannot get the object. " << id << " " << err.what() << std::endl;
                        valid = false;
                        continue;
                    }
                }
                if (id >= repo.size()) {
                    std::cerr << "Error. No." << id << " is not registerd in the object repository. " << repo.size()
                              << std::endl;
                    valid = false;
                }
                if (id < repo.size() && repo[id] != 0) {
                    try {
                        GraphNode *objects = getNode(id);
                        if (objects == 0) {
                            std::cerr << "Error! Cannot get the node. " << id << std::endl;
                            valid = false;
                        }
                        for (GraphNode::iterator ri = objects->begin();
                             ri != objects->end(); ++ri) {
                            for (GraphNode::iterator rj = objects->begin() + std::distance(objects->begin(), ri);
                                 rj != objects->end(); ++rj) {
                                if ((*ri).id == (*rj).id &&
                                    std::distance(objects->begin(), ri) != std::distance(objects->begin(), rj)) {
                                    std::cerr << "Error! More than two identical objects! ID=" << (*rj).id << " idx="
                                              << std::distance(objects->begin(), ri) << ":"
                                              << std::distance(objects->begin(), rj)
                                              << " disntace=" << (*ri).distance << ":" << (*rj).distance << std::endl;
                                    valid = false;
                                }
                            }

                            if ((*ri).id == 0 || (*ri).id >= repo.size()) {
                                std::cerr << "Error! Neighbor's ID of the node is out of range. ID=" << id << std::endl;
                                valid = false;
                            } else if (repo[(*ri).id] == 0) {
                                std::cerr << "Error! The neighbor ID of the node is invalid. ID=" << id
                                          << " Invalid ID=" << (*ri).id << std::endl;
                                if (fr[(*ri).id] == 0) {
                                    std::cerr << "The neighbor doesn't exist in the object repository as well. ID="
                                              << (*ri).id << std::endl;
                                } else {
                                    std::cerr << "The neighbor exists in the object repository. ID=" << (*ri).id
                                              << std::endl;
                                }
                                valid = false;
                            }
                            if ((*ri).distance < 0.0) {
                                std::cerr << "Error! Neighbor's distance is munus. ID=" << id << std::endl;
                                valid = false;
                            }
                        }
                    } catch (polaris::PolarisException &err) {
                        std::cerr << "Error! Cannot get the node. " << id << " " << err.what() << std::endl;
                        valid = false;
                    }
                }
            }
            return valid;
        }

        void extractSparseness(InsertionOrder &insertionOrder);

        void extractInsertionOrder(InsertionOrder &insertionOrder);

        static bool showStatisticsOfGraph(polaris::GraphIndex &outGraph, char mode = '-', size_t edgeSize = UINT_MAX);

        size_t getNumberOfObjects() { return objectSpace->getRepository().count(); }

        size_t getNumberOfIndexedObjects() {
            ObjectRepository &repo = objectSpace->getRepository();
            GraphRepository &graphRepo = repository;
            size_t count = 0;
            for (polaris::ObjectID id = 1; id < repo.size() && id < graphRepo.size(); id++) {
                if (repo[id] != 0 && graphRepo[id] != 0) {
                    count++;
                }
            }
            return count;
        }

        size_t getObjectRepositorySize() { return objectSpace->getRepository().size(); }

        size_t getGraphRepositorySize() {
#ifdef NGT_GRAPH_READ_ONLY_GRAPH
            return std::max(repository.size(), searchRepository.size());
#else
            return repository.size();
#endif
        }

        size_t getSizeOfElement() { return objectSpace->getSizeOfElement(); }

        Object *allocateObject(const std::string &textLine, const std::string &sep) {
            return objectSpace->allocateNormalizedObject(textLine, sep);
        }

        Object *allocateObject(const std::vector<double> &obj) {
            return objectSpace->allocateNormalizedObject(obj);
        }

        Object *allocateObject(const std::vector<float> &obj) {
            return objectSpace->allocateNormalizedObject(obj);
        }

        Object *allocateObject(const std::vector<float16> &obj) {
            return objectSpace->allocateNormalizedObject(obj);
        }

        Object *allocateObject(const std::vector<uint8_t> &obj) {
            return objectSpace->allocateNormalizedObject(obj);
        }

        Object *allocateObject(const float *obj, size_t size) {
            return objectSpace->allocateNormalizedObject(obj, size);
        }

        void deleteObject(Object *po) {
            return objectSpace->deleteObject(po);
        }

        ObjectSpace &getObjectSpace() { return *objectSpace; }

#ifdef NGT_REFINEMENT
        ObjectSpace &getRefinementObjectSpace() { return *refinementObjectSpace; }
#endif

        void setupPrefetch(polaris::NgtParameters &prop);

        void setProperty(polaris::NgtParameters &prop) {
            setupPrefetch(prop);
            GraphIndex::property.set(prop);
            NeighborhoodGraph::property.set(prop);
            assert(property.dimension != 0);
            accuracyTable.set(property.accuracyTable);
        }

        void getProperty(polaris::NgtParameters &prop) {
            GraphIndex::property.get(prop);
            NeighborhoodGraph::property.get(prop);
        }

        NgtGraphParameters &getGraphProperty() { return NeighborhoodGraph::property; }

        NgtIndexParameters &getGraphIndexProperty() { return GraphIndex::property; }

        virtual size_t getSharedMemorySize(std::ostream &os, SharedMemoryAllocator::GetMemorySizeType t) {
            size_t size = 0;
            os << "graph=" << size << std::endl;
            return size;
        }

        float getEpsilonFromExpectedAccuracy(double accuracy) { return accuracyTable.getEpsilon(accuracy); }

        NgtIndexParameters &getProperty() { return property; }

        bool getReadOnly() { return readOnly; }

        template<class REPOSITORY>
        void getSeedsFromGraph(REPOSITORY &repo, ObjectDistances &seeds) {
            if (repo.size() != 0) {
                size_t seedSize = repo.size() - 1 < (size_t) NeighborhoodGraph::property.seedSize ?
                                  repo.size() - 1 : (size_t) NeighborhoodGraph::property.seedSize;
                if (NeighborhoodGraph::property.seedType == SeedType::SeedTypeRandomNodes ||
                    NeighborhoodGraph::property.seedType == SeedType::SeedTypeNone) {
                    getRandomSeeds(repo, seeds, seedSize);
                } else if (NeighborhoodGraph::property.seedType == SeedType::SeedTypeFixedNodes) {
                    // To check speed using fixed seeds.
                    for (size_t i = 1; i <= seedSize; i++) {
                        ObjectDistance obj(i, 0.0);
                        seeds.push_back(obj);
                    }
                } else if (NeighborhoodGraph::property.seedType == SeedType::SeedTypeFirstNode) {
                    ObjectDistance obj(1, 0.0);
                    seeds.push_back(obj);
                } else {
                    getRandomSeeds(repo, seeds, seedSize);
                }
            }
        }

    protected:

        // GraphIndex
        virtual void search(polaris::SearchContainer &sc, ObjectDistances &seeds) {
            if (sc.size == 0) {
                while (!sc.workingResult.empty()) sc.workingResult.pop();
                return;
            }
            if (seeds.size() == 0) {
#if !defined(NGT_GRAPH_READ_ONLY_GRAPH)
                getSeedsFromGraph(repository, seeds);
#else
                if (readOnly) {
                    getSeedsFromGraph(searchRepository, seeds);
                } else {
                    getSeedsFromGraph(repository, seeds);
                }
#endif
            }
            if (sc.expectedAccuracy > 0.0) {
                sc.setEpsilon(getEpsilonFromExpectedAccuracy(sc.expectedAccuracy));
            }

            try {
                if (readOnly) {
#if !defined(NGT_GRAPH_READ_ONLY_GRAPH)
                    NeighborhoodGraph::search(sc, seeds);
#else
                    (*searchUnupdatableGraph)(*this, sc, seeds);
#endif
                } else {
                    NeighborhoodGraph::search(sc, seeds);
                }
            } catch (polaris::PolarisException &err) {
                polaris::PolarisException e(err);
                throw e;
            }
        }

    public:
        NgtIndexParameters property;

    protected:
        bool readOnly;
#ifdef NGT_GRAPH_READ_ONLY_GRAPH

        void (*searchUnupdatableGraph)(polaris::NeighborhoodGraph &, polaris::SearchContainer &, polaris::ObjectDistances &);

#endif

        NgtIndex::AccuracyTable accuracyTable;
    };

    class GraphAndTreeIndex : public GraphIndex, public DVPTree {
    public:

        GraphAndTreeIndex(const std::string &database, bool rdOnly = false) : GraphIndex(database, rdOnly) {
            GraphAndTreeIndex::loadIndex(database, rdOnly);
        }

        GraphAndTreeIndex(polaris::NgtParameters &prop) : GraphIndex(prop) {
            DVPTree::objectSpace = GraphIndex::objectSpace;
        }

        virtual ~GraphAndTreeIndex() {}

        void create() {}

        void alignObjects() {
            polaris::ObjectSpace &space = getObjectSpace();
            polaris::ObjectRepository &repo = space.getRepository();
            Object **object = repo.getPtr();
            std::vector<bool> exist(repo.size(), false);
            std::vector<polaris::Node::ID> leafNodeIDs;
            DVPTree::getAllLeafNodeIDs(leafNodeIDs);
            size_t objectCount = 0;
            for (size_t i = 0; i < leafNodeIDs.size(); i++) {
                ObjectDistances objects;
                DVPTree::getObjectIDsFromLeaf(leafNodeIDs[i], objects);
                for (size_t j = 0; j < objects.size(); j++) {
                    exist[objects[j].id] = true;
                    objectCount++;
                }
            }
            std::multimap<uint32_t, uint32_t> notexist;
            if (objectCount != repo.size()) {
                for (size_t id = 1; id < exist.size(); id++) {
                    if (!exist[id]) {
                        DVPTree::SearchContainer tso(*object[id]);
                        tso.mode = DVPTree::SearchContainer::SearchLeaf;
                        tso.radius = 0.0;
                        tso.size = 1;
                        try {
                            DVPTree::search(tso);
                        } catch (polaris::PolarisException &err) {
                            std::stringstream msg;
                            msg << "GraphAndTreeIndex::getSeeds: Cannot search for tree.:" << err.what();
                            POLARIS_THROW_EX(msg);
                        }
                        notexist.insert(std::pair<uint32_t, uint32_t>(tso.nodeID.getID(), id));
                        objectCount++;
                    }
                }
            }
            assert(objectCount == repo.size() - 1);

            objectCount = 1;
            std::vector<std::pair<uint32_t, uint32_t> > order;
            for (size_t i = 0; i < leafNodeIDs.size(); i++) {
                ObjectDistances objects;
                DVPTree::getObjectIDsFromLeaf(leafNodeIDs[i], objects);
                for (size_t j = 0; j < objects.size(); j++) {
                    order.push_back(std::pair<uint32_t, uint32_t>(objects[j].id, objectCount));
                    objectCount++;
                }
                auto nei = notexist.equal_range(leafNodeIDs[i].getID());
                for (auto ii = nei.first; ii != nei.second; ++ii) {
                    order.push_back(std::pair<uint32_t, uint32_t>((*ii).second, objectCount));
                    objectCount++;
                }
            }
            assert(objectCount == repo.size());
            Object *tmp = space.allocateObject();
            std::unordered_set<uint32_t> uncopiedObjects;
            for (size_t i = 1; i < repo.size(); i++) {
                uncopiedObjects.insert(i);
            }
            size_t copycount = 0;
            while (!uncopiedObjects.empty()) {
                size_t startID = *uncopiedObjects.begin();
                if (startID == order[startID - 1].first) {
                    uncopiedObjects.erase(startID);
                    copycount++;
                    continue;
                }
                size_t id = startID;
                space.copy(*tmp, *object[id]);
                uncopiedObjects.erase(id);
                do {
                    space.copy(*object[id], *object[order[id - 1].first]);
                    copycount++;
                    id = order[id - 1].first;
                    uncopiedObjects.erase(id);
                } while (order[id - 1].first != startID);
                space.copy(*object[id], *tmp);
                copycount++;
            }
            space.deleteObject(tmp);

            assert(copycount == repo.size() - 1);

            sort(order.begin(), order.end());
            uncopiedObjects.clear();
            for (size_t i = 1; i < repo.size(); i++) {
                uncopiedObjects.insert(i);
            }
            copycount = 0;
            Object *tmpPtr;
            while (!uncopiedObjects.empty()) {
                size_t startID = *uncopiedObjects.begin();
                if (startID == order[startID - 1].second) {
                    uncopiedObjects.erase(startID);
                    copycount++;
                    continue;
                }
                size_t id = startID;
                tmpPtr = object[id];
                uncopiedObjects.erase(id);
                do {
                    object[id] = object[order[id - 1].second];
                    copycount++;
                    id = order[id - 1].second;
                    uncopiedObjects.erase(id);
                } while (order[id - 1].second != startID);
                object[id] = tmpPtr;
                copycount++;
            }
            assert(copycount == repo.size() - 1);
        }

        void load(const std::string &ifile) {
            GraphIndex::load(ifile);
            DVPTree::objectSpace = GraphIndex::objectSpace;
        }

        void saveIndex(const std::string &ofile) {
            GraphIndex::saveIndex(ofile);
            std::string fname = ofile + "/tre";
            std::ofstream ost(fname);
            if (!ost.is_open()) {
                std::stringstream msg;
                msg << "saveIndex:: Cannot open. " << fname;
                POLARIS_THROW_EX(msg);
            }
            DVPTree::serialize(ost);
        }

        void loadIndex(const std::string &ifile, bool readOnly) {
            DVPTree::objectSpace = GraphIndex::objectSpace;
            std::ifstream ist(ifile + "/tre");
            DVPTree::deserialize(ist);
#ifdef NGT_GRAPH_READ_ONLY_GRAPH
            if (property.objectAlignment == polaris::ObjectAlignmentTrue) {
                alignObjects();
            }
#endif
        }

        void exportIndex(const std::string &ofile) {
            GraphIndex::exportIndex(ofile);
            std::ofstream ost(ofile + "/tre");
            DVPTree::serializeAsText(ost);
        }

        void importIndex(const std::string &ifile) {
            std::string fname = ifile + "/tre";
            std::ifstream ist(fname);
            if (!ist.is_open()) {
                std::stringstream msg;
                msg << "importIndex:: Cannot open. " << fname;
                POLARIS_THROW_EX(msg);
            }
            DVPTree::deserializeAsText(ist);
            GraphIndex::importIndex(ifile);
        }

        void remove(const ObjectID id, bool force = false) {
            Object *obj = 0;
            try {
                obj = GraphIndex::objectSpace->getRepository().get(id);
            } catch (polaris::PolarisException &err) {
                if (force) {
                    try {
                        DVPTree::removeNaively(id);
                    } catch (...) {}
                    try {
                        GraphIndex::remove(id, force);
                    } catch (...) {}
                    std::stringstream msg;
                    msg << err.what()
                        << " Even though the object could not be found, the object could be removed from the tree and graph if it existed in them.";
                    POLARIS_THROW_EX(msg);
                }
                throw err;
            }
            if (NeighborhoodGraph::repository.isEmpty(id)) {
                if (force) {
                    try {
                        DVPTree::removeNaively(id);
                    } catch (...) {}
                }
                GraphIndex::remove(id, force);
                return;
            }
            polaris::SearchContainer so(*obj);
            ObjectDistances results;
            so.setResults(&results);
            so.id = 0;
            so.size = 2;
            so.radius = 0.0;
            so.explorationCoefficient = 1.1;
            ObjectDistances seeds;
            seeds.push_back(ObjectDistance(id, 0.0));
            GraphIndex::search(so, seeds);
            if (results.size() == 0) {
                if (!GraphIndex::objectSpace->isNormalizedDistance()) {
                    std::stringstream msg;
                    msg << "Not found the specified id. ID=" << id;
                    POLARIS_THROW_EX(msg);
                }
                so.radius = FLT_MAX;
                so.size = 10;
                results.clear();
                GraphIndex::search(so, seeds);
                for (size_t i = 0; i < results.size(); i++) {
                    try {
                        auto *robj = GraphIndex::objectSpace->getRepository().get(results[i].id);
                        results[i].distance = GraphIndex::objectSpace->compareWithL1(*obj, *robj);
                    } catch (polaris::PolarisException &err) {
                        std::stringstream msg;
                        msg << "remove: Fatal Inner Error! Cannot get an object. ID=" << id;
                        POLARIS_THROW_EX(msg);
                    }
                }
                std::sort(results.begin(), results.end());
                results.resize(2);
                for (auto i = results.begin(); i != results.end(); ++i) {
                    if ((*i).distance != 0.0) {
                        results.resize(distance(results.begin(), i));
                        break;
                    }
                }
                if (results.size() == 0) {
                    std::stringstream msg;
                    msg << "Not found the specified id. ID=" << id;
                    POLARIS_THROW_EX(msg);
                }
            }
            if (results.size() == 1) {
                try {
                    DVPTree::remove(id);
                } catch (polaris::PolarisException &err) {
                    std::stringstream msg;
                    msg << "remove:: cannot remove from tree. id=" << id << " " << err.what();
                    POLARIS_THROW_EX(msg);
                }
            } else {
                ObjectID replaceID = id == results[0].id ? results[1].id : results[0].id;
                try {
                    DVPTree::replace(id, replaceID);
                } catch (polaris::PolarisException &err) {
                }
            }
            GraphIndex::remove(id, force);
        }

        void searchForNNGInsertion(Object &po, ObjectDistances &result) {
            polaris::SearchContainer sc(po);
            sc.setResults(&result);
            sc.size = NeighborhoodGraph::property.edgeSizeForCreation;
            sc.radius = FLT_MAX;
            sc.explorationCoefficient = NeighborhoodGraph::property.insertionRadiusCoefficient;
            sc.useAllNodesInLeaf = true;
            try {
                GraphAndTreeIndex::search(sc);
            } catch (polaris::PolarisException &err) {
                throw err;
            }
            if (static_cast<int>(result.size()) < NeighborhoodGraph::property.edgeSizeForCreation &&
                result.size() < repository.size()) {
                if (sc.edgeSize != 0) {
                    try {
                        GraphAndTreeIndex::search(sc);
                    } catch (polaris::PolarisException &err) {
                        throw err;
                    }
                }
            }
        }

        void insert(ObjectID id) {
            ObjectRepository &fr = GraphIndex::objectSpace->getRepository();
            if (fr[id] == 0) {
                std::cerr << "GraphAndTreeIndex::insert empty " << id << std::endl;
                return;
            }
            Object &po = *fr[id];
            ObjectDistances rs;
            if (NeighborhoodGraph::property.graphType == GraphType::GraphTypeANNG ||
                NeighborhoodGraph::property.graphType == GraphType::GraphTypeIANNG ||
                NeighborhoodGraph::property.graphType == GraphType::GraphTypeRANNG) {
                searchForNNGInsertion(po, rs);
            } else {
                searchForKNNGInsertion(po, id, rs);
            }

            GraphIndex::insertNode(id, rs);

            if (((rs.size() > 0) && (rs[0].distance != 0.0)) || rs.size() == 0) {
                DVPTree::InsertContainer tiobj(po, id);
                try {
                    DVPTree::insert(tiobj);
                } catch (polaris::PolarisException &err) {
                    std::cerr << "GraphAndTreeIndex::insert: Fatal error" << std::endl;
                    std::cerr << err.what() << std::endl;
                    return;
                }
            }
        }

        void createIndexWithInsertionOrder(InsertionOrder &insertionOrder, size_t threadNumber = 0,
                                           size_t sizeOfRepository = 0);

        void
        createIndex(const std::vector<std::pair<polaris::Object *, size_t> > &objects, std::vector<InsertionResult> &ids,
                    float range, size_t threadNumber);

        void createTreeIndex();

        void getSeeds(polaris::SearchContainer &sc, ObjectDistances &seeds, size_t n) {
            DVPTree::SearchContainer tso(sc.object);
            tso.mode = DVPTree::SearchContainer::SearchLeaf;
            tso.radius = 0.0;
            tso.size = 1;
            tso.distanceComputationCount = 0;
            tso.visitCount = 0;
            try {
                DVPTree::search(tso);
            } catch (polaris::PolarisException &err) {
                std::stringstream msg;
                msg << "GraphAndTreeIndex::getSeeds: Cannot search for tree.:" << err.what();
                POLARIS_THROW_EX(msg);
            }
            try {
                DVPTree::getObjectIDsFromLeaf(tso.nodeID, seeds);
            } catch (polaris::PolarisException &err) {
                std::stringstream msg;
                msg << "GraphAndTreeIndex::getSeeds: Cannot get a leaf.:" << err.what();
                POLARIS_THROW_EX(msg);
            }
            sc.distanceComputationCount += tso.distanceComputationCount;
            sc.visitCount += tso.visitCount;
            if (seeds.size() < n) {
                GraphIndex::getRandomSeeds(repository, seeds, n);
            }
            GraphIndex::setupDistances(sc, seeds);
            std::sort(seeds.begin(), seeds.end());
            if (seeds.size() > n) {
                seeds.resize(n);
            }
        }

        // GraphAndTreeIndex
        void getSeedsFromTree(polaris::SearchContainer &sc, ObjectDistances &seeds) {
            DVPTree::SearchContainer tso(sc.object);
            tso.mode = DVPTree::SearchContainer::SearchLeaf;
            tso.radius = 0.0;
            tso.size = 1;
            tso.distanceComputationCount = 0;
            tso.visitCount = 0;
            try {
                DVPTree::search(tso);
            } catch (polaris::PolarisException &err) {
                std::stringstream msg;
                msg << "GraphAndTreeIndex::getSeeds: Cannot search for tree.:" << err.what();
                POLARIS_THROW_EX(msg);
            }

            try {
                DVPTree::getObjectIDsFromLeaf(tso.nodeID, seeds);
            } catch (polaris::PolarisException &err) {
                std::stringstream msg;
                msg << "GraphAndTreeIndex::getSeeds: Cannot get a leaf.:" << err.what();
                POLARIS_THROW_EX(msg);
            }
            sc.distanceComputationCount += tso.distanceComputationCount;
            sc.visitCount += tso.visitCount;
            if (sc.useAllNodesInLeaf ||
                NeighborhoodGraph::property.seedType == SeedType::SeedTypeAllLeafNodes) {
                return;
            }
            // if seedSize is zero, the result size of the query is used as seedSize.
            size_t seedSize =
                    NeighborhoodGraph::property.seedSize == 0 ? sc.size : NeighborhoodGraph::property.seedSize;
            seedSize = seedSize > sc.size ? sc.size : seedSize;
            if (seeds.size() > seedSize) {
                srand(tso.nodeID.getID());
                // to accelerate thinning data.
                for (size_t i = seeds.size(); i > seedSize; i--) {
                    double random = ((double) rand() + 1.0) / ((double) RAND_MAX + 2.0);
                    size_t idx = floor(i * random);
                    seeds[idx] = seeds[i - 1];
                }
                seeds.resize(seedSize);
            } else if (seeds.size() < seedSize) {
                // A lack of the seeds is compansated by random seeds.
                //getRandomSeeds(seeds, seedSize);
            }
        }

        // GraphAndTreeIndex
        void search(polaris::SearchContainer &sc) {
            sc.distanceComputationCount = 0;
            sc.visitCount = 0;
            ObjectDistances seeds;
            getSeedsFromTree(sc, seeds);
            sc.visitCount = sc.distanceComputationCount;
            GraphIndex::search(sc, seeds);
        }

        // GraphAndTreeIndex
        void search(polaris::SearchQuery &searchQuery) {
            Object *query = NgtIndex::allocateQuery(searchQuery);
            try {
                polaris::SearchContainer sc(searchQuery, *query);
#ifdef NGT_REFINEMENT
                auto expansion = searchQuery.getRefinementExpansion();
                if (expansion < 1.0) {
                  GraphAndTreeIndex::search(sc);
                  searchQuery.workingResult = std::move(sc.workingResult);
                } else {
                  size_t poffset = 12;
                  size_t psize = 64;
                  auto size = sc.size;
                  sc.size *= expansion;
                  try {
                    GraphAndTreeIndex::search(sc);
                  } catch(polaris::PolarisException &err) {
                    sc.size = size;
                    throw err;
                  }
                  auto &ros = getRefinementObjectSpace();
                  auto &rrepo = ros.getRepository();
                  polaris::Object *robject = 0;
                  if (searchQuery.getQueryType() == typeid(float)) {
                    auto &v = *static_cast<std::vector<float>*>(searchQuery.getQuery());
                    robject = ros.allocateNormalizedObject(v);
                  } else if (searchQuery.getQueryType() == typeid(uint8_t)) {
                    auto &v = *static_cast<std::vector<uint8_t>*>(searchQuery.getQuery());
                    robject = ros.allocateNormalizedObject(v);
                  } else if (searchQuery.getQueryType() == typeid(float16)) {
                    auto &v = *static_cast<std::vector<float16>*>(searchQuery.getQuery());
                    robject = ros.allocateNormalizedObject(v);
                  } else {
                    std::stringstream msg;
                    msg << "Invalid query object type.";
                    POLARIS_THROW_EX(msg);
                  }
                  sc.size = size;
                  auto &comparator = getRefinementObjectSpace().getComparator();
                  if (sc.resultIsAvailable()) {
                    auto &results = sc.getResult();
                    for (auto &r : results) {
                      r.distance = comparator(*robject, *rrepo.get(r.id));
                    }
                    std::sort(results.begin(), results.end());
                    results.resize(size);
                  } else {
                    ObjectDistances rs;
                    rs.resize(sc.workingResult.size());
                    size_t counter = 0;
                    while (!sc.workingResult.empty()) {
                      if (counter < poffset) {
                    auto *ptr = rrepo.get(sc.workingResult.top().id)->getPointer();
                    MemoryCache::prefetch(static_cast<uint8_t*>(ptr), psize);
                      }
                      rs[counter++].id = sc.workingResult.top().id;
                      sc.workingResult.pop();
                    }
                    for (size_t idx = 0; idx < rs.size(); idx++) {
                      if (idx + poffset < rs.size()) {
                    auto *ptr = rrepo.get(rs[idx + poffset].id)->getPointer();
                    MemoryCache::prefetch(static_cast<uint8_t*>(ptr), psize);
                      }
                      auto &r = rs[idx];
                      r.distance = comparator(*robject, *rrepo.get(r.id));
                      searchQuery.workingResult.emplace(r);
                    }
                    while (searchQuery.workingResult.size() > sc.size) { searchQuery.workingResult.pop(); }
                  }
                  ros.deleteObject(robject);
                }
#else
                GraphAndTreeIndex::search(sc);
                searchQuery.workingResult = std::move(sc.workingResult);
#endif
                searchQuery.distanceComputationCount = sc.distanceComputationCount;
                searchQuery.visitCount = sc.visitCount;
            } catch (polaris::PolarisException &err) {
                deleteObject(query);
                throw err;
            }
            deleteObject(query);
        }

        size_t getSharedMemorySize(std::ostream &os, SharedMemoryAllocator::GetMemorySizeType t) {
            return GraphIndex::getSharedMemorySize(os, t) + DVPTree::getSharedMemorySize(os, t);
        }

        bool verify(std::vector<uint8_t> &status, bool info, char mode);

    };

} // namespace polaris

template<typename T>
size_t polaris::NgtIndex::append(const std::vector<T> &object) {
    auto &os = getObjectSpace();
    auto &repo = os.getRepository();
    if (repo.size() == 0) {
        repo.initialize();
    }

    auto *o = repo.allocateNormalizedPersistentObject(object);
    repo.push_back(dynamic_cast<PersistentObject *>(o));
    size_t oid = repo.size() - 1;
    return oid;
}

#ifdef NGT_REFINEMENT
template<typename T>
size_t polaris::NgtIndex::appendToRefinement(const std::vector<T> &object)
{
  auto &os = getRefinementObjectSpace();
  auto &repo = os.getRepository();
  if (repo.size() == 0) {
    repo.initialize();
  }

  auto *o = repo.allocateNormalizedPersistentObject(object);
  repo.push_back(dynamic_cast<PersistentObject*>(o));
  size_t oid = repo.size() - 1;
  return oid;
}
#endif

template<typename T>
size_t polaris::NgtIndex::insert(const std::vector<T> &object) {
    auto &os = getObjectSpace();
    auto &repo = os.getRepository();
    if (repo.size() == 0) {
        repo.initialize();
    }

    auto *o = repo.allocateNormalizedPersistentObject(object);
    size_t oid = repo.insert(dynamic_cast<PersistentObject *>(o));
    return oid;
}

template<typename T>
void polaris::NgtIndex::update(ObjectID id, const std::vector<T> &object) {
    auto &os = getObjectSpace();
    auto &repo = os.getRepository();

    Object *obj = 0;
    try {
        obj = repo.get(id);
    } catch (polaris::PolarisException &err) {
        std::stringstream msg;
        msg << "Invalid ID. " << id;
        POLARIS_THROW_EX(msg);
    }
    repo.setObject(*obj, object);
    return;
}

#ifdef NGT_REFINEMENT
template<typename T>
  void polaris::NgtIndex::updateToRefinement(ObjectID id, const std::vector<T> &object)
{
  auto &os = getRefinementObjectSpace();
  auto &repo = os.getRepository();

  Object *obj = 0;
  try {
    obj = repo.get(id);
  } catch (polaris::PolarisException &err) {
    std::stringstream msg;
    msg << "Invalid ID. " << id;
    POLARIS_THROW_EX(msg);
  }
  repo.setObject(*obj, object);
  return;
}
#endif
