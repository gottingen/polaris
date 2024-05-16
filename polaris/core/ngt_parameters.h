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
#include <polaris/core/log.h>
#include <collie/meta/reflect.h>


#ifndef NGT_SEED_SIZE
#define    NGT_SEED_SIZE                10
#endif

namespace polaris {

    struct NgtParameters;

    struct NgtGraphParameters {
    public:
        NgtGraphParameters() { set_default(); }

        void set_default() {
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

        void set(polaris::NgtParameters &prop);

        void get(polaris::NgtParameters &prop) const;

        [[nodiscard]] turbo::Status export_property(polaris::PropertySet &p) const;

        [[nodiscard]] turbo::Status import_property(polaris::PropertySet &p);

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

    std::ostream &operator<<(std::ostream &os, const NgtGraphParameters &p) {
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


    class NgtIndexParameters {
    public:

        NgtIndexParameters() { set_default(); }

        void set_default() {
            dimension = 0;
            threadPoolSize = 32;
            objectType = ObjectType::FLOAT;
#ifdef NGT_REFINEMENT
            refinementObjectType	= ObjectType::FLOAT;
#endif
            distanceType = MetricType::METRIC_L2;
            indexType = IndexType::INDEX_NGT_GRAPH_AND_TREE;
            objectAlignment = ObjectAlignment::ObjectAlignmentFalse;
            pathAdjustmentInterval = 0;
            databaseType = DatabaseType::Memory;
            prefetchOffset = 0;
            prefetchSize = 0;
            maxMagnitude = 0.0;
            nOfNeighborsForInsertionOrder = 0;
            epsilonForInsertionOrder = 0.1;
        }

        void clear() {
            dimension = -1;
            threadPoolSize = -1;
            objectType = ObjectType::ObjectTypeNone;
#ifdef NGT_REFINEMENT
            refinementObjectType	= ObjectSpace::ObjectTypeNone;
#endif
            distanceType = MetricType::METRIC_NONE;
            indexType = IndexType::INDEX_NONE;
            databaseType = DatabaseTypeNone;
            objectAlignment = ObjectAlignment::ObjectAlignmentNone;
            pathAdjustmentInterval = -1;
            prefetchOffset = -1;
            prefetchSize = -1;
            accuracyTable = "";
            maxMagnitude = -1;
            nOfNeighborsForInsertionOrder = -1;
            epsilonForInsertionOrder = -1;
        }

        [[nodiscard]] turbo::Status export_property(polaris::PropertySet &p) const;

        [[nodiscard]] turbo::Status import_property(polaris::PropertySet &p);

        void set(polaris::NgtParameters &prop);

        void get(polaris::NgtParameters &prop) const;

        int dimension;
        int threadPoolSize;
        polaris::ObjectType objectType;
        MetricType distanceType;
        IndexType indexType;
        DatabaseType databaseType;
        ObjectAlignment objectAlignment;
        int pathAdjustmentInterval;
        int prefetchOffset;
        int prefetchSize;
        std::string accuracyTable;
        std::string searchType;    // test
        float maxMagnitude;
        int nOfNeighborsForInsertionOrder;
        float epsilonForInsertionOrder;
#ifdef NGT_REFINEMENT
        polaris::ObjectType	refinementObjectType;
#endif
    };

    struct NgtParameters : public NgtIndexParameters, public NgtGraphParameters {
    public:
        void setDefault() {
            NgtIndexParameters::set_default();
            NgtGraphParameters::set_default();
        }

        void clear() {
            NgtIndexParameters::clear();
            NgtGraphParameters::clear();
        }

        void set(polaris::NgtParameters &p) {
            NgtIndexParameters::set(p);
            NgtGraphParameters::set(p);
        }

        void load(const std::string &file) {
            polaris::PropertySet prop;
            prop.load(file + "/prf");
            NgtIndexParameters::import_property(prop);
            NgtGraphParameters::import_property(prop);
        }

        void save(const std::string &file) {
            polaris::PropertySet prop;
            NgtIndexParameters::export_property(prop);
            NgtGraphParameters::export_property(prop);
            prop.save(file + "/prf");
        }

        void importProperty(const std::string &file) {
            polaris::PropertySet prop;
            prop.load(file + "/prf");
            NgtIndexParameters::import_property(prop);
            NgtGraphParameters::import_property(prop);
        }
        /*static void save(GraphIndex &graphIndex, const std::string &file) {
            polaris::PropertySet prop;
            graphIndex.getGraphIndexProperty().exportProperty(prop);
            graphIndex.getGraphProperty().exportProperty(prop);
            prop.save(file + "/prf");
        }
         static void exportProperty(GraphIndex &graphIndex, const std::string &file) {
            polaris::PropertySet prop;
            graphIndex.getGraphIndexProperty().exportProperty(prop);
            graphIndex.getGraphProperty().exportProperty(prop);
            prop.save(file + "/prf");
        }
         */

    };

}  // namespace polaris

