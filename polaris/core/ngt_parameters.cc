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

#include <polaris/core/ngt_parameters.h>

namespace polaris {

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

    void NgtGraphParameters::set(polaris::NgtParameters &prop) {
        if (prop.truncationThreshold != -1) truncationThreshold = prop.truncationThreshold;
        if (prop.edgeSizeForCreation != -1) edgeSizeForCreation = prop.edgeSizeForCreation;
        if (prop.edgeSizeForSearch != -1) edgeSizeForSearch = prop.edgeSizeForSearch;
        if (prop.edgeSizeLimitForCreation != -1) edgeSizeLimitForCreation = prop.edgeSizeLimitForCreation;
        if (prop.insertionRadiusCoefficient != -1) insertionRadiusCoefficient = prop.insertionRadiusCoefficient;
        if (prop.seedSize != -1) seedSize = prop.seedSize;
        if (prop.seedType != SeedTypeNone) seedType = prop.seedType;
        if (prop.truncationThreadPoolSize != -1) truncationThreadPoolSize = prop.truncationThreadPoolSize;
        if (prop.batchSizeForCreation != -1) batchSizeForCreation = prop.batchSizeForCreation;
        if (prop.dynamicEdgeSizeBase != -1) dynamicEdgeSizeBase = prop.dynamicEdgeSizeBase;
        if (prop.dynamicEdgeSizeRate != -1) dynamicEdgeSizeRate = prop.dynamicEdgeSizeRate;
        if (prop.buildTimeLimit != -1) buildTimeLimit = prop.buildTimeLimit;
        if (prop.outgoingEdge != -1) outgoingEdge = prop.outgoingEdge;
        if (prop.incomingEdge != -1) incomingEdge = prop.incomingEdge;
        if (prop.graphType != GraphTypeNone) graphType = prop.graphType;
    }

    void NgtGraphParameters::get(polaris::NgtParameters &prop) const {
        prop.truncationThreshold = truncationThreshold;
        prop.edgeSizeForCreation = edgeSizeForCreation;
        prop.edgeSizeForSearch = edgeSizeForSearch;
        prop.edgeSizeLimitForCreation = edgeSizeLimitForCreation;
        prop.insertionRadiusCoefficient = insertionRadiusCoefficient;
        prop.seedSize = seedSize;
        prop.seedType = seedType;
        prop.truncationThreadPoolSize = truncationThreadPoolSize;
        prop.batchSizeForCreation = batchSizeForCreation;
        prop.dynamicEdgeSizeBase = dynamicEdgeSizeBase;
        prop.dynamicEdgeSizeRate = dynamicEdgeSizeRate;
        prop.graphType = graphType;
        prop.buildTimeLimit = buildTimeLimit;
        prop.outgoingEdge = outgoingEdge;
        prop.incomingEdge = incomingEdge;
    }

    turbo::Status NgtGraphParameters::export_property(polaris::PropertySet &p) const {
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
            POLARIS_LOG(ERROR)<< "Graph::exportProperty: Fatal error! Invalid Graph Type. " << NAMEOF_ENUM(graphType);
            return rs;
        }
        rs = PropertySerializer::seed_type_export(p, seedType);
        if (!rs.ok()) {
            POLARIS_LOG(ERROR)<< "Graph::exportProperty: Fatal error! Invalid Seed Type. " << NAMEOF_ENUM(seedType);
            return rs;
        }
        return turbo::ok_status();
    }

    [[nodiscard]] turbo::Status NgtGraphParameters::import_property(polaris::PropertySet &p) {
        set_default();
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
            POLARIS_LOG(ERROR)<< "Graph::importProperty: Fatal error! Invalid Graph Type. " << grs.status().message();
            return grs.status();
        }
        graphType = (GraphType) grs.value();
        auto srs = PropertySerializer::seed_type_import(p);
        if (!srs.ok()) {
            POLARIS_LOG(ERROR)<< "Graph::importProperty: Fatal error! Invalid Seed Type. " << srs.status().message();
            return srs.status();
        }
        seedType = (SeedType) srs.value();
        return turbo::ok_status();
    }

    [[nodiscard]] turbo::Status NgtIndexParameters::export_property(polaris::PropertySet &p) const {
        PropertySerializer::dimension_export(p, dimension);
        p.set("ThreadPoolSize", threadPoolSize);
        static std::set<ObjectType> allow_set = {ObjectType::FLOAT, ObjectType::UINT8, ObjectType::FLOAT16,
                                                 ObjectType::BFLOAT16};
        auto rs = PropertySerializer::object_type_export(p, objectType, &allow_set);
        if(!rs.ok()) {
            POLARIS_LOG(ERROR) << rs.message();
            return rs;
        }
#ifdef NGT_REFINEMENT
        static std::set<ObjectType> refinement_allow_set = {ObjectType::FLOAT, ObjectType::UINT8, ObjectType::FLOAT16, ObjectType::BFLOAT16};
                auto rrs = PropertySerializer::object_type_export(p, refinementObjectType, &refinement_allow_set);
                if(!rrs.ok()) {
                    POLARIS_LOG(ERROR) << rs.message();
                    return rrs;
                }
#endif
        static std::set<MetricType> allow_dis_set = {MetricType::METRIC_NONE, MetricType::METRIC_L1, MetricType::METRIC_L2,
                                                     MetricType::METRIC_HAMMING, MetricType::METRIC_JACCARD, MetricType::METRIC_SPARSE_JACCARD,
                                                     MetricType::METRIC_ANGLE, MetricType::METRIC_COSINE, MetricType::METRIC_NORMALIZED_ANGLE,
                                                     MetricType::METRIC_NORMALIZED_COSINE, MetricType::METRIC_NORMALIZED_L2, MetricType::METRIC_INNER_PRODUCT,
                                                     MetricType::METRIC_POINCARE, MetricType::METRIC_LORENTZ};
        rs = PropertySerializer::distance_type_export(p, distanceType, &allow_dis_set);
        if(!rs.ok()) {
            POLARIS_LOG(ERROR) << rs.message();
            return rs;
        }

        rs = PropertySerializer::index_type_export(p, indexType);
        if(!rs.ok()) {
            POLARIS_LOG(ERROR) << rs.message();
            return rs;
        }

        rs = PropertySerializer::database_type_export(p, databaseType);
        if(!rs.ok()) {
            POLARIS_LOG(ERROR) << rs.message();
            return rs;
        }
        rs = PropertySerializer::object_alignment_export(p, objectAlignment);
        if(!rs.ok()) {
            POLARIS_LOG(ERROR) << rs.message();
            return rs;
        }
        p.set("PathAdjustmentInterval", pathAdjustmentInterval);
        p.set("PrefetchOffset", prefetchOffset);
        p.set("PrefetchSize", prefetchSize);
        p.set("AccuracyTable", accuracyTable);
        p.set("MaxMagnitude", maxMagnitude);
        p.set("NumberOfNeighborsForInsertionOrder", nOfNeighborsForInsertionOrder);
        p.set("EpsilonForInsertionOrder", epsilonForInsertionOrder);
        return turbo::ok_status();
    }

    [[nodiscard]] turbo::Status NgtIndexParameters::import_property(polaris::PropertySet &p) {
        set_default();
        auto dmrs = PropertySerializer::dimension_import(p);
        if(!dmrs.ok()) {
            POLARIS_LOG(ERROR) << dmrs.status().message();
            dimension = 0;
        } else {
            dimension = dmrs.value();
        }
        threadPoolSize = p.getl("ThreadPoolSize", threadPoolSize);

        static std::set<ObjectType> allow_object_set = {ObjectType::FLOAT, ObjectType::UINT8, ObjectType::FLOAT16, ObjectType::BFLOAT16};
        auto otrs = PropertySerializer::object_type_import(p, &allow_object_set);
        if(!otrs.ok()) {
            POLARIS_LOG(ERROR) << otrs.status().message();
            objectType = ObjectType::ObjectTypeNone;
        }
        objectType = otrs.value();
#ifdef NGT_REFINEMENT
        {
                    static std::set<ObjectType> allow_refinement_object_set = {ObjectType::FLOAT, ObjectType::UINT8, ObjectType::FLOAT16, ObjectType::BFLOAT16};
                    auto rtrs = PropertySerializer::object_type_import(p, &allow_refinement_object_set);
                    if(!rtrs.ok()) {
                        POLARIS_LOG(ERROR) << rtrs.status().message();
                        refinementObjectType = ObjectType::ObjectTypeNone;
                    }
                    refinementObjectType = rtrs.value();
                }
#endif
        static std::set<MetricType> allow_dis_set = {MetricType::METRIC_NONE, MetricType::METRIC_L1, MetricType::METRIC_L2,
                                                     MetricType::METRIC_HAMMING, MetricType::METRIC_JACCARD, MetricType::METRIC_SPARSE_JACCARD,
                                                     MetricType::METRIC_ANGLE, MetricType::METRIC_COSINE, MetricType::METRIC_NORMALIZED_ANGLE,
                                                     MetricType::METRIC_NORMALIZED_COSINE, MetricType::METRIC_NORMALIZED_L2, MetricType::METRIC_INNER_PRODUCT,
                                                     MetricType::METRIC_POINCARE, MetricType::METRIC_LORENTZ};
        auto drs = PropertySerializer::distance_type_import(p, &allow_dis_set);
        if(!drs.ok()) {
            POLARIS_LOG(ERROR) << drs.status().message();
            distanceType = MetricType::METRIC_NONE;
        }
        distanceType = drs.value();
        auto irs = PropertySerializer::index_type_import(p);
        if(!irs.ok()) {
            POLARIS_LOG(ERROR) << irs.status().message();
            indexType = IndexType::INDEX_NONE;
        } else {
            indexType = irs.value();
        }
        std::set<DatabaseType> allow_database_set = {DatabaseType::Memory, DatabaseType::MemoryMappedFile};
        auto dbtrs = PropertySerializer::database_type_import(p, &allow_database_set);
        if(!dbtrs.ok()) {
            POLARIS_LOG(ERROR) << dbtrs.status().message();
            databaseType = DatabaseType::DatabaseTypeNone;
        } else {
            databaseType = dbtrs.value();
        }
        auto alrs = PropertySerializer::object_alignment_import(p);
        if(!alrs.ok()) {
            POLARIS_LOG(ERROR) << alrs.status().message();
            objectAlignment = ObjectAlignment::ObjectAlignmentFalse;
        } else {
            objectAlignment = alrs.value();
        }
        pathAdjustmentInterval = p.getl("PathAdjustmentInterval", pathAdjustmentInterval);
        prefetchOffset = p.getl("PrefetchOffset", prefetchOffset);
        prefetchSize = p.getl("PrefetchSize", prefetchSize);
        auto it = p.find("AccuracyTable");
        if (it != p.end()) {
            accuracyTable = it->second;
        }
        it = p.find("SearchType");
        if (it != p.end()) {
            searchType = it->second;
        }
        maxMagnitude = p.getf("MaxMagnitude", maxMagnitude);
        nOfNeighborsForInsertionOrder = p.getl("NumberOfNeighborsForInsertionOrder",
                                               nOfNeighborsForInsertionOrder);
        epsilonForInsertionOrder = p.getf("EpsilonForInsertionOrder", epsilonForInsertionOrder);
        return turbo::ok_status();
    }


    void
    NgtIndexParameters::set(polaris::NgtParameters &prop) {
        if (prop.dimension != -1) dimension = prop.dimension;
        if (prop.threadPoolSize != -1) threadPoolSize = prop.threadPoolSize;
        if (prop.objectType != ObjectType::ObjectTypeNone) objectType = prop.objectType;
#ifdef NGT_REFINEMENT
        if (prop.refinementObjectType != ObjectSpace::ObjectTypeNone) refinementObjectType = prop.refinementObjectType;
#endif
        if (prop.distanceType != MetricType::METRIC_NONE) distanceType = prop.distanceType;
        if (prop.indexType != IndexType::INDEX_NONE) indexType = prop.indexType;
        if (prop.databaseType != DatabaseTypeNone) databaseType = prop.databaseType;
        if (prop.objectAlignment != ObjectAlignmentNone) objectAlignment = prop.objectAlignment;
        if (prop.pathAdjustmentInterval != -1) pathAdjustmentInterval = prop.pathAdjustmentInterval;
        if (prop.prefetchOffset != -1) prefetchOffset = prop.prefetchOffset;
        if (prop.prefetchSize != -1) prefetchSize = prop.prefetchSize;
        if (prop.accuracyTable != "") accuracyTable = prop.accuracyTable;
        if (prop.maxMagnitude != -1) maxMagnitude = prop.maxMagnitude;
        if (prop.nOfNeighborsForInsertionOrder != -1) nOfNeighborsForInsertionOrder = prop.nOfNeighborsForInsertionOrder;
        if (prop.epsilonForInsertionOrder != -1) epsilonForInsertionOrder = prop.epsilonForInsertionOrder;
    }

    void
    NgtIndexParameters::get(polaris::NgtParameters &prop) const {
        prop.dimension = dimension;
        prop.threadPoolSize = threadPoolSize;
        prop.objectType = objectType;
#ifdef NGT_REFINEMENT
        prop.refinementObjectType = refinementObjectType;
#endif
        prop.distanceType = distanceType;
        prop.indexType = indexType;
        prop.databaseType = databaseType;
        prop.pathAdjustmentInterval = pathAdjustmentInterval;
        prop.prefetchOffset = prefetchOffset;
        prop.prefetchSize = prefetchSize;
        prop.accuracyTable = accuracyTable;
        prop.maxMagnitude = maxMagnitude;
        prop.nOfNeighborsForInsertionOrder = nOfNeighborsForInsertionOrder;
        prop.epsilonForInsertionOrder = epsilonForInsertionOrder;
    }


}  // namespace polaris
