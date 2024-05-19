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

#include <polaris/core/common.h>

namespace polaris {

    const std::string PropertySerializer::OBJECT_TYPE = "ObjectType";
    const std::string PropertySerializer::DISTANCE_TYPE = "MetricType";
    const std::string PropertySerializer::DATABASE_TYPE = "DatabaseType";
    const std::string PropertySerializer::OBJECT_ALIGNMENT = "ObjectAlignment";
    const std::string PropertySerializer::GRAPH_TYPE = "GraphType";
    const std::string PropertySerializer::SEED_TYPE = "SeedType";
    const std::string PropertySerializer::INDEX_TYPE = "IndexType";
    const std::string PropertySerializer::DIMENSION = "Dimension";

    collie::Status
    PropertySerializer::object_type_export(PropertySet &ps, ObjectType type, std::set<ObjectType> *allow_set) {
        if (allow_set != nullptr && allow_set->find(type) == allow_set->end()) {
            return collie::Status::invalid_argument("Invalid object type");
        }
        switch (type) {
            case ObjectType::INT8:
                ps.set(OBJECT_TYPE, "Integer-1");
                break;
            case polaris::ObjectType::UINT8:
                ps.set(OBJECT_TYPE, "UInteger-1");
                break;
            case polaris::ObjectType::INT16:
                ps.set(OBJECT_TYPE, "Integer-2");
                break;
            case polaris::ObjectType::UINT16:
                ps.set(OBJECT_TYPE, "UInteger-2");
                break;
            case polaris::ObjectType::INT32:
                ps.set(OBJECT_TYPE, "Integer-4");
                break;
            case polaris::ObjectType::UINT32:
                ps.set(OBJECT_TYPE, "UInteger-4");
                break;
            case polaris::ObjectType::INT64:
                ps.set(OBJECT_TYPE, "Integer-8");
                break;
            case polaris::ObjectType::UINT64:
                ps.set(OBJECT_TYPE, "UInteger-8");
                break;
            case polaris::ObjectType::FLOAT:
                ps.set(OBJECT_TYPE, "Float-4");
                break;
            case polaris::ObjectType::FLOAT16:
                ps.set(OBJECT_TYPE, "Float-2");
                break;
            case polaris::ObjectType::BFLOAT16:
                ps.set(OBJECT_TYPE, "Bfloat-2");
                break;
            case polaris::ObjectType::DOUBLE:
                ps.set(OBJECT_TYPE, "Float-8");
                break;
            default :
                return collie::Status::not_found("Unknown object type");
        }
        return collie::Status::ok_status();
    }

    collie::Status
    PropertySerializer::distance_type_export(PropertySet &ps, MetricType diss, std::set<MetricType> *allow_set) {
        if (allow_set != nullptr && allow_set->find(diss) == allow_set->end()) {
            return collie::Status::invalid_argument( "Invalid distance type");
        }
        switch (diss) {
            case MetricType::METRIC_NONE:
                ps.set(DISTANCE_TYPE, "None");
                break;
            case MetricType::METRIC_L1:
                ps.set(DISTANCE_TYPE, "L1");
                break;
            case MetricType::METRIC_L2:
                ps.set(DISTANCE_TYPE, "L2");
                break;
            case MetricType::METRIC_HAMMING:
                ps.set(DISTANCE_TYPE, "Hamming");
                break;
            case MetricType::METRIC_JACCARD:
                ps.set(DISTANCE_TYPE, "Jaccard");
                break;
            case MetricType::METRIC_SPARSE_JACCARD:
                ps.set(DISTANCE_TYPE, "SparseJaccard");
                break;
            case MetricType::METRIC_ANGLE:
                ps.set(DISTANCE_TYPE, "Angle");
                break;
            case MetricType::METRIC_COSINE:
                ps.set(DISTANCE_TYPE, "Cosine");
                break;
            case MetricType::METRIC_NORMALIZED_ANGLE:
                ps.set(DISTANCE_TYPE, "NormalizedAngle");
                break;
            case MetricType::METRIC_NORMALIZED_COSINE:
                ps.set(DISTANCE_TYPE, "NormalizedCosine");
                break;
            case MetricType::METRIC_NORMALIZED_L2:
                ps.set(DISTANCE_TYPE, "NormalizedL2");
                break;
            case MetricType::METRIC_INNER_PRODUCT:
                ps.set(DISTANCE_TYPE, "InnerProduct");
                break;
            case MetricType::METRIC_POINCARE:
                ps.set(DISTANCE_TYPE, "Poincare");
                break;  // added by Nyapicom
            case MetricType::METRIC_LORENTZ:
                ps.set(DISTANCE_TYPE, "Lorentz");
                break;  // added by Nyapicom
            default :
                return collie::Status::not_found("Unknown distance type");
        }
        return collie::Status::ok_status();
    }

    collie::Status
    PropertySerializer::database_type_export(PropertySet &ps, DatabaseType type, std::set<DatabaseType> *allow_set) {
        if (allow_set != nullptr && allow_set->find(type) == allow_set->end()) {
            return collie::Status::invalid_argument( "Invalid database type");
        }
        switch (type) {
            case DatabaseType::Memory:
                ps.set(DATABASE_TYPE, "Memory");
                break;
            case DatabaseType::MemoryMappedFile:
                ps.set(DATABASE_TYPE, "MemoryMappedFile");
                break;
            case DatabaseType::SSD:
                ps.set(DATABASE_TYPE, "SSD");
                break;
            default:
                return collie::Status::not_found("Unknown database type");
        }
        return collie::Status::ok_status();
    }

    collie::Status
    PropertySerializer::database_type_export(PropertySet &ps, const std::string &key, DatabaseType type, std::set<DatabaseType> *allow_set) {
        if (allow_set != nullptr && allow_set->find(type) == allow_set->end()) {
            return collie::Status::invalid_argument("Invalid database type");
        }
        switch (type) {
            case DatabaseType::Memory:
                ps.set(key, "Memory");
                break;
            case DatabaseType::MemoryMappedFile:
                ps.set(key, "MemoryMappedFile");
                break;
            case DatabaseType::SSD:
                ps.set(key, "SSD");
                break;
            default:
                return collie::Status::not_found("Unknown database type");
        }
        return collie::Status::ok_status();
    }

    collie::Status PropertySerializer::object_alignment_export(PropertySet &ps, ObjectAlignment type) {
        switch (type) {
            case ObjectAlignment::ObjectAlignmentNone:
                ps.set(OBJECT_ALIGNMENT, "None");
                break;
            case ObjectAlignment::ObjectAlignmentTrue:
                ps.set(OBJECT_ALIGNMENT, "True");
                break;
            case ObjectAlignment::ObjectAlignmentFalse:
                ps.set(OBJECT_ALIGNMENT, "False");
                break;
            default:
                return collie::Status::not_found("Unknown object alignment");
        }
        return collie::Status::ok_status();
    }

    collie::Result<ObjectType>
    PropertySerializer::object_type_import(const PropertySet &ps, std::set<ObjectType> *allow_set) {
        auto it = ps.find(OBJECT_TYPE);
        ObjectType object_type = ObjectType::ObjectTypeNone;
        if (it != ps.end()) {
            if (it->second == "Integer-1") {
                object_type = polaris::ObjectType::INT8;
            } else if (it->second == "UInteger-1") {
                object_type = polaris::ObjectType::UINT8;
            } else if (it->second == "Integer-2") {
                object_type = polaris::ObjectType::INT16;
            } else if (it->second == "UInteger-2") {
                object_type = polaris::ObjectType::UINT16;
            } else if (it->second == "Integer-4") {
                object_type = polaris::ObjectType::INT32;
            } else if (it->second == "UInteger-4") {
                object_type = polaris::ObjectType::UINT32;
            } else if (it->second == "Integer-8") {
                object_type = polaris::ObjectType::INT64;
            } else if (it->second == "UInteger-8") {
                object_type = polaris::ObjectType::UINT64;
            } else if (it->second == "Float-2") {
                object_type = polaris::ObjectType::FLOAT16;
            } else if (it->second == "Bfloat-2") {
                object_type = polaris::ObjectType::BFLOAT16;
            } else if (it->second == "Float-4") {
                object_type = polaris::ObjectType::FLOAT;
            } else if (it->second == "Float-8") {
                object_type = polaris::ObjectType::DOUBLE;
            } else {
                std::cerr << "Invalid Object Type in the property. " << it->first << ":" << it->second
                          << std::endl;
                return collie::Status::invalid_argument( "Invalid Object Type");
            }
        } else {
            return collie::Status::not_found("Object Type not found in the property");
        }
        if (allow_set != nullptr && allow_set->find(object_type) == allow_set->end()) {
            return collie::Status::invalid_argument( "not allow Object Type");
        }
        return object_type;
    }

    collie::Result<MetricType>
    PropertySerializer::distance_type_import(const PropertySet &ps, std::set<MetricType> *allow_set) {
        auto it = ps.find(DISTANCE_TYPE);
        MetricType distanceType = MetricType::METRIC_NONE;
        if (it != ps.end()) {
            if (it->second == "None") {
                distanceType = MetricType::METRIC_NONE;
            } else if (it->second == "L1") {
                distanceType = MetricType::METRIC_L1;
            } else if (it->second == "L2") {
                distanceType = MetricType::METRIC_L2;
            } else if (it->second == "Hamming") {
                distanceType = MetricType::METRIC_HAMMING;
            } else if (it->second == "Jaccard") {
                distanceType = MetricType::METRIC_JACCARD;
            } else if (it->second == "SparseJaccard") {
                distanceType = MetricType::METRIC_SPARSE_JACCARD;
            } else if (it->second == "Angle") {
                distanceType = MetricType::METRIC_ANGLE;
            } else if (it->second == "Cosine") {
                distanceType = MetricType::METRIC_COSINE;
            } else if (it->second == "Poincare") {  // added by Nyapicom
                distanceType = MetricType::METRIC_POINCARE;
            } else if (it->second == "Lorentz") {  // added by Nyapicom
                distanceType = MetricType::METRIC_LORENTZ;
            } else if (it->second == "NormalizedAngle") {
                distanceType = MetricType::METRIC_NORMALIZED_ANGLE;
            } else if (it->second == "NormalizedCosine") {
                distanceType = MetricType::METRIC_NORMALIZED_COSINE;
            } else if (it->second == "NormalizedL2") {
                distanceType = MetricType::METRIC_NORMALIZED_L2;
            } else if (it->second == "InnerProduct") {
                distanceType = MetricType::METRIC_INNER_PRODUCT;
            } else {
                POLARIS_LOG(ERROR) << "Invalid distance_t Type in the property. " << it->first << ":" << it->second;
                return collie::Status::invalid_argument( "Invalid distance_t Type");
            }
        } else {
            return collie::Status::not_found("distance_t Type not found in the property");
        }
        if (allow_set != nullptr && allow_set->find(distanceType) == allow_set->end()) {
            return collie::Status::invalid_argument( "not allow distance_t Type");
        }
        return distanceType;
    }

    collie::Result<DatabaseType>
    PropertySerializer::database_type_import(const PropertySet &ps, std::set<DatabaseType> *allow_set) {
        auto it = ps.find(DATABASE_TYPE);
        DatabaseType databaseType = DatabaseType::DatabaseTypeNone;
        if (it != ps.end()) {
            if (it->second == "Memory") {
                databaseType = DatabaseType::Memory;
            } else if (it->second == "MemoryMappedFile") {
                databaseType = DatabaseType::MemoryMappedFile;
            } else if (it->second == "SSD") {
                databaseType = DatabaseType::SSD;
            } else {
                POLARIS_LOG(ERROR) << "Invalid Database Type in the property. " << it->first << ":" << it->second;
                return collie::Status::invalid_argument( "Invalid Database Type");
            }
        } else {
            return collie::Status::not_found("Database Type not found in the property");
        }
        if (allow_set != nullptr && allow_set->find(databaseType) == allow_set->end()) {
            return collie::Status::invalid_argument( "not allow Database Type {}",
                                      static_cast<int>(databaseType));
        }
        return databaseType;
    }

    collie::Result<DatabaseType>
    PropertySerializer::database_type_import(const PropertySet &ps, const std::string &key, std::set<DatabaseType> *allow_set) {
        auto it = ps.find(key);
        DatabaseType databaseType = DatabaseType::DatabaseTypeNone;
        if (it != ps.end()) {
            if (it->second == "Memory") {
                databaseType = DatabaseType::Memory;
            } else if (it->second == "MemoryMappedFile") {
                databaseType = DatabaseType::MemoryMappedFile;
            } else if (it->second == "SSD") {
                databaseType = DatabaseType::SSD;
            } else {
                POLARIS_LOG(ERROR) << "Invalid Database Type in the property. " << it->first << ":" << it->second;
                return collie::Status::invalid_argument( "Invalid Database Type");
            }
        } else {
            return collie::Status::not_found("Database Type not found in the property");
        }
        if (allow_set != nullptr && allow_set->find(databaseType) == allow_set->end()) {
            return collie::Status::invalid_argument( "not allow Database Type {}",
                                      static_cast<int>(databaseType));
        }
        return databaseType;
    }

    collie::Result<ObjectAlignment> PropertySerializer::object_alignment_import(const PropertySet &ps) {
        auto it = ps.find(OBJECT_ALIGNMENT);
        ObjectAlignment objectAlignment = ObjectAlignment::ObjectAlignmentNone;
        if (it != ps.end()) {
            if (it->second == "None") {
                objectAlignment = ObjectAlignment::ObjectAlignmentNone;
            } else if (it->second == "True") {
                objectAlignment = ObjectAlignment::ObjectAlignmentTrue;
            } else if (it->second == "False") {
                objectAlignment = ObjectAlignment::ObjectAlignmentFalse;
            } else {
                POLARIS_LOG(ERROR) << "Invalid Object Alignment in the property. " << it->first << ":" << it->second;
                return collie::Status::invalid_argument( "Invalid Object Alignment");
            }
        } else {
            return collie::Status::not_found("Object Alignment not found in the property");
        }
        return objectAlignment;
    }

    collie::Status PropertySerializer::graph_type_export(PropertySet &ps, GraphType type) {
        switch (type) {
            case GraphType::GraphTypeANNG:
                ps.set(GRAPH_TYPE, "ANNG");
                break;
            case GraphType::GraphTypeKNNG:
                ps.set(GRAPH_TYPE, "KNNG");
                break;
            case GraphType::GraphTypeBKNNG:
                ps.set(GRAPH_TYPE, "BKNNG");
                break;
            case GraphType::GraphTypeONNG:
                ps.set(GRAPH_TYPE, "ONNG");
                break;
            case GraphType::GraphTypeIANNG:
                ps.set(GRAPH_TYPE, "IANNG");
                break;
            case GraphType::GraphTypeDNNG:
                ps.set(GRAPH_TYPE, "DNNG");
                break;
            case GraphType::GraphTypeRANNG:
                ps.set(GRAPH_TYPE, "RANNG");
                break;
            case GraphType::GraphTypeRIANNG:
                ps.set(GRAPH_TYPE, "RIANNG");
                break;
            case GraphType::GraphTypeHNSW:
                ps.set(GRAPH_TYPE, "HNSW");
                break;
            case GraphType::GraphTypeVAMANA:
                ps.set(GRAPH_TYPE, "VAMANA");
                break;
            default:
                return collie::Status::not_found("Unknown graph type");
        }
        return collie::Status::ok_status();
    }
    collie::Result<GraphType> PropertySerializer::graph_type_import(const PropertySet &ps) {
        auto it = ps.find(GRAPH_TYPE);
        GraphType graphType = GraphType::GraphTypeNone;
        if (it != ps.end()) {
            if (it->second == "ANNG") {
                graphType = GraphType::GraphTypeANNG;
            } else if (it->second == "KNNG") {
                graphType = GraphType::GraphTypeKNNG;
            } else if (it->second == "BKNNG") {
                graphType = GraphType::GraphTypeBKNNG;
            } else if (it->second == "ONNG") {
                graphType = GraphType::GraphTypeONNG;
            } else if (it->second == "IANNG") {
                graphType = GraphType::GraphTypeIANNG;
            } else if (it->second == "DNNG") {
                graphType = GraphType::GraphTypeDNNG;
            } else if (it->second == "RANNG") {
                graphType = GraphType::GraphTypeRANNG;
            } else if (it->second == "RIANNG") {
                graphType = GraphType::GraphTypeRIANNG;
            } else if (it->second == "HNSW") {
                graphType = GraphType::GraphTypeHNSW;
            } else if (it->second == "VAMANA") {
                graphType = GraphType::GraphTypeVAMANA;
            } else {
                POLARIS_LOG(ERROR) << "Invalid Graph Type in the property. " << it->first << ":" << it->second;
                return collie::Status::invalid_argument( "Invalid Graph Type");
            }
        } else {
            return collie::Status::not_found("Graph Type not found in the property");
        }
        return graphType;
    }


    collie::Status PropertySerializer::seed_type_export(PropertySet &ps, SeedType type) {
        switch (type) {
            case SeedType::SeedTypeNone:
                ps.set(SEED_TYPE, "None");
                break;
            case SeedType::SeedTypeRandomNodes:
                ps.set(SEED_TYPE, "RandomNodes");
                break;
            case SeedType::SeedTypeFixedNodes:
                ps.set(SEED_TYPE, "FixedNodes");
                break;
            case SeedType::SeedTypeFirstNode:
                ps.set(SEED_TYPE, "FirstNode");
                break;
            case SeedType::SeedTypeAllLeafNodes:
                ps.set(SEED_TYPE, "AllLeafNodes");
                break;
            default:
                return collie::Status::not_found( "Unknown seed type");
        }
        return collie::Status::ok_status();
    }

    collie::Result<SeedType> PropertySerializer::seed_type_import(const PropertySet &ps) {
        auto it = ps.find(SEED_TYPE);
        SeedType seedType = SeedType::SeedTypeNone;
        if (it != ps.end()) {
            if (it->second == "None") {
                seedType = SeedType::SeedTypeNone;
            } else if (it->second == "RandomNodes") {
                seedType = SeedType::SeedTypeRandomNodes;
            } else if (it->second == "FixedNodes") {
                seedType = SeedType::SeedTypeFixedNodes;
            } else if (it->second == "FirstNode") {
                seedType = SeedType::SeedTypeFirstNode;
            } else if (it->second == "AllLeafNodes") {
                seedType = SeedType::SeedTypeAllLeafNodes;
            } else {
                POLARIS_LOG(ERROR) << "Invalid Seed Type in the property. " << it->first << ":" << it->second;
                return collie::Status::invalid_argument( "Invalid Seed Type");
            }
        } else {
            return collie::Status::not_found("Seed Type not found in the property");
        }
        return seedType;
    }

    collie::Status PropertySerializer::index_type_export(polaris::PropertySet &ps, polaris::IndexType type) {
        switch (type) {
            case IndexType::INDEX_NONE:
                ps.set(INDEX_TYPE, "None");
                break;
            case IndexType::INDEX_NGT_GRAPH_AND_TREE:
                ps.set(INDEX_TYPE, "ngt-GraphAndTree");
                break;
            case IndexType::INDEX_NGT_GRAPH:
                ps.set(INDEX_TYPE, "ngt-Graph");
                break;
            case IndexType::INDEX_HNSW_FLAT:
                ps.set(INDEX_TYPE, "hnsw-Flat");
                break;
            case IndexType::IT_FLAT:
                ps.set(INDEX_TYPE, "faiss-Flat");
                break;
            case IndexType::IT_FLATIP:
                ps.set(INDEX_TYPE, "faiss-FlatIP");
                break;
            case IndexType::IT_FLATL2:
                ps.set(INDEX_TYPE, "faiss-FlatL2");
                break;
            case IndexType::IT_LSH:
                ps.set(INDEX_TYPE, "faiss-LSH");
                break;
            case IndexType::IT_IVFFLAT:
                ps.set(INDEX_TYPE, "faiss-IVFFlat");
                break;
            case IndexType::INDEX_VAMANA_DISK:
                ps.set(INDEX_TYPE, "vamana-Disk");
                break;
            case IndexType::INDEX_VAMANA:
                ps.set(INDEX_TYPE, "vamana");
                break;
            case IndexType::INDEX_HNSW:
                ps.set(INDEX_TYPE, "hnsw");
                break;
            default:
                return collie::Status::not_found( "Unknown index type");
        }
        return collie::Status::ok_status();
    }

    collie::Result<IndexType> PropertySerializer::index_type_import(const polaris::PropertySet &ps) {
        auto it = ps.find(INDEX_TYPE);
        IndexType indexType = IndexType::INDEX_NONE;
        if (it != ps.end()) {
            if (it->second == "None") {
                indexType = IndexType::INDEX_NONE;
            } else if (it->second == "ngt-GraphAndTree") {
                indexType = IndexType::INDEX_NGT_GRAPH_AND_TREE;
            } else if (it->second == "ngt-Graph") {
                indexType = IndexType::INDEX_NGT_GRAPH;
            } else if (it->second == "hnsw-Flat") {
                indexType = IndexType::INDEX_HNSW_FLAT;
            } else if (it->second == "faiss-Flat") {
                indexType = IndexType::IT_FLAT;
            } else if (it->second == "faiss-FlatIP") {
                indexType = IndexType::IT_FLATIP;
            } else if (it->second == "faiss-FlatL2") {
                indexType = IndexType::IT_FLATL2;
            } else if (it->second == "faiss-LSH") {
                indexType = IndexType::IT_LSH;
            } else if (it->second == "faiss-IVFFlat") {
                indexType = IndexType::IT_IVFFLAT;
            } else if (it->second == "vamana-Disk") {
                indexType = IndexType::INDEX_VAMANA_DISK;
            } else if (it->second == "vamana") {
                indexType = IndexType::INDEX_VAMANA;
            } else if (it->second == "hnsw") {
                indexType = IndexType::INDEX_HNSW;
            } else {
                POLARIS_LOG(ERROR) << "Invalid Index Type in the property. " << it->first << ":" << it->second;
                return collie::Status::invalid_argument( "Invalid Index Type");
            }
        } else {
            return collie::Status::not_found("Index Type not found in the property");
        }
        return indexType;
    }

    void PropertySerializer::dimension_export(PropertySet &ps, size_t dimension) {
        ps.set(DIMENSION, std::to_string(dimension));
    }
    collie::Result<size_t> PropertySerializer::dimension_import(const PropertySet &ps) {
        auto it = ps.find(DIMENSION);
        if (it != ps.end()) {
            uint64_t dimension = 0;
            auto r = turbo::simple_atoi(it->second, &dimension);
            if (r) {
                return dimension;
            } else {
                return collie::Status::invalid_argument( "Invalid dimension");
            }
        } else {
            return collie::Status::not_found("Dimension not found in the property");
        }
    }

}  // namespace polaris
