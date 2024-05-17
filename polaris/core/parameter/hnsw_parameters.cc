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

#include <polaris/core/parameter/hnsw_parameters.h>

namespace polaris {

    [[nodiscard]] turbo::Status HnswParameters::export_property(polaris::PropertySet &p) const {
        auto idxts = PropertySerializer::index_type_export(p, IndexType::INDEX_HNSW);
        if (!idxts.ok()) {
            POLARIS_LOG(ERROR) << idxts.message();
            return idxts;
        }
        auto rs = PropertySerializer::distance_type_export(p, metric);
        if (!rs.ok()) {
            POLARIS_LOG(ERROR) << rs.message();
            return rs;
        }
        rs = PropertySerializer::object_type_export(p, object_type);
        if (!rs.ok()) {
            POLARIS_LOG(ERROR) << rs.message();
            return rs;
        }
        rs = PropertySerializer::database_type_export(p, databaseType);
        if (!rs.ok()) {
            POLARIS_LOG(ERROR) << rs.message();
            return rs;
        }
        rs = PropertySerializer::object_alignment_export(p, objectAlignment);
        if (!rs.ok()) {
            POLARIS_LOG(ERROR) << rs.message();
            return rs;
        }
        PropertySerializer::dimension_export(p, dimension);
        p.set("max_points", max_points);
        p.set("load_threads", load_threads);
        p.set("work_threads", work_threads);
        p.set("hnsw_m", m);
        p.set("hnsw_ef", ef);
        p.set("hnsw_ef_construction", ef_construction);
        p.set("hnsw_random_seed", random_seed);
        return turbo::ok_status();
    }

    [[nodiscard]] turbo::Status HnswParameters::import_property(polaris::PropertySet &p) {
        auto idxts = PropertySerializer::index_type_import(p);
        if (!idxts.ok()) {
            POLARIS_LOG(ERROR) << idxts.status().message();
            return idxts.status();
        }
        if(idxts.value() != IndexType::INDEX_HNSW) {
            POLARIS_LOG(ERROR) << "Index type mismatch";
            return turbo::make_status(turbo::kEINVAL, "Index type mismatch");
        }
        auto drs = PropertySerializer::distance_type_import(p);
        if (!drs.ok()) {
            POLARIS_LOG(ERROR) << drs.status().message();
            return drs.status();
        }
        metric = drs.value();
        auto ors = PropertySerializer::object_type_import(p);
        if (!ors.ok()) {
            POLARIS_LOG(ERROR) << ors.status().message();
            return ors.status();
        }
        object_type = ors.value();
        auto dtrs = PropertySerializer::database_type_import(p);
        if (!dtrs.ok()) {
            POLARIS_LOG(ERROR) << dtrs.status().message();
            return dtrs.status();
        }
        databaseType = dtrs.value();
        auto ars = PropertySerializer::object_alignment_import(p);
        if (!ars.ok()) {
            POLARIS_LOG(ERROR) << ars.status().message();
            return ars.status();
        }
        objectAlignment = ars.value();
        auto  dmrs = PropertySerializer::dimension_import(p);
        if (!dmrs.ok()) {
            POLARIS_LOG(ERROR) << dmrs.status().message();
            return dmrs.status();
        }
        dimension = dmrs.value();
        max_points = p.getl("max_points", max_points);
        load_threads = p.getl("load_threads", load_threads);
        work_threads = p.getl("work_threads", work_threads);
        m = p.getl("hnsw_m", m);
        ef = p.getl("hnsw_ef", ef);
        ef_construction = p.getl("hnsw_ef_construction", ef_construction);
        random_seed = p.getl("hnsw_random_seed", random_seed);
        return turbo::ok_status();
    }
}  // namespace polaris
