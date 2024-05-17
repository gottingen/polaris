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

#include <polaris/core/parameter/vamana_parameters.h>
#include <collie/meta/reflect.h>

namespace polaris {

    turbo::Status IndexWriteParameters::export_property(polaris::PropertySet &p) const {
        p.set("search_list_size", search_list_size);
        p.set("max_degree", max_degree);
        p.setb("saturate_graph", saturate_graph);
        p.set("max_occlusion_size", max_occlusion_size);
        p.set("alpha", alpha);
        p.set("num_threads", num_threads);
    }

    turbo::Status IndexWriteParameters::import_property(const polaris::PropertySet &p) {
        search_list_size = p.getl("search_list_size", search_list_size);
        max_degree = p.getl("max_degree", max_degree);
        saturate_graph = p.getb("saturate_graph", saturate_graph);
        max_occlusion_size = p.getl("max_occlusion_size", max_occlusion_size);
        alpha = p.getf("alpha", alpha);
        num_threads = p.getl("num_threads", num_threads);
        return turbo::ok_status();
    }
    [[nodiscard]] turbo::Status IndexSearchParams::export_property(polaris::PropertySet &p) const {
        p.set("initial_search_list_size", initial_search_list_size);
        p.set("num_search_threads", num_search_threads);
        return turbo::ok_status();
    }

    turbo::Status IndexSearchParams::import_property(const polaris::PropertySet &p) {
        initial_search_list_size = p.getl("initial_search_list_size", initial_search_list_size);
        num_search_threads = p.getl("num_search_threads", num_search_threads);
        return turbo::ok_status();
    }

    turbo::Status VamanaIndexConfig::export_property(polaris::PropertySet &p) const {
        auto idxts = PropertySerializer::index_type_export(p, IndexType::INDEX_VAMANA);
        if (!idxts.ok()) {
            POLARIS_LOG(ERROR) << idxts.message();
            return idxts;
        }
        /// basic parameters
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
        rs = PropertySerializer::database_type_export(p, "data_strategy", data_strategy);
        if (!rs.ok()) {
            POLARIS_LOG(ERROR) << rs.message();
            return rs;
        }
        rs = PropertySerializer::database_type_export(p, "graph_strategy", graph_strategy);
        if (!rs.ok()) {
            POLARIS_LOG(ERROR) << rs.message();
            return rs;
        }
        p.setb("dynamic_index", dynamic_index);
        p.setb("pq_dist_build", pq_dist_build );
        p.setb("concurrent_consolidate", concurrent_consolidate);
        p.setb("use_opq", use_opq );
        p.set("num_pq_chunks", num_pq_chunks);
        p.set("num_frozen_pts", num_frozen_pts);
        p.setb("has_index_write_params", (bool)index_write_params);
        p.setb("has_index_search_params", (bool)index_search_params);
        if (index_write_params) {
            rs = index_write_params->export_property(p);
            if (!rs.ok()) {
                POLARIS_LOG(ERROR) << rs.message();
                return rs;
            }
        }
        if (index_search_params) {
            rs = index_search_params->export_property(p);
            if (!rs.ok()) {
                POLARIS_LOG(ERROR) << rs.message();
                return rs;
            }
        }
        return turbo::ok_status();
    }

    [[nodiscard]] turbo::Status VamanaIndexConfig::import_property(const polaris::PropertySet &p) {
        auto idxts = PropertySerializer::index_type_import(p);
        if (!idxts.ok()) {
            POLARIS_LOG(ERROR) << idxts.status().message();
            return idxts.status();
        }
        if (idxts.value() != IndexType::INDEX_VAMANA) {
            return turbo::make_status(turbo::kEINVAL, "IndexType mismatch index_type: {} != INDEX_VAMANA", NAMEOF_ENUM(idxts.value()));
        }
        /// basic parameters
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
        /// data and graph strategy
        auto rs = PropertySerializer::database_type_import(p, "data_strategy");
        if (!rs.ok()) {
            POLARIS_LOG(ERROR) << rs.status().message();
            return rs.status();
        }
        data_strategy = rs.value();
        auto grs = PropertySerializer::database_type_import(p, "graph_strategy");
        if (!grs.ok()) {
            POLARIS_LOG(ERROR) << grs.status().message();
            return grs.status();
        }
        graph_strategy = grs.value();

        dynamic_index = p.getb("dynamic_index", dynamic_index);
        pq_dist_build = p.getb("pq_dist_build", pq_dist_build);
        concurrent_consolidate = p.getb("concurrent_consolidate", concurrent_consolidate);
        use_opq = p.getb("use_opq", use_opq);
        num_pq_chunks = p.getl("num_pq_chunks", num_pq_chunks);
        num_frozen_pts = p.getl("num_frozen_pts", num_frozen_pts);
        bool has_index_write_params = p.getb("has_index_write_params", false);
        bool has_index_search_params = p.getb("has_index_search_params", false);
        if (has_index_write_params) {
            index_write_params = std::make_shared<IndexWriteParameters>();
            auto irs = index_write_params->import_property(p);
            if (!rs.ok()) {
                POLARIS_LOG(ERROR) << irs.message();
                return irs;
            }
        }
        if (has_index_search_params) {
            index_search_params = std::make_shared<IndexSearchParams>();
            auto irs = index_search_params->import_property(p);
            if (!rs.ok()) {
                POLARIS_LOG(ERROR) << irs.message();
                return irs;
            }
        }
        return turbo::ok_status();
    }

    [[nodiscard]] turbo::Status VamanaDiskIndexConfig::export_property(polaris::PropertySet &p) const {
        auto idxts = PropertySerializer::index_type_export(p, IndexType::INDEX_VAMANA_DISK);
        if (!idxts.ok()) {
            POLARIS_LOG(ERROR) << idxts.message();
            return idxts;
        }
        /// basic parameters
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
        /// disk index parameters
        /**
         * uint32_t R{0};
        uint32_t L{0};
        float    B{0.0f};
        float    M{0.0f};
        uint32_t num_threads{0};
        uint32_t pq_dims{0};
        bool    append_reorder_data{false};
        uint32_t  build_pq_bytes{0};
        uint32_t pq_chunks{0};
        bool use_opq{false};
        uint32_t num_nodes_to_cache{0};
         */
        p.set("vdisk_R", R);
        p.set("vdisk_L", L);
        p.set("vdisk_B", B);
        p.set("vdisk_M", M);
        p.set("vdisk_num_threads", num_threads);
        p.set("vdisk_pq_dims", pq_dims);
        p.setb("vdisk_append_reorder_data", append_reorder_data);
        p.set("vdisk_build_pq_bytes", build_pq_bytes);
        p.set("vdisk_pq_chunks", pq_chunks);
        p.setb("vdisk_use_opq", use_opq);
        p.set("vdisk_num_nodes_to_cache", num_nodes_to_cache);
        return turbo::ok_status();
    }

    [[nodiscard]] turbo::Status VamanaDiskIndexConfig::import_property(const polaris::PropertySet &p) {
        auto idxts = PropertySerializer::index_type_import(p);
        if (!idxts.ok()) {
            POLARIS_LOG(ERROR) << idxts.status().message();
            return idxts.status();
        }
        if (idxts.value() != IndexType::INDEX_VAMANA_DISK) {
            return turbo::make_status(turbo::kEINVAL, "IndexType mismatch index_type: {} != INDEX_VAMANA_DISK", NAMEOF_ENUM(idxts.value()));
        }
        /// basic parameters
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
        /// disk index parameters

        R = p.getl("vdisk_R", R);
        L = p.getl("vdisk_L", L);
        B = p.getf("vdisk_B", B);
        M = p.getf("vdisk_M", M);
        num_threads = p.getl("vdisk_num_threads", num_threads);
        pq_dims = p.getl("vdisk_pq_dims", pq_dims);
        append_reorder_data = p.getb("vdisk_append_reorder_data", append_reorder_data);
        build_pq_bytes = p.getl("vdisk_build_pq_bytes", build_pq_bytes);
        pq_chunks = p.getl("vdisk_pq_chunks", pq_chunks);
        use_opq = p.getb("vdisk_use_opq", use_opq);
        num_nodes_to_cache = p.getl("vdisk_num_nodes_to_cache", num_nodes_to_cache);
        return turbo::ok_status();
    }
}  // namespace polaris
