// Copyright 2024 The EA Authors.
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

#include <polaris/storage/in_mem_graph_store.h>
#include <polaris/core/log.h>
#include <polaris/io/utils.h>

namespace polaris {
    InMemGraphStore::InMemGraphStore(const size_t total_pts, const size_t reserve_graph_degree)
            : AbstractGraphStore(total_pts, reserve_graph_degree) {
        this->resize_graph(total_pts);
        for (size_t i = 0; i < total_pts; i++) {
            _graph[i].reserve(reserve_graph_degree);
        }
    }

    std::tuple<uint32_t, uint32_t, size_t> InMemGraphStore::load(const std::string &index_path_prefix,
                                                                 const size_t num_points) {
        return load_impl(index_path_prefix, num_points);
    }

    int InMemGraphStore::store(const std::string &index_path_prefix, const size_t num_points,
                               const size_t num_frozen_points, const uint32_t start) {
        return save_graph(index_path_prefix, num_points, num_frozen_points, start);
    }

    const std::vector<location_t> &InMemGraphStore::get_neighbours(const location_t i) const {
        return _graph.at(i);
    }

    void InMemGraphStore::add_neighbour(const location_t i, location_t neighbour_id) {
        _graph[i].emplace_back(neighbour_id);
        if (_max_observed_degree < _graph[i].size()) {
            _max_observed_degree = (uint32_t) (_graph[i].size());
        }
    }

    void InMemGraphStore::clear_neighbours(const location_t i) {
        _graph[i].clear();
    };

    void InMemGraphStore::swap_neighbours(const location_t a, location_t b) {
        _graph[a].swap(_graph[b]);
    };

    void InMemGraphStore::set_neighbours(const location_t i, std::vector<location_t> &neighbours) {
        _graph[i].assign(neighbours.begin(), neighbours.end());
        if (_max_observed_degree < neighbours.size()) {
            _max_observed_degree = (uint32_t) (neighbours.size());
        }
    }

    size_t InMemGraphStore::resize_graph(const size_t new_size) {
        _graph.resize(new_size);
        set_total_points(new_size);
        return _graph.size();
    }

    void InMemGraphStore::clear_graph() {
        _graph.clear();
    }

    std::tuple<uint32_t, uint32_t, size_t> InMemGraphStore::load_impl(const std::string &filename,
                                                                      size_t expected_num_points) {
        size_t expected_file_size;
        size_t file_frozen_pts;
        uint32_t start;
        size_t file_offset = 0; // will need this for single file format support

        std::ifstream in;
        in.exceptions(std::ios::badbit | std::ios::failbit);
        in.open(filename, std::ios::binary);
        in.seekg(file_offset, in.beg);
        in.read((char *) &expected_file_size, sizeof(size_t));
        in.read((char *) &_max_observed_degree, sizeof(uint32_t));
        in.read((char *) &start, sizeof(uint32_t));
        in.read((char *) &file_frozen_pts, sizeof(size_t));
        size_t vamana_metadata_size = sizeof(size_t) + sizeof(uint32_t) + sizeof(uint32_t) + sizeof(size_t);

        POLARIS_LOG(INFO)<< "From graph header, expected_file_size: " << expected_file_size
                      << ", _max_observed_degree: " << _max_observed_degree << ", _start: " << start
                      << ", file_frozen_pts: " << file_frozen_pts;

        POLARIS_LOG(INFO)<< "Loading vamana graph " << filename;

        // If user provides more points than max_points
        // resize the _graph to the larger size.
        if (get_total_points() < expected_num_points) {
            POLARIS_LOG(INFO) << "resizing graph to " << expected_num_points;
            this->resize_graph(expected_num_points);
        }

        size_t bytes_read = vamana_metadata_size;
        size_t cc = 0;
        uint32_t nodes_read = 0;
        while (bytes_read != expected_file_size) {
            uint32_t k;
            in.read((char *) &k, sizeof(uint32_t));

            if (k == 0) {
                POLARIS_LOG(ERROR) << "ERROR: Point found with no out-neighbours, point#" << nodes_read;
            }

            cc += k;
            ++nodes_read;
            std::vector<uint32_t> tmp(k);
            tmp.reserve(k);
            in.read((char *) tmp.data(), k * sizeof(uint32_t));
            _graph[nodes_read - 1].swap(tmp);
            bytes_read += sizeof(uint32_t) * ((size_t) k + 1);
            if (nodes_read % 10000000 == 0)
                POLARIS_LOG(INFO) << ".";
            if (k > _max_range_of_graph) {
                _max_range_of_graph = k;
            }
        }

        POLARIS_LOG(INFO) << "done. Index has " << nodes_read << " nodes and " << cc << " out-edges, _start is set to "
                      << start;
        return std::make_tuple(nodes_read, start, file_frozen_pts);
    }

    int InMemGraphStore::save_graph(const std::string &index_path_prefix, const size_t num_points,
                                    const size_t num_frozen_points, const uint32_t start) {
        std::ofstream out;
        open_file_to_write(out, index_path_prefix);

        size_t file_offset = 0;
        out.seekp(file_offset, out.beg);
        size_t index_size = 24;
        uint32_t max_degree = 0;
        out.write((char *) &index_size, sizeof(uint64_t));
        out.write((char *) &_max_observed_degree, sizeof(uint32_t));
        uint32_t ep_u32 = start;
        out.write((char *) &ep_u32, sizeof(uint32_t));
        out.write((char *) &num_frozen_points, sizeof(size_t));

        // Note: num_points = _nd + _num_frozen_points
        for (uint32_t i = 0; i < num_points; i++) {
            uint32_t GK = (uint32_t) _graph[i].size();
            out.write((char *) &GK, sizeof(uint32_t));
            out.write((char *) _graph[i].data(), GK * sizeof(uint32_t));
            max_degree = _graph[i].size() > max_degree ? (uint32_t) _graph[i].size() : max_degree;
            index_size += (size_t) (sizeof(uint32_t) * (GK + 1));
        }
        out.seekp(file_offset, out.beg);
        out.write((char *) &index_size, sizeof(uint64_t));
        out.write((char *) &max_degree, sizeof(uint32_t));
        out.close();
        return (int) index_size;
    }

    size_t InMemGraphStore::get_max_range_of_graph() {
        return _max_range_of_graph;
    }

    uint32_t InMemGraphStore::get_max_observed_degree() {
        return _max_observed_degree;
    }

} // namespace polaris
