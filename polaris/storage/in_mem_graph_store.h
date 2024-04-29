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

#pragma once

#include <polaris/storage/abstract_graph_store.h>

namespace polaris
{

class InMemGraphStore : public AbstractGraphStore
{
  public:
    InMemGraphStore(const size_t total_pts, const size_t reserve_graph_degree);

    // returns tuple of <nodes_read, start, num_frozen_points>
    virtual std::tuple<uint32_t, uint32_t, size_t> load(const std::string &index_path_prefix,
                                                        const size_t num_points) override;
    virtual int store(const std::string &index_path_prefix, const size_t num_points, const size_t num_frozen_points,
                      const uint32_t start) override;

    virtual const std::vector<location_t> &get_neighbours(const location_t i) const override;
    virtual void add_neighbour(const location_t i, location_t neighbour_id) override;
    virtual void clear_neighbours(const location_t i) override;
    virtual void swap_neighbours(const location_t a, location_t b) override;

    virtual void set_neighbours(const location_t i, std::vector<location_t> &neighbors) override;

    virtual size_t resize_graph(const size_t new_size) override;
    virtual void clear_graph() override;

    virtual size_t get_max_range_of_graph() override;
    virtual uint32_t get_max_observed_degree() override;

  protected:
    virtual std::tuple<uint32_t, uint32_t, size_t> load_impl(const std::string &filename, size_t expected_num_points);

    int save_graph(const std::string &index_path_prefix, const size_t active_points, const size_t num_frozen_points,
                   const uint32_t start);

  private:
    size_t _max_range_of_graph = 0;
    uint32_t _max_observed_degree = 0;

    std::vector<std::vector<uint32_t>> _graph;
};

} // namespace polaris
