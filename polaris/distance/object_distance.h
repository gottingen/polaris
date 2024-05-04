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

#include <polaris/utility/serialize.h>
#include <fstream>
#include <iomanip>
#include <queue>

namespace polaris {
#pragma pack(2)

    class ObjectDistance {
    public:
        ObjectDistance() : id(0), distance(0.0) {}

        ObjectDistance(unsigned int i, float d) : id(i), distance(d) {}

        inline bool operator==(const ObjectDistance &o) const {
            return (distance == o.distance) && (id == o.id);
        }

        inline void set(unsigned int i, float d) {
            id = i;
            distance = d;
        }

        inline bool operator<(const ObjectDistance &o) const {
            if (distance == o.distance) {
                return id < o.id;
            } else {
                return distance < o.distance;
            }
        }

        inline bool operator>(const ObjectDistance &o) const {
            if (distance == o.distance) {
                return id > o.id;
            } else {
                return distance > o.distance;
            }
        }

        void serialize(std::ofstream &os) {
            polaris::Serializer::write(os, id);
            polaris::Serializer::write(os, distance);
        }

        void deserialize(std::ifstream &is) {
            polaris::Serializer::read(is, id);
            polaris::Serializer::read(is, distance);
        }

        void serializeAsText(std::ofstream &os) {
            os.unsetf(std::ios_base::floatfield);
            os << std::setprecision(8) << id << " " << distance;
        }

        void deserializeAsText(std::ifstream &is) {
            is >> id;
            is >> distance;
        }

        friend std::ostream &operator<<(std::ostream &os, const ObjectDistance &o) {
            os << o.id << " " << o.distance;
            return os;
        }

        friend std::istream &operator>>(std::istream &is, ObjectDistance &o) {
            is >> o.id;
            is >> o.distance;
            return is;
        }

        uint32_t id;
        float distance;
    };
    typedef std::priority_queue<ObjectDistance, std::vector<ObjectDistance>, std::less<ObjectDistance> > ResultPriorityQueue;
}  // namespace polaris