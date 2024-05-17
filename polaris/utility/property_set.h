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
#include <cstddef>
#include <sstream>
#include <iostream>
#include <cstring>
#include <map>
#include <iomanip>
#include <iostream>
#include <fstream>
#include <polaris/utility/polaris_exception.h>
#include <polaris/utility/utils.h>
#include <polaris/core/log.h>
#include <turbo/status/result_status.h>
#include <turbo/strings/numbers.h>
#include <cstdint>

namespace polaris {
    class PropertySet : public std::map<std::string, std::string> {
    public:
        void set(const std::string &key, const std::string &value) {
            iterator it = find(key);
            if (it == end()) {
                insert(std::pair<std::string, std::string>(key, value));
            } else {
                (*it).second = value;
            }
        }

        template<class VALUE_TYPE>
        void set(const std::string &key, VALUE_TYPE value) {
            std::stringstream vstr;
            auto precision = vstr.precision();
            vstr << std::setprecision(7);
            vstr << value;
            auto it = find(key);
            if (it == end()) {
                insert(std::pair<std::string, std::string>(key, vstr.str()));
            } else {
                (*it).second = vstr.str();
            }
            vstr << std::setprecision(precision);
        }

        std::string get(const std::string &key) const {
            auto it = find(key);
            if (it != end()) {
                return it->second;
            }
            return "";
        }

        float getf(const std::string &key, float defvalue) const {
            auto it = find(key);
            if (it != end()) {
                float val;
                auto b = turbo::simple_atof(it->second, &val);
                if (!b) {
                    POLARIS_LOG(ERROR) << "Warning: Illegal property. " << key << ":" << it->second;
                    return defvalue;
                }
                return val;
            }
            return defvalue;
        }

        void updateAndInsert(PropertySet &prop) {
            for (auto i = prop.begin(); i != prop.end(); ++i) {
                set((*i).first, (*i).second);
            }
        }

        long getl(const std::string &key, long defvalue) const {
            auto it = find(key);
            if (it != end()) {
                long val;
                auto b = turbo::simple_atoi(it->second, &val);
                if (!b) {
                    POLARIS_LOG(ERROR)<< "Warning: Illegal property. " << key << ":" << it->second;
                    return defvalue;
                }
                return val;
            }
            return defvalue;
        }

        void setb(const std::string &key, bool value) {
            set(key, value ? "true" : "false");
        }

        long getb(const std::string &key, bool defvalue) const {
            auto it = find(key);
            if (it != end()) {
                bool val;
                auto b = turbo::simple_atob(it->second, &val);
                if (!b) {
                    POLARIS_LOG(ERROR)<< "Warning: Illegal property. " << key << ":" << it->second;
                    return defvalue;
                }
                return val;
            }
            return defvalue;
        }

        [[nodiscard]] turbo::Status load(const std::string &f) {
            std::ifstream st(f);
            if (!st) {
                return turbo::make_status(errno, "PropertySet::load: Cannot load the property file {}", f);
            }
            return load(st);
        }

        [[nodiscard]] turbo::Status save(const std::string &f) {
            std::ofstream st(f);
            if (!st) {
                return turbo::make_status(errno, "PropertySet::save: Cannot save. {}", f);
            }
            return save(st);
        }

        [[nodiscard]] turbo::Status save(std::ofstream &os) const {

            try {
                for (auto i = this->begin(); i != this->end(); i++) {
                    os << i->first << "\t" << i->second << std::endl;
                }
            } catch (std::exception &e) {
                return turbo::make_status(errno, "PropertySet::save: {}", e.what());
            }
            return turbo::ok_status();
        }

        [[nodiscard]] turbo::Status load(std::ifstream &is) {
            std::string line;
            try {
            while (getline(is, line)) {
                std::vector<std::string> tokens;
                polaris::Common::tokenize(line, tokens, "\t");
                if (tokens.size() != 2) {
                    std::cerr << "Property file is illegal. " << line << std::endl;
                    continue;
                }
                set(tokens[0], tokens[1]);
            }
            } catch (std::exception &e) {
                return turbo::make_status(errno, "PropertySet::load: {}", e.what());
            }
            return turbo::ok_status();
        }
    };

}  // namespace polaris
