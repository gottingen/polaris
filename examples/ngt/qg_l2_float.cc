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

#include    <polaris/graph/ngt/ngtq/quantized_graph.h>

int main(int argc, char **argv) {
#ifdef NGTQ_QBG
    string indexPath = "index";
    string objectFile = "./data/sift-dataset-5k.tsv";
    string queryFile = "./data/sift-query-3.tsv";

    // index construction
    try {
        polaris::NgtParameters property;
        property.dimension = 128;
        property.object_type = polaris::ObjectType::UINT8;
        property.metric = polaris::MetricType::METRIC_L2;
        std::cout << "creating the index framework..." << std::endl;
        polaris::NgtIndex::create(indexPath, property);
        polaris::NgtIndex index(indexPath);
        ifstream is(objectFile);
        string line;
        std::cout << "appending the objects..." << std::endl;
        while (getline(is, line)) {
            vector<float> obj;
            stringstream linestream(line);
            while (!linestream.eof()) {
                int value;
                linestream >> value;
                if (linestream.fail()) {
                    obj.clear();
                    break;
                }
                obj.push_back(value);
            }
            if (obj.empty()) {
                cerr << "An empty line or invalid value: " << line << endl;
                continue;
            }
            obj.resize(property.dimension);  // cut off additional data in the file.
            index.insert(obj);
        }
        std::cout << "building the index..." << std::endl;
        index.createIndex(16);
        index.save();
    } catch (polaris::PolarisException &err) {
        cerr << "Error " << err.what() << endl;
        return 1;
    } catch (...) {
        cerr << "Error" << endl;
        return 1;
    }

    // quantization
    size_t dimensionOfSubvector = 1;
    size_t maxNumberOfEdges = 50;
    try {
        std::cout << "quantizing the index..." << std::endl;
        NGTQG::NgtqgIndex::quantize(indexPath, dimensionOfSubvector, maxNumberOfEdges, true);
    } catch (polaris::PolarisException &err) {
        cerr << "Error " << err.what() << endl;
        return 1;
    } catch (...) {
        cerr << "Error" << endl;
        return 1;
    }

    // nearest neighbor search
    try {
        NGTQG::NgtqgIndex index(indexPath);
        polaris::NgtParameters property;
        index.getProperty(property);
        ifstream is(queryFile);
        string line;
        std::cout << "searching the index..." << std::endl;
        while (getline(is, line)) {
            vector<uint8_t> query;
            {
                stringstream linestream(line);
                while (!linestream.eof()) {
                    int value;
                    linestream >> value;
                    query.push_back(value);
                }
                query.resize(property.dimension);
                cout << "Query : ";
                for (size_t i = 0; i < 5; i++) {
                    cout << static_cast<int>(query[i]) << " ";
                }
                cout << "...";
            }

            NGTQG::SearchQuery sc(query);
            polaris::ObjectDistances objects;
            sc.setResults(&objects);
            sc.setSize(10);
            sc.setEpsilon(0.1);

            index.search(sc);
            cout << endl << "Rank\tID\tDistance: Object" << std::showbase << endl;
            for (size_t i = 0; i < objects.size(); i++) {
                cout << i + 1 << "\t" << objects[i].id << "\t" << objects[i].distance << "\t: ";
                polaris::ObjectSpace &objectSpace = index.getObjectSpace();
                uint8_t *object = static_cast<uint8_t *>(objectSpace.getObject(objects[i].id));
                for (size_t idx = 0; idx < 5; idx++) {
                    cout << static_cast<int>(object[idx]) << " ";
                }
                cout << "..." << endl;
            }
            cout << endl;
        }
    } catch (polaris::PolarisException &err) {
        cerr << "Error " << err.what() << endl;
        return 1;
    } catch (...) {
        cerr << "Error" << endl;
        return 1;
    }
#endif
    return 0;
}


