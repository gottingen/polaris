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

#include <polaris/graph/ngt/index.h>
#include <collie/cli/cli.h>

using namespace std;

string indexPath = "index";
string objectFile = "./data/sift-dataset-5k.tsv";
string queryFile = "./data/sift-query-3.tsv";
static void run_ann();
int main(int argc, char **argv) {
    collie::App app("cosine_float");

    app.add_option("--index", indexPath, "Index path");
    app.add_option("--object", objectFile, "Object file path");
    app.add_option("--query", queryFile, "Query file path");
    app.callback(run_ann);
    COLLIE_CLI_PARSE(app, argc, argv);
    // index construction
    return 0;
}
void run_ann() {
    try {
        NGT::Property property;
        property.dimension = 128;
        property.objectType = NGT::ObjectSpace::ObjectType::Float;
        property.distanceType = NGT::Index::Property::DistanceType::DistanceTypeCosine;
        NGT::Index::create(indexPath, property);
        NGT::Index index(indexPath);
        ifstream is(objectFile);
        string line;
        while (getline(is, line)) {
            vector<float> obj;
            stringstream linestream(line);
            while (!linestream.eof()) {
                float value;
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
            index.append(obj);
        }
        index.createIndex(16);
        index.save();
    } catch (polaris::PolarisException &err) {
        cerr << "Error " << err.what() << endl;
        exit(1);
    } catch (...) {
        cerr << "Error" << endl;
        exit(1);
    }

    // nearest neighbor search
    try {
        NGT::Index index(indexPath);
        NGT::Property property;
        index.getProperty(property);
        ifstream is(queryFile);
        string line;
        while (getline(is, line)) {
            vector<float> query;
            {
                stringstream linestream(line);
                while (!linestream.eof()) {
                    float value;
                    linestream >> value;
                    query.push_back(value);
                }
                query.resize(property.dimension);
                cout << "Query : ";
                for (size_t i = 0; i < 5; i++) {
                    cout << query[i] << " ";
                }
                cout << "...";
            }
            NGT::SearchQuery sc(query);
            NGT::ObjectDistances objects;
            sc.setResults(&objects);
            sc.setSize(10);
            sc.setEpsilon(0.1);

            index.search(sc);
            cout << endl << "Rank\tID\tDistance" << std::showbase << endl;
            for (size_t i = 0; i < objects.size(); i++) {
                cout << i + 1 << "\t" << objects[i].id << "\t" << objects[i].distance << "\t: ";
                NGT::ObjectSpace &objectSpace = index.getObjectSpace();
                float *object = static_cast<float *>(objectSpace.getObject(objects[i].id));
                for (size_t idx = 0; idx < 5; idx++) {
                    cout << object[idx] << " ";
                }
                cout << endl;
            }
            cout << endl;
        }
    } catch (polaris::PolarisException &err) {
        cerr << "Error " << err.what() << endl;
        exit(1);
    } catch (...) {
        cerr << "Error" << endl;
        exit(1);
    }
}


