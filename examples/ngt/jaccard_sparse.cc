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

// sort -R sparse_binary.tsv |head -10 > sparse_binary_query_10.tsv
// ./jaccard-sparse create -d 100 -D J sparse
// ./jaccard-sparse append sparse sparse_binary.tsv
// ./jaccard-sparse search sparse sparse_binary_query_10.tsv
//

#include <polaris/graph/ngt/command.h>
#include <polaris/utility/timer.h>

using namespace std;

void help() {
    cerr << "Usage : jaccard-sparse command index [data]" << endl;
    cerr << "           command : info create search append" << endl;
}

void
append(polaris::Args &args) {
    const string usage = "Usage: jaccard-sparse append [-p #-of-thread] [-n data-size] "
                         "index(output) [data.tsv(input)]";
    string database;
    try {
        database = args.get("#1");
    } catch (...) {
        cerr << "jaccard-sparse: Error: DB is not specified." << endl;
        cerr << usage << endl;
        return;
    }
    string data;
    try {
        data = args.get("#2");
    } catch (...) {
        cerr << "jaccard-sparse: Warning: No specified object file. Just build an index for the existing objects."
             << endl;
    }

    int threadSize = args.getl("p", 50);
    size_t dataSize = args.getl("n", 0);

    std::istream *is;
    std::ifstream *ifs = 0;

    try {
        polaris::NgtIndex index(database);
        if (data == "-") {
            is = &std::cin;
        } else {
            ifs = new std::ifstream;
            ifs->std::ifstream::open(data);
            if (!(*ifs)) {
                cerr << "Cannot open the specified data file. " << data << endl;
                return;
            }
            is = ifs;
        }
        string line;
        size_t count = 0;
        while (getline(*is, line)) {
            if (dataSize > 0 && count >= dataSize) {
                break;
            }
            count++;
            vector<uint32_t> object;
            stringstream linestream(line);
            while (!linestream.eof()) {
                uint32_t value;
                linestream >> value;
                if (linestream.fail()) {
                    object.clear();
                    break;
                }
                object.push_back(value);
            }
            if (object.empty()) {
                std::cerr << "jaccard-sparse: Empty line or invalid value. " << count << ":" << line << std::endl;
                continue;
            }
        }
        if (data != "-") {
            delete ifs;
        }
        index.createIndex(threadSize);
        index.saveIndex(database);
    } catch (polaris::PolarisException &err) {
        if (data != "-") {
            delete ifs;
        }
        cerr << "jaccard-sparse: Error " << err.what() << endl;
        cerr << usage << endl;
    }
    return;
}


void
search(polaris::NgtIndex &index, polaris::Command::SearchParameters &searchParameters, ostream &stream) {

    std::ifstream is(searchParameters.query);
    if (!is) {
        std::cerr << "Cannot open the specified file. " << searchParameters.query << std::endl;
        return;
    }

    if (searchParameters.outputMode[0] == 'e') {
        stream << "# Beginning of Evaluation" << endl;
    }

    string line;
    double totalTime = 0;
    size_t queryCount = 0;
    double epsilon = searchParameters.beginOfEpsilon;

    while (getline(is, line)) {
        if (searchParameters.querySize > 0 && queryCount >= searchParameters.querySize) {
            break;
        }
        vector<uint32_t> query;
        stringstream linestream(line);
        while (!linestream.eof()) {
            uint32_t value;
            linestream >> value;
            query.push_back(value);
        }
        auto sparseQuery = index.makeSparseObject(query);
        queryCount++;
        polaris::SearchQuery sc(sparseQuery);
        polaris::ObjectDistances objects;
        sc.setResults(&objects);
        sc.setSize(searchParameters.size);
        sc.setRadius(searchParameters.radius);
        if (searchParameters.accuracy > 0.0) {
            sc.setExpectedAccuracy(searchParameters.accuracy);
        } else {
            sc.setEpsilon(epsilon);
        }
        sc.setEdgeSize(searchParameters.edgeSize);
        polaris::Timer timer;
        switch (searchParameters.indexType) {
            case 't':
                timer.start();
                index.search(sc);
                timer.stop();
                break;
            case 'g':
                timer.start();
                index.searchUsingOnlyGraph(sc);
                timer.stop();
                break;
            case 's':
                timer.start();
                index.linearSearch(sc);
                timer.stop();
                break;
        }
        totalTime += timer.delta.to_seconds<double>();
        if (searchParameters.outputMode[0] == 'e') {
            stream << "# Query No.=" << queryCount << endl;
            stream << "# Query=" << line.substr(0, 20) + " ..." << endl;
            stream << "# Index Type=" << searchParameters.indexType << endl;
            stream << "# Size=" << searchParameters.size << endl;
            stream << "# Radius=" << searchParameters.radius << endl;
            stream << "# Epsilon=" << epsilon << endl;
            stream << "# Query Time (msec)=" << timer.delta.to_milliseconds<double>() << endl;
            stream << "# Distance Computation=" << sc.distanceComputationCount << endl;
            stream << "# Visit Count=" << sc.visitCount << endl;
        } else {
            stream << "Query No." << queryCount << endl;
            stream << "Rank\tID\tDistance" << endl;
        }
        for (size_t i = 0; i < objects.size(); i++) {
            stream << i + 1 << "\t" << objects[i].id << "\t";
            stream << objects[i].distance << endl;
        }
        if (searchParameters.outputMode[0] == 'e') {
            stream << "# End of Search" << endl;
        } else {
            stream << "Query Time= " << timer.delta.to_seconds<double>() << " (sec), " << timer.delta.to_milliseconds<double>() << " (msec)" << endl;
        }
        if (searchParameters.outputMode[0] == 'e') {
            stream << "# End of Query" << endl;
        }
    }
    if (searchParameters.outputMode[0] == 'e') {
        stream << "# Average Query Time (msec)=" << totalTime * 1000.0 / (double) queryCount << endl;
        stream << "# Number of queries=" << queryCount << endl;
        stream << "# End of Evaluation" << endl;
    } else {
        stream << "Average Query Time= " << totalTime / (double) queryCount << " (sec), "
               << totalTime * 1000.0 / (double) queryCount << " (msec), ("
               << totalTime << "/" << queryCount << ")" << endl;
    }
}

void
search(polaris::Args &args) {
    const string usage = "Usage: ngt search [-i index-type(g|t|s)] [-n result-size] [-e epsilon] [-E edge-size] "
                         "[-m open-mode(r|w)] [-o output-mode] index(input) query.tsv(input)";

    string database;
    try {
        database = args.get("#1");
    } catch (...) {
        cerr << "jaccard-sparse: Error: DB is not specified" << endl;
        cerr << usage << endl;
        return;
    }

    polaris::Command::SearchParameters searchParameters(args);

    try {
        polaris::NgtIndex index(database, searchParameters.openMode == 'r');
        search(index, searchParameters, cout);
    } catch (polaris::PolarisException &err) {
        cerr << "jaccard-sparse: Error " << err.what() << endl;
        cerr << usage << endl;
    } catch (...) {
        cerr << "jaccard-sparse: Error" << endl;
        cerr << usage << endl;
    }

}

int
main(int argc, char **argv) {

    polaris::Args args(argc, argv);

    polaris::Command ngt;

    string command;
    try {
        command = args.get("#0");
    } catch (...) {
        help();
        return 0;
    }

    try {
        if (command == "create") {
            ngt.create(args);
        } else if (command == "append") {
            append(args);
        } else if (command == "search") {
            search(args);
        } else {
            cerr << "jaccard-sparse: Error: Illegal command. " << command << endl;
            help();
        }
    } catch (polaris::PolarisException &err) {
        cerr << "jaccard-sparse: Error: " << err.what() << endl;
        help();
        return 0;
    }
    return 0;
}


