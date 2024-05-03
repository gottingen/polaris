//
// Copyright (C) 2021 Yahoo Japan Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//

#pragma once

#include <polaris/graph/ngt/ngtq/quantized_blob_graph.h>
#include <polaris/graph/ngt/command.h>

namespace QBG {
  
  class CLI {
  public:

    int debugLevel;

#if !defined(NGTQ_QBG) || defined(NGTQ_SHARED_INVERTED_INDEX)
    void create(polaris::Args &args) { std::cerr << "not implemented." << std::endl; };
    void load(polaris::Args &args) { std::cerr << "not implemented." << std::endl; };
    void append(polaris::Args &args) { std::cerr << "not implemented." << std::endl; };
    void insert(polaris::Args &args) { std::cerr << "not implemented." << std::endl; };
    void remove(polaris::Args &args) { std::cerr << "not implemented." << std::endl; };
    void buildIndex(polaris::Args &args) { std::cerr << "not implemented." << std::endl; };
    void hierarchicalKmeans(polaris::Args &args) { std::cerr << "not implemented." << std::endl; };
    void search(polaris::Args &args) { std::cerr << "not implemented." << std::endl; };
    void assign(polaris::Args &args) { std::cerr << "not implemented." << std::endl; };
    void extract(polaris::Args &args) { std::cerr << "not implemented." << std::endl; };
    void gt(polaris::Args &args) { std::cerr << "not implemented." << std::endl; };
    void gtRange(polaris::Args &args) { std::cerr << "not implemented." << std::endl; };
    void optimize(polaris::Args &args) { std::cerr << "not implemented." << std::endl; };
    void build(polaris::Args &args) { std::cerr << "not implemented." << std::endl; };
    void rebuild(polaris::Args &args) { std::cerr << "not implemented." << std::endl; };
    void createQG(polaris::Args &args) { std::cerr << "not implemented." << std::endl; };
    void buildQG(polaris::Args &args) { std::cerr << "not implemented." << std::endl; };
    void appendQG(polaris::Args &args) { std::cerr << "not implemented." << std::endl; };
    void searchQG(polaris::Args &args) { std::cerr << "not implemented." << std::endl; };
    void info(polaris::Args &args) { std::cerr << "not implemented." << std::endl; };
#else
    void create(polaris::Args &args);
    void load(polaris::Args &args);
    void append(polaris::Args &args);
    void insert(polaris::Args &args);
    void remove(polaris::Args &args);
    void buildIndex(polaris::Args &args);
    void hierarchicalKmeans(polaris::Args &args);
    void search(polaris::Args &args);
    void assign(polaris::Args &args);
    void extract(polaris::Args &args);
    void gt(polaris::Args &args);
    void gtRange(polaris::Args &args);
    void optimize(polaris::Args &args);
    void build(polaris::Args &args);
    void rebuild(polaris::Args &args);
    void createQG(polaris::Args &args);
    void buildQG(polaris::Args &args);
    void appendQG(polaris::Args &args);
    void searchQG(polaris::Args &args);
    void info(polaris::Args &args);
#endif
    
    void setDebugLevel(int level) { debugLevel = level; }
    int getDebugLevel() { return debugLevel; }

    void help() {
      cerr << "Usage : qbg command database [data]" << endl;
      cerr << "           command : create build quantize search" << endl;
    }

    void execute(polaris::Args args) {
      string command;
      try {
	command = args.get("#0");
      } catch(...) {
	help();
	return;
      }

      debugLevel = args.getl("X", 0);

      try {
	if (debugLevel >= 1) {
	  cerr << "ngt::command=" << command << endl;
	}
	if (command == "search") {
	  search(args);
	} else if (command == "create") {
	  create(args);
	} else if (command == "load") {
	  load(args);
	} else if (command == "append") {
	  append(args);
	} else if (command == "insert") {
	  insert(args);
	} else if (command == "remove") {
	  remove(args);
	} else if (command == "build-index") {
	  buildIndex(args);
	} else if (command == "kmeans") {
	  hierarchicalKmeans(args);
	} else if (command == "assign") {
	  assign(args);
	} else if (command == "extract") {
	  extract(args);
	} else if (command == "gt") {
	  gt(args);
	} else if (command == "gt-range") {
	  gtRange(args);
	} else if (command == "optimize") {
	  optimize(args);
	} else if (command == "build") {
	  build(args);
	} else if (command == "rebuild") {
	  rebuild(args);
	} else if (command == "create-qg") {
	  createQG(args);
	} else if (command == "build-qg") {
	  buildQG(args);
	} else if (command == "append-qg") {
	  appendQG(args);
	} else if (command == "search-qg") {
	  searchQG(args);
	} else if (command == "info") {
	  info(args);
	} else {
	  cerr << "qbg: Illegal command. " << command << endl;
	}
      } catch(polaris::PolarisException &err) {
	cerr << "qbg: Error: " << err.what() << endl;
      }
    }

  };

}; // NGTQBG
