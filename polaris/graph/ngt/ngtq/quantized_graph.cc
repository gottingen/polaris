//
// Copyright (C) 2020 Yahoo Japan Corporation
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

#include <polaris/graph/ngt/ngtq/quantized_graph.h>
#include <polaris/graph/ngt/ngtq/quantized_blob_graph.h>
#include <polaris/graph/ngt/ngtq/optimizer.h>

#ifdef NGTQ_QBG
void NGTQG::NgtqgIndex::quantize(const std::string indexPath, size_t dimensionOfSubvector, size_t maxNumOfEdges, bool verbose) {
  {
    polaris::NgtIndex	index(indexPath);
    const std::string quantizedIndexPath = indexPath + "/qg";
    struct stat st;
    if (stat(quantizedIndexPath.c_str(), &st) != 0) {
      polaris::Property ngtProperty;
      index.getProperty(ngtProperty);
      QBG::BuildParameters buildParameters;
      buildParameters.creation.dimensionOfSubvector = dimensionOfSubvector;
      buildParameters.setVerbose(verbose);
 
      NGTQG::NgtqgIndex::create(indexPath, buildParameters);

      NGTQG::NgtqgIndex::append(indexPath, buildParameters);

      QBG::Optimizer optimizer(buildParameters);
#ifdef NGTQG_NO_ROTATION
      if (optimizer.rotation || optimizer.repositioning) {
	std::cerr << "build-qg: Warning! Although rotation or repositioning is specified, turn off rotation and repositioning because of unavailable options." << std::endl;
	optimizer.rotation = false;
	optimizer.repositioning = false;
      }
#endif

      if (optimizer.globalType == QBG::Optimizer::GlobalTypeNone) {
	if (verbose) std::cerr << "build-qg: Warning! None is unavailable for the global type. Zero is set to the global type." << std::endl;
	optimizer.globalType = QBG::Optimizer::GlobalTypeZero;
      }

      optimizer.optimize(quantizedIndexPath);

      QBG::QbgIndex::buildNGTQ(quantizedIndexPath, verbose);

      NGTQG::NgtqgIndex::realign(indexPath, maxNumOfEdges, verbose);
    }
  }

}

void NGTQG::NgtqgIndex::create(const std::string indexPath, QBG::BuildParameters &buildParameters) {
  auto dimensionOfSubvector = buildParameters.creation.dimensionOfSubvector;
  auto dimension = buildParameters.creation.dimension;
  if (dimension != 0 && buildParameters.creation.numOfSubvectors != 0) {
    if (dimension % buildParameters.creation.numOfSubvectors != 0) {
      std::stringstream msg;
      msg << "NGTQBG::NgtqgIndex::create: Invalid dimension and local division No. " << dimension << ":" << buildParameters.creation.numOfSubvectors;
      POLARIS_THROW_EX(msg);
    }
    dimensionOfSubvector = dimension / buildParameters.creation.numOfSubvectors;
  }
  create(indexPath, dimensionOfSubvector, dimension);
}


void NGTQG::NgtqgIndex::append(const std::string indexPath, QBG::BuildParameters &buildParameters) {
  QBG::QbgIndex::appendFromObjectRepository(indexPath, indexPath + "/qg", buildParameters.verbose);
}
#endif
