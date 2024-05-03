//
// Copyright (C) 2015 Yahoo Japan Corporation
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

#include <polaris/core/defines.h>
#include <polaris/graph/ngt/common.h>
#include <polaris/graph/ngt/object_space.h>
#include <polaris/graph/ngt/object_repository.h>

NGT::Distance NGT::ObjectSpace::compareWithL1(NGT::Object &o1, NGT::Object &o2) {
  auto dim = getPaddedDimension();
  NGT::Distance d;
  if (getObjectType() == typeid(uint8_t)) {
    d = PrimitiveComparator::compareL1(reinterpret_cast<uint8_t*>(o1.getPointer()), 
				       reinterpret_cast<uint8_t*>(o2.getPointer()), dim);
  } else if (getObjectType() == typeid(float16)) {
    d = PrimitiveComparator::compareL1(reinterpret_cast<float16*>(o1.getPointer()), 
				       reinterpret_cast<float16*>(o2.getPointer()), dim);
  } else if (getObjectType() == typeid(float)) {
    d = PrimitiveComparator::compareL1(reinterpret_cast<float*>(o1.getPointer()), 
				       reinterpret_cast<float*>(o2.getPointer()), dim);
  } else {
    std::stringstream msg;
    msg << "ObjectSpace::compareWithL1: Fatal Inner Error! Unexpected object type.";
    POLARIS_THROW_EX(msg);
  }
  return d;
}

