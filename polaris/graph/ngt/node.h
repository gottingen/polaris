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

#pragma once

#include <polaris/core/defines.h>
#include <polaris/graph/ngt/common.h>
#include <polaris/graph/ngt/object_space_repository.h>
#include <algorithm>

namespace NGT {
  class DVPTree;
  class InternalNode;
  class LeafNode;
  class Node {
  public:
    typedef unsigned int	NodeID;
    class ID {
    public:
      enum Type {
	Leaf		= 1,
	Internal	= 0
      };
    ID():id(0) {}
      ID &operator=(const ID &n) {
	id = n.id;
	return *this;
      }
      ID &operator=(int i) {
	setID(i);
	return *this;
      }
      bool operator==(ID &n) { return id == n.id; }
      bool operator<(ID &n) { return id < n.id; }
      Type getType() { return (Type)((0x80000000 & id) >> 31); }
      NodeID getID() { return 0x7fffffff & id; }
      NodeID get() { return id; }
      void setID(NodeID i) { id = (0x80000000 & id) | i; }
      void setType(Type t) { id = (t << 31) | getID(); }
      void setRaw(NodeID i) { id = i; }
      void setNull() { id = 0; }
      void serialize(std::ofstream &os) { NGT::Serializer::write(os, id); }
      void deserialize(std::ifstream &is) { NGT::Serializer::read(is, id); }
      void serializeAsText(std::ofstream &os) { NGT::Serializer::writeAsText(os, id); }
      void deserializeAsText(std::ifstream &is) { NGT::Serializer::readAsText(is, id); }
    protected:
      NodeID id;
    };

    class Object {
    public:
      Object():object(0) {}
      bool operator<(const Object &o) const { return distance < o.distance; }
      static const double	Pivot;
      ObjectID		id;
      PersistentObject	*object;
      Distance		distance;
      Distance		leafDistance;
      int		clusterID;
    };

    typedef std::vector<Object>	Objects;

    Node() {
      parent.setNull();
      id.setNull();
    }

    virtual ~Node() {}

    Node &operator=(const Node &n) {
      id = n.id;
      parent = n.parent;
      return *this;
    }

    void serialize(std::ofstream &os) {
      id.serialize(os);
      parent.serialize(os);
    }

    void deserialize(std::ifstream &is) {
      id.deserialize(is);
      parent.deserialize(is);
    }

    void serializeAsText(std::ofstream &os) {
      id.serializeAsText(os);
      os << " ";
      parent.serializeAsText(os);
    }

    void deserializeAsText(std::ifstream &is) {
      id.deserializeAsText(is);
      parent.deserializeAsText(is);
    }

    void setPivot(NGT::Object &f, ObjectSpace &os) {
      if (pivot == 0) {
	pivot = NGT::Object::allocate(os);
      }
      os.copy(getPivot(), f);
    }
    NGT::Object &getPivot() { return *pivot; }
    void deletePivot(ObjectSpace &os) {
      os.deleteObject(pivot);
    }

    bool pivotIsEmpty() {
      return pivot == 0;
    }

    ID		id;
    ID		parent;

    NGT::Object		*pivot;

  };


  class InternalNode : public Node {
  public:
    InternalNode(size_t csize) : childrenSize(csize) { initialize(); }
    InternalNode(NGT::ObjectSpace *os = 0) : childrenSize(5) { initialize(); }

    ~InternalNode() {
      if (children != 0) {
        delete[] children;
      }
      if (borders != 0) {
        delete[] borders;
      }
    }
    void initialize() {
      id = 0;
      id.setType(ID::Internal);
      pivot = 0;
      children = new ID[childrenSize];
      for (size_t i = 0; i < childrenSize; i++) {
  	getChildren()[i] = 0;
      }
      borders = new Distance[childrenSize - 1];
      for (size_t i = 0; i < childrenSize - 1; i++) {
  	getBorders()[i] = 0;
      }
    }

    void updateChild(DVPTree &dvptree, ID src, ID dst);

    ID *getChildren() { return children; }
    Distance *getBorders() { return borders; }

    void serialize(std::ofstream &os, ObjectSpace *objectspace = 0) {
      Node::serialize(os);
      if (pivot == 0) {
        POLARIS_THROW_EX("Node::write: pivot is null!");
      }
      assert(objectspace != 0);
      getPivot().serialize(os, objectspace);
      NGT::Serializer::write(os, childrenSize);
      for (size_t i = 0; i < childrenSize; i++) {
	getChildren()[i].serialize(os);
      }
      for (size_t i = 0; i < childrenSize - 1; i++) {
	NGT::Serializer::write(os, getBorders()[i]);
      }
    }
    void deserialize(std::ifstream &is, ObjectSpace *objectspace = 0) {
      Node::deserialize(is);
      if (pivot == 0) {
	pivot = PersistentObject::allocate(*objectspace);
      }
      assert(objectspace != 0);
      getPivot().deserialize(is, objectspace);
      NGT::Serializer::read(is, childrenSize);
      assert(children != 0);
      for (size_t i = 0; i < childrenSize; i++) {
	getChildren()[i].deserialize(is);
      }
      assert(borders != 0);
      for (size_t i = 0; i < childrenSize - 1; i++) {
	NGT::Serializer::read(is, getBorders()[i]);
      }
    }

    void serializeAsText(std::ofstream &os, ObjectSpace *objectspace = 0) {
      Node::serializeAsText(os);
      if (pivot == 0) {
        POLARIS_THROW_EX("Node::write: pivot is null!");
      }
      os << " ";
      assert(objectspace != 0);
      getPivot().serializeAsText(os, objectspace);
      os << " ";
      NGT::Serializer::writeAsText(os, childrenSize);
      os << " ";
      for (size_t i = 0; i < childrenSize; i++) {
	getChildren()[i].serializeAsText(os);
	os << " ";
      }
      for (size_t i = 0; i < childrenSize - 1; i++) {
	NGT::Serializer::writeAsText(os, getBorders()[i]);
	os << " ";
      }
    }
    void deserializeAsText(std::ifstream &is, ObjectSpace *objectspace = 0) {
      Node::deserializeAsText(is);
      if (pivot == 0) {
	pivot = PersistentObject::allocate(*objectspace);
      }
      assert(objectspace != 0);
      getPivot().deserializeAsText(is, objectspace);
      size_t csize;
      NGT::Serializer::readAsText(is, csize);
      assert(children != 0);
      assert(childrenSize == csize);
      for (size_t i = 0; i < childrenSize; i++) {
	getChildren()[i].deserializeAsText(is);
      }
      assert(borders != 0);
      for (size_t i = 0; i < childrenSize - 1; i++) {
	NGT::Serializer::readAsText(is, getBorders()[i]);
      }
    }

    void show() {
      std::cout << "Show internal node " << childrenSize << ":";
      for (size_t i = 0; i < childrenSize; i++) {
	std::cout << getChildren()[i].getID() << " ";
      }
      std::cout << std::endl;
    }

    bool verify(Repository<InternalNode> &internalNodes, Repository<LeafNode> &leafNodes);

    static const int InternalChildrenSizeMax	= 5;
    const size_t	childrenSize;
    ID			*children;
    Distance		*borders;
  };


  class LeafNode : public Node {
  public:
    LeafNode(NGT::ObjectSpace *os = 0) {
      id = 0;
      id.setType(ID::Leaf);
      pivot = 0;
#ifdef NGT_NODE_USE_VECTOR
      objectIDs.reserve(LeafObjectsSizeMax);
#else
      objectSize = 0;
      objectIDs = new NGT::ObjectDistance[LeafObjectsSizeMax];
#endif
    }

    ~LeafNode() {
#ifndef NGT_NODE_USE_VECTOR
      if (objectIDs != 0) {
        delete[] objectIDs;
      }
#endif
    }

    static int
      selectPivotByMaxDistance(Container &iobj, Node::Objects &fs);

    static int
      selectPivotByMaxVariance(Container &iobj, Node::Objects &fs);

    static void
      splitObjects(Container &insertedObject, Objects &splitObjectSet, int pivot);

    void removeObject(size_t id, size_t replaceId);

    NGT::ObjectDistance *getObjectIDs() { return objectIDs; }

    void serialize(std::ofstream &os, ObjectSpace *objectspace = 0) {
      Node::serialize(os);
#ifdef NGT_NODE_USE_VECTOR
      NGT::Serializer::write(os, objectIDs);
#else
      NGT::Serializer::write(os, objectSize);
      for (int i = 0; i < objectSize; i++) {
	objectIDs[i].serialize(os);
      }
#endif // NGT_NODE_USE_VECTOR
      if (pivot == 0) {
	// Before insertion, parent ID == 0 and object size == 0, that indicates an empty index
	if (parent.getID() != 0 || objectSize != 0) {
	  POLARIS_THROW_EX("Node::write: pivot is null!");
	}
      } else {
	assert(objectspace != 0);
	pivot->serialize(os, objectspace);
      }
    }
    void deserialize(std::ifstream &is, ObjectSpace *objectspace = 0) {
      Node::deserialize(is);

#ifdef NGT_NODE_USE_VECTOR
      objectIDs.clear();
      NGT::Serializer::read(is, objectIDs);
#else
      assert(objectIDs != 0);
      NGT::Serializer::read(is, objectSize);
      for (int i = 0; i < objectSize; i++) {
	getObjectIDs()[i].deserialize(is);
      }
#endif
      if (parent.getID() == 0 && objectSize == 0) {
	// The index is empty
	return;
      }
      if (pivot == 0) {
	pivot = PersistentObject::allocate(*objectspace);
	assert(pivot != 0);
      }
      assert(objectspace != 0);
      getPivot().deserialize(is, objectspace);
    }

    void serializeAsText(std::ofstream &os, ObjectSpace *objectspace = 0) {
      Node::serializeAsText(os);
      os << " ";
      if (pivot == 0) {
        POLARIS_THROW_EX("Node::write: pivot is null!");
      }
      assert(pivot != 0);
      assert(objectspace != 0);
      pivot->serializeAsText(os, objectspace);
      os << " ";
#ifdef NGT_NODE_USE_VECTOR
      NGT::Serializer::writeAsText(os, objectIDs);
#else
      NGT::Serializer::writeAsText(os, objectSize);
      for (int i = 0; i < objectSize; i++) {
	os << " ";
	objectIDs[i].serializeAsText(os);
      }
#endif
    }

    void deserializeAsText(std::ifstream &is, ObjectSpace *objectspace = 0) {
      Node::deserializeAsText(is);
      if (pivot == 0) {
	pivot = PersistentObject::allocate(*objectspace);
      }
      assert(objectspace != 0);
      getPivot().deserializeAsText(is, objectspace);
#ifdef NGT_NODE_USE_VECTOR
      objectIDs.clear();
      NGT::Serializer::readAsText(is, objectIDs);
#else
      assert(objectIDs != 0);
      NGT::Serializer::readAsText(is, objectSize);
      for (int i = 0; i < objectSize; i++) {
	getObjectIDs()[i].deserializeAsText(is);
      }
#endif
    }

    void show() {
      std::cout << "Show leaf node " << objectSize << ":";
      for (int i = 0; i < objectSize; i++) {
	std::cout << getObjectIDs()[i].id << "," << getObjectIDs()[i].distance << " ";
      }
      std::cout << std::endl;
    }

    bool verify(size_t nobjs, std::vector<uint8_t> &status);


#ifdef NGT_NODE_USE_VECTOR
    size_t getObjectSize() { return objectIDs.size(); }
#else
    size_t getObjectSize() { return objectSize; }
#endif

    static const size_t LeafObjectsSizeMax		= 100;

#ifdef NGT_NODE_USE_VECTOR
    std::vector<Object>	objectIDs;
#else
    unsigned short	objectSize;
    ObjectDistance	*objectIDs;
#endif
  };


}
