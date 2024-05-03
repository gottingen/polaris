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
#include <polaris/graph/ngt/node.h>

#include <string>
#include <vector>
#include <stack>
#include <set>

namespace NGT {

  class DVPTree {

  public:
    enum SplitMode {
      MaxDistance	= 0,
      MaxVariance	= 1
    };

    typedef std::vector<Node::ID>	IDVector;

    class Container : public NGT::Container {
    public:
      Container(Object &f, ObjectID i):NGT::Container(f, i) {}
      DVPTree			*vptree;
    };

    class SearchContainer : public NGT::SearchContainer {
    public:
      enum Mode {
	SearchLeaf	= 0,
	SearchObject	= 1
      };

      SearchContainer(Object &f, ObjectID i):NGT::SearchContainer(f, i) {}
      SearchContainer(Object &f):NGT::SearchContainer(f, 0) {}

      DVPTree			*vptree;

      Mode		mode;
      Node::ID	nodeID;
    };
    class InsertContainer : public Container {
    public:
      InsertContainer(Object &f, ObjectID i):Container(f, i) {}
    };

    class RemoveContainer : public Container {
    public:
      RemoveContainer(Object &f, ObjectID i):Container(f, i) {}
    };

    DVPTree() {
      leafObjectsSize = LeafNode::LeafObjectsSizeMax;
      internalChildrenSize = InternalNode::InternalChildrenSizeMax;
      splitMode = MaxVariance;
      insertNode(new LeafNode);
    }

    virtual ~DVPTree() {
      deleteAll();
    }

    void deleteAll() {
      for (size_t i = 0; i < leafNodes.size(); i++) {
	if (leafNodes[i] != 0) {
	  leafNodes[i]->deletePivot(*objectSpace);
	  delete leafNodes[i];
	}
      }
      leafNodes.clear();
      for (size_t i = 0; i < internalNodes.size(); i++) {
	if (internalNodes[i] != 0) {
	  internalNodes[i]->deletePivot(*objectSpace);
	  delete internalNodes[i];
	}
      }
      internalNodes.clear();
    }


    void insert(InsertContainer &iobj);

    void insert(InsertContainer &iobj, LeafNode *n);

    Node::ID split(InsertContainer &iobj, LeafNode &leaf);

    Node::ID recombineNodes(InsertContainer &ic, Node::Objects &fs, LeafNode &leaf);

    void insertObject(InsertContainer &obj, LeafNode &leaf);

    typedef std::stack<Node::ID> UncheckedNode;

    void search(SearchContainer &so);
    void search(SearchContainer &so, InternalNode &node, UncheckedNode &uncheckedNode);
    void search(SearchContainer &so, LeafNode &node, UncheckedNode &uncheckedNode);

    bool searchObject(ObjectID id) {
      LeafNode &ln = getLeaf(id);
      for (size_t i = 0; i < ln.getObjectSize(); i++) {
	if (ln.getObjectIDs()[i].id == id) {
	  return true;
	}
      }
      return false;
    }

    LeafNode &getLeaf(ObjectID id) {
      SearchContainer q(*getObjectRepository().get(id));
      q.mode = SearchContainer::SearchLeaf;
      q.vptree = this;
      q.radius = 0.0;
      q.size = 1;

      search(q);

      return *(LeafNode*)getNode(q.nodeID);

    }

    void replace(ObjectID id, ObjectID replacedId) { remove(id, replacedId); }

    // remove the specified object.
    void remove(ObjectID id, ObjectID replaceId = 0) {
      LeafNode &ln = getLeaf(id);
      try {
	ln.removeObject(id, replaceId);
      } catch(polaris::PolarisException &err) {
	std::stringstream msg;
	msg << "VpTree::remove: Inner error. Cannot remove object. leafNode=" << ln.id.getID() << ":" << err.what();
	POLARIS_THROW_EX(msg);
      }
      if (ln.getObjectSize() == 0) {
	if (ln.parent.getID() != 0) {
	  InternalNode &inode = *(InternalNode*)getNode(ln.parent);
	  removeEmptyNodes(inode);
	}
      }

      return;
    }

    void removeNaively(ObjectID id, ObjectID replaceId = 0) {
      for (size_t i = 0; i < leafNodes.size(); i++) {
	if (leafNodes[i] != 0) {
	  try {
	    leafNodes[i]->removeObject(id, replaceId);
	    break;
	  } catch(...) {}
	}
      }
    }

    Node *getRootNode() {
      size_t nid = 1;
      Node *root;
      try {
  	root = internalNodes.get(nid);
      } catch(polaris::PolarisException &err) {
        try {
  	  root = leafNodes.get(nid);
        } catch(polaris::PolarisException &e) {
          std::stringstream msg;
          msg << "VpTree::getRootNode: Inner error. Cannot get a leaf root node. " << nid << ":" << e.what();
          POLARIS_THROW_EX(msg);
        }
      }

      return root;
    }

    InternalNode *createInternalNode() {
      InternalNode *n = new InternalNode(internalChildrenSize);
      insertNode(n);
      return n;
    }

    void
      removeNode(Node::ID id) {
      size_t idx = id.getID();
      if (id.getType() == Node::ID::Leaf) {
	LeafNode &n = *static_cast<LeafNode*>(getNode(id));
	delete n.pivot;
	leafNodes.remove(idx);
      } else {
	InternalNode &n = *static_cast<InternalNode*>(getNode(id));
	delete n.pivot;
	internalNodes.remove(idx);	  
      }
    }

    void removeEmptyNodes(InternalNode &node);

    Node::Objects * getObjects(LeafNode	&n, Container	&iobj);

    void getObjectIDsFromLeaf(Node::ID		nid,      ObjectDistances	&rl) {
      LeafNode &ln = *(LeafNode*)getNode(nid);
      rl.clear();
      ObjectDistance	r;
      for (size_t i = 0; i < ln.getObjectSize(); i++) {
        r.id = ln.getObjectIDs()[i].id;
        r.distance = ln.getObjectIDs()[i].distance;
        rl.push_back(r);
      }
      return;
    }
    void
      insertNode(LeafNode *n) {
      size_t id = leafNodes.insert(n);
      n->id.setID(id);
      n->id.setType(Node::ID::Leaf);
    }

    // replace
    void replaceNode(LeafNode *n) {
      int id = n->id.getID();
      leafNodes[id] = n;
    }

    void
      insertNode(InternalNode *n)
    {
      size_t id = internalNodes.insert(n);
      n->id.setID(id);
      n->id.setType(Node::ID::Internal);
    }

    Node *getNode(Node::ID &id) {
      Node *n = 0;
      Node::NodeID idx = id.getID();
      if (id.getType() == Node::ID::Leaf) {
	n = leafNodes.get(idx);
      } else {
	n = internalNodes.get(idx);
      }
      return n;
    }

    void getAllLeafNodeIDs(std::vector<Node::ID> &leafIDs) {
      leafIDs.clear();
      Node *root = getRootNode();
      if (root->id.getType() == Node::ID::Leaf) {
	leafIDs.push_back(root->id);
	return;
      }
      UncheckedNode uncheckedNode;
      uncheckedNode.push(root->id);
      while (!uncheckedNode.empty()) {
	Node::ID nodeid = uncheckedNode.top();
	uncheckedNode.pop();
	Node *cnode = getNode(nodeid);
	if (cnode->id.getType() == Node::ID::Internal) {
	  InternalNode &inode = static_cast<InternalNode&>(*cnode);
	  for (size_t ci = 0; ci < internalChildrenSize; ci++) {
	    uncheckedNode.push(inode.getChildren()[ci]);
	  }
	} else if (cnode->id.getType() == Node::ID::Leaf) {
	  leafIDs.push_back(static_cast<LeafNode&>(*cnode).id);
	} else {
	  std::cerr << "Tree: Inner fatal error!: Node type error!" << std::endl;
	  abort();
	}
      }
    }

    void serialize(std::ofstream &os) {
      leafNodes.serialize(os, objectSpace);
      internalNodes.serialize(os, objectSpace);
    }

    void deserialize(std::ifstream &is) {
      leafNodes.deserialize(is, objectSpace);
      internalNodes.deserialize(is, objectSpace);
    }

    void serializeAsText(std::ofstream &os) {
      leafNodes.serializeAsText(os, objectSpace);
      internalNodes.serializeAsText(os, objectSpace);
    }

    void deserializeAsText(std::ifstream &is) {
      leafNodes.deserializeAsText(is, objectSpace);
      internalNodes.deserializeAsText(is, objectSpace);
    }

    void show() {
      std::cout << "Show tree " << std::endl;
      for (size_t i = 0; i < leafNodes.size(); i++) {
	if (leafNodes[i] != 0) {
	  std::cout << i << ":";
	  (*leafNodes[i]).show();
	}
      }
      for (size_t i = 0; i < internalNodes.size(); i++) {
	if (internalNodes[i] != 0) {
	  std::cout << i << ":";
	  (*internalNodes[i]).show();
	}
      }
    }

    bool verify(size_t objCount, std::vector<uint8_t> &status) {
      std::cerr << "Started verifying internal nodes. size=" << internalNodes.size() << "..." << std::endl;
      bool valid = true;
      for (size_t i = 0; i < internalNodes.size(); i++) {
	if (internalNodes[i] != 0) {
	  valid = valid && (*internalNodes[i]).verify(internalNodes, leafNodes);
	}
      }
      std::cerr << "Started verifying leaf nodes. size=" << leafNodes.size() << " ..." << std::endl;
      for (size_t i = 0; i < leafNodes.size(); i++) {
	if (leafNodes[i] != 0) {
	  valid = valid && (*leafNodes[i]).verify(objCount, status);
	}
      }
      return valid;
    }

    void deleteInMemory() {
      for (std::vector<NGT::LeafNode*>::iterator i = leafNodes.begin(); i != leafNodes.end(); i++) {
	if ((*i) != 0) {
	  delete (*i);
	}
      }
      leafNodes.clear();
      for (std::vector<NGT::InternalNode*>::iterator i = internalNodes.begin(); i != internalNodes.end(); i++) {
	if ((*i) != 0) {
	  delete (*i);
	}
      }
      internalNodes.clear();
    }

    ObjectRepository &getObjectRepository() { return objectSpace->getRepository(); }

    size_t getSharedMemorySize(std::ostream &os, SharedMemoryAllocator::GetMemorySizeType t) {
      return 0;
    }

    void getAllObjectIDs(std::set<ObjectID> &ids) {
      for (size_t i = 0; i < leafNodes.size(); i++) {
	if (leafNodes[i] != 0) {
	  LeafNode &ln = *leafNodes[i];
	  auto objs = ln.getObjectIDs();
	  for (size_t idx = 0; idx < ln.objectSize; ++idx) {
	    ids.insert(objs[idx].id);
	  }
	}
      }
    }

  public:
    size_t		internalChildrenSize;
    size_t		leafObjectsSize;

    SplitMode		splitMode;

    std::string		name;

    Repository<LeafNode>	leafNodes;
    Repository<InternalNode>	internalNodes;

    ObjectSpace		*objectSpace;

  };
} // namespace DVPTree


