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

#include <polaris/graph/ngt/tree.h>
#include <polaris/graph/ngt/node.h>

#include <vector>

using namespace std;
using namespace NGT;

void
DVPTree::insert(InsertContainer &iobj) {
  SearchContainer q(iobj.object);
  q.mode = SearchContainer::SearchLeaf;
  q.vptree = this;
  q.radius = 0.0;

  search(q);

  iobj.vptree = this;

  assert(q.nodeID.getType() == Node::ID::Leaf);
  LeafNode *ln = (LeafNode*)getNode(q.nodeID);
  insert(iobj, ln);

  return;
}

void
DVPTree::insert(InsertContainer &iobj,  LeafNode *leafNode)
{
  LeafNode &leaf = *leafNode;
  size_t fsize = leaf.getObjectSize();
  if (fsize != 0) {
    NGT::ObjectSpace::Comparator &comparator = objectSpace->getComparator();
    Distance d = comparator(iobj.object, leaf.getPivot());

    NGT::ObjectDistance *objects = leaf.getObjectIDs();

    for (size_t i = 0; i < fsize; i++) {
      if (objects[i].distance == d) {
	Distance idd = 0.0;
	ObjectID loid;
        try {
	  loid = objects[i].id;
	  if (objectSpace->isNormalizedDistance()) {
	    idd = objectSpace->compareWithL1(iobj.object, *getObjectRepository().get(loid));
	  } else {
	    idd = comparator(iobj.object, *getObjectRepository().get(loid));
	  }
        } catch (Exception &e) {
          stringstream msg;
          msg << "LeafNode::insert: Cannot find object which belongs to a leaf node. id="
              << objects[i].id << ":" << e.what() << endl;
          NGTThrowException(msg.str());
        }
        if (idd == 0.0) {
	  if (loid == iobj.id) {
	    stringstream msg;
	    msg << "DVPTree::insert:already existed. " << iobj.id;
	    NGTThrowException(msg);
	  }
	  return;
        }
      }
    }
  }

  if (leaf.getObjectSize() >= leafObjectsSize) {
    split(iobj, leaf);
  } else {
    insertObject(iobj, leaf);
  }

  return;
}
Node::ID
DVPTree::split(InsertContainer &iobj, LeafNode &leaf)
{
  Node::Objects *fs = getObjects(leaf, iobj);
  int pv = DVPTree::MaxVariance;
  switch (splitMode) {
  case DVPTree::MaxVariance:
    pv = LeafNode::selectPivotByMaxVariance(iobj, *fs);
    break;
  case DVPTree::MaxDistance:
    pv = LeafNode::selectPivotByMaxDistance(iobj, *fs);
    break;
  }

  LeafNode::splitObjects(iobj, *fs, pv);

  Node::ID nid = recombineNodes(iobj, *fs, leaf);
  delete fs;

  return nid;
}

Node::ID
DVPTree::recombineNodes(InsertContainer &ic, Node::Objects &fs, LeafNode &leaf)
{
  LeafNode *ln[internalChildrenSize];
  Node::ID targetParent = leaf.parent;
  Node::ID targetId = leaf.id;
  ln[0] = &leaf;
  ln[0]->objectSize = 0;
  for (size_t i = 1; i < internalChildrenSize; i++) {
    ln[i] = new LeafNode;
  }
  InternalNode *in = createInternalNode();
  Node::ID inid = in->id;
  try {
    if (targetParent.getID() != 0) {
      InternalNode &pnode = *(InternalNode*)getNode(targetParent);
      for (size_t i = 0; i < internalChildrenSize; i++) {
	if (pnode.getChildren()[i] == targetId) {
	  pnode.getChildren()[i] = inid;
	  break;
	}
      }
    }
    in->setPivot(*getObjectRepository().get(fs[0].id), *objectSpace);

    in->parent = targetParent;

    int fsize = fs.size();
    int cid = fs[0].clusterID;
#ifdef NGT_NODE_USE_VECTOR
    LeafNode::ObjectIDs fid;
    fid.id = fs[0].id;
    fid.distance = 0.0;
    ln[cid]->objectIDs.push_back(fid);
#else
    ln[cid]->getObjectIDs()[ln[cid]->objectSize].id = fs[0].id;
    ln[cid]->getObjectIDs()[ln[cid]->objectSize++].distance = 0.0;
#endif
    if (fs[0].leafDistance == Node::Object::Pivot) {
      ln[cid]->setPivot(*getObjectRepository().get(fs[0].id), *objectSpace);
    } else {
      NGTThrowException("recombineNodes: internal error : illegal pivot.");
    }
    ln[cid]->parent = inid;
    int maxClusterID = cid;
    for (int i = 1; i < fsize; i++) {
      int clusterID = fs[i].clusterID;
      if (clusterID > maxClusterID) {
	maxClusterID = clusterID;
      }
      Distance ld;
      if (fs[i].leafDistance == Node::Object::Pivot) {
        // pivot
	ln[clusterID]->setPivot(*getObjectRepository().get(fs[i].id), *objectSpace);
        ld = 0.0;
      } else {
        ld = fs[i].leafDistance;
      }

#ifdef NGT_NODE_USE_VECTOR
      fid.id = fs[i].id;
      fid.distance = ld;
      ln[clusterID]->objectIDs.push_back(fid);
#else
      ln[clusterID]->getObjectIDs()[ln[clusterID]->objectSize].id = fs[i].id;
      ln[clusterID]->getObjectIDs()[ln[clusterID]->objectSize++].distance = ld;
#endif
      ln[clusterID]->parent = inid;
      if (clusterID != cid) {
        in->getBorders()[cid] = fs[i].distance;
        cid = fs[i].clusterID;
      }
    }
    // When the number of the children is less than the expected,
    // proper values are set to the empty children.
    for (size_t i = maxClusterID + 1; i < internalChildrenSize; i++) {
      ln[i]->parent = inid;
      // dummy
      ln[i]->setPivot(*getObjectRepository().get(fs[0].id), *objectSpace);
      if (i < (internalChildrenSize - 1)) {
	in->getBorders()[i] = FLT_MAX;
      }
    }

    in->getChildren()[0] = targetId;
    for (size_t i = 1; i < internalChildrenSize; i++) {
      insertNode(ln[i]);
      in->getChildren()[i] = ln[i]->id;
    }
  } catch(Exception &e) {
    throw e;
  }
  return inid;
}

void
DVPTree::insertObject(InsertContainer &ic, LeafNode &leaf) {
  if (leaf.getObjectSize() == 0) {
    leaf.setPivot(*getObjectRepository().get(ic.id), *objectSpace);
#ifdef NGT_NODE_USE_VECTOR
    LeafNode::ObjectIDs fid;
    fid.id = ic.id;
    fid.distance = 0;
    leaf.objectIDs.push_back(fid);
#else
    leaf.getObjectIDs()[leaf.objectSize].id = ic.id;
    leaf.getObjectIDs()[leaf.objectSize++].distance = 0;
#endif
  } else {
    Distance d = objectSpace->getComparator()(ic.object, leaf.getPivot());

#ifdef NGT_NODE_USE_VECTOR
    LeafNode::ObjectIDs fid;
    fid.id = ic.id;
    fid.distance = d;
    leaf.objectIDs.push_back(fid);
    std::sort(leaf.objectIDs.begin(), leaf.objectIDs.end(), LeafNode::ObjectIDs());
#else
    leaf.getObjectIDs()[leaf.objectSize].id = ic.id;
    leaf.getObjectIDs()[leaf.objectSize++].distance = d;
#endif
  }
}

Node::Objects *
DVPTree::getObjects(LeafNode &n, Container &iobj)
{
  int size = n.getObjectSize() + 1;

  Node::Objects *fs = new Node::Objects(size);
  for (size_t i = 0; i < n.getObjectSize(); i++) {
    (*fs)[i].object = getObjectRepository().get(n.getObjectIDs()[i].id);
    (*fs)[i].id = n.getObjectIDs()[i].id;
  }
  (*fs)[n.getObjectSize()].object = &iobj.object;
  (*fs)[n.getObjectSize()].id = iobj.id;
  return fs;
}

void
DVPTree::removeEmptyNodes(InternalNode &inode) {

  int csize = internalChildrenSize;
  InternalNode *target = &inode;

  for(;;) {
    Node::ID *children = target->getChildren();
    for (int i = 0; i < csize; i++) {
      if (children[i].getType() == Node::ID::Internal) {
	return;
      }
      LeafNode &ln = *static_cast<LeafNode*>(getNode(children[i]));
      if (ln.getObjectSize() != 0) {
	return;
      }
    }

    for (int i = 0; i < csize; i++) {
      removeNode(children[i]);
    }
    if (target->parent.getID() == 0) {
      removeNode(target->id);
      LeafNode *root = new LeafNode;
      insertNode(root);
      if (root->id.getID() != 1) {
	NGTThrowException("Root id Error");
      }
      return;
    }

    LeafNode *ln = new LeafNode;
    ln->parent = target->parent;
    insertNode(ln);

    InternalNode &in = *(InternalNode*)getNode(ln->parent);
    in.updateChild(*this, target->id, ln->id);
    removeNode(target->id);
    target = &in;
  }

  return;
}


void
DVPTree::search(SearchContainer &sc, InternalNode &node, UncheckedNode &uncheckedNode)
{
  Distance d = objectSpace->getComparator()(sc.object, node.getPivot());
#ifdef NGT_DISTANCE_COMPUTATION_COUNT
  sc.distanceComputationCount++;
#endif

  int bsize = internalChildrenSize - 1;

  vector<ObjectDistance> regions;
  regions.reserve(internalChildrenSize);

  ObjectDistance child;
  Distance *borders = node.getBorders();
  int mid;
  for (mid = 0; mid < bsize; mid++) {
    if (d < borders[mid]) {
        child.id = mid;
        child.distance = 0.0;
        regions.push_back(child);
      if (d + sc.radius < borders[mid]) {
        break;
      } else {
        continue;
      }
    } else {
      if (d < borders[mid] + sc.radius) {
        child.id = mid;
        child.distance = d - borders[mid];
        regions.push_back(child);
        continue;
      } else {
        continue;
      }
    }
  }

  if (mid == bsize) {
    if (d >= borders[mid - 1]) {
      child.id = mid;
      child.distance = 0.0;
      regions.push_back(child);
    } else {
      child.id = mid;
      child.distance = borders[mid - 1] - d;
      regions.push_back(child);
    }
  }

  sort(regions.begin(), regions.end());

  Node::ID *children = node.getChildren();

  vector<ObjectDistance>::iterator i;
  if (sc.mode == DVPTree::SearchContainer::SearchLeaf) {
    if (children[regions.front().id].getType() == Node::ID::Leaf) {
      sc.nodeID.setRaw(children[regions.front().id].get());
      assert(uncheckedNode.empty());
    } else {
      uncheckedNode.push(children[regions.front().id]);
    }
  } else {
    for (i = regions.begin(); i != regions.end(); i++) {
      uncheckedNode.push(children[i->id]);
    }
  }
  
}

void
DVPTree::search(SearchContainer &so, LeafNode &node, UncheckedNode &uncheckedNode)
{
  DVPTree::SearchContainer &q = (DVPTree::SearchContainer&)so;

  if (node.getObjectSize() == 0) {
    return;
  }
  Distance pq = objectSpace->getComparator()(q.object, node.getPivot());
#ifdef NGT_DISTANCE_COMPUTATION_COUNT
  so.distanceComputationCount++;
#endif

  ObjectDistance r;
  NGT::ObjectDistance *objects = node.getObjectIDs();

  for (size_t i = 0; i < node.getObjectSize(); i++) {
    if ((objects[i].distance <= pq + q.radius) &&
        (objects[i].distance >= pq - q.radius)) {
      Distance d = 0;
      try {
	d = objectSpace->getComparator()(q.object, *q.vptree->getObjectRepository().get(objects[i].id));
#ifdef NGT_DISTANCE_COMPUTATION_COUNT
	so.distanceComputationCount++;
#endif
      } catch(...) {
        NGTThrowException("VpTree::LeafNode::search: Internal fatal error : Cannot get object");
      }
      if (d <= q.radius) {
        r.id = objects[i].id;
        r.distance = d;
	so.getResult().push_back(r);
	std::sort(so.getResult().begin(), so.getResult().end());
	if (so.getResult().size() > q.size) {
	  so.getResult().resize(q.size);
	}
      }
    }
  }
}

void
DVPTree::search(SearchContainer &sc) {
  ((SearchContainer&)sc).vptree = this;
  Node *root = getRootNode();
  assert(root != 0);
  if (sc.mode == DVPTree::SearchContainer::SearchLeaf) {
    if (root->id.getType() == Node::ID::Leaf) {
      sc.nodeID.setRaw(root->id.get());
      return;
    }
  }

  UncheckedNode uncheckedNode;
  uncheckedNode.push(root->id);

  while (!uncheckedNode.empty()) {
    Node::ID nodeid = uncheckedNode.top();
    uncheckedNode.pop();
    Node *cnode = getNode(nodeid);
    if (cnode == 0) {
      cerr << "Error! child node is null. but continue." << endl;
      continue;
    }
    if (cnode->id.getType() == Node::ID::Internal) {
      search(sc, (InternalNode&)*cnode, uncheckedNode);
    } else if (cnode->id.getType() == Node::ID::Leaf) {
      search(sc, (LeafNode&)*cnode, uncheckedNode);
    } else {
      cerr << "Tree: Inner fatal error!: Node type error!" << endl;
      abort();
    }
  }
}

