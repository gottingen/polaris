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

#include <polaris/graph/ngt/node.h>
#include <polaris/graph/ngt/tree.h>

#include <algorithm>

namespace polaris {

    const double Node::Object::Pivot = -1.0;


    void
    InternalNode::updateChild(DVPTree &dvptree, Node::ID src, Node::ID dst) {
        int cs = dvptree.internalChildrenSize;
        for (int i = 0; i < cs; i++) {
            if (getChildren()[i] == src) {
                getChildren()[i] = dst;
                return;
            }
        }
    }

    int
    LeafNode::selectPivotByMaxDistance(Container &c, Node::Objects &fs) {
        DVPTree::InsertContainer &iobj = (DVPTree::InsertContainer &) c;
        int fsize = fs.size();
        distance_t maxd = 0.0;
        int maxid = 0;
        for (int i = 1; i < fsize; i++) {
            distance_t d = iobj.vptree->objectSpace->getComparator()(*fs[0].object, *fs[i].object);
            if (d >= maxd) {
                maxd = d;
                maxid = i;
            }
        }

        int aid = maxid;
        maxd = 0.0;
        maxid = 0;
        for (int i = 0; i < fsize; i++) {
            distance_t d = iobj.vptree->objectSpace->getComparator()(*fs[aid].object, *fs[i].object);
            if (i == aid) {
                continue;
            }
            if (d >= maxd) {
                maxd = d;
                maxid = i;
            }
        }

        int bid = maxid;
        maxd = 0.0;
        maxid = 0;
        for (int i = 0; i < fsize; i++) {
            distance_t d = iobj.vptree->objectSpace->getComparator()(*fs[bid].object, *fs[i].object);
            if (i == bid) {
                continue;
            }
            if (d >= maxd) {
                maxd = d;
                maxid = i;
            }
        }
        return maxid;
    }

    int
    LeafNode::selectPivotByMaxVariance(Container &c, Node::Objects &fs) {
        DVPTree::InsertContainer &iobj = (DVPTree::InsertContainer &) c;

        int fsize = fs.size();
        distance_t *distance = new distance_t[fsize * fsize];

        for (int i = 0; i < fsize; i++) {
            distance[i * fsize + i] = 0;
        }

        for (int i = 0; i < fsize; i++) {
            for (int j = i + 1; j < fsize; j++) {
                distance_t d = iobj.vptree->objectSpace->getComparator()(*fs[i].object, *fs[j].object);
                distance[i * fsize + j] = d;
                distance[j * fsize + i] = d;
            }
        }

        double *variance = new double[fsize];
        for (int i = 0; i < fsize; i++) {
            double avg = 0.0;
            for (int j = 0; j < fsize; j++) {
                avg += distance[i * fsize + j];
            }
            avg /= (double) fsize;

            double v = 0.0;
            for (int j = 0; j < fsize; j++) {
                v += pow(distance[i * fsize + j] - avg, 2.0);
            }
            variance[i] = v / (double) fsize;
        }

        double maxv = variance[0];
        int maxid = 0;
        for (int i = 0; i < fsize; i++) {
            if (variance[i] > maxv) {
                maxv = variance[i];
                maxid = i;
            }
        }
        delete[] variance;
        delete[] distance;

        return maxid;
    }

    void
    LeafNode::splitObjects(Container &c, Objects &fs, int pv) {
        DVPTree::InsertContainer &iobj = (DVPTree::InsertContainer &) c;

        // sort the objects by distance
        int fsize = fs.size();
        for (int i = 0; i < fsize; i++) {
            if (i == pv) {
                fs[i].distance = 0;
            } else {
                distance_t d = iobj.vptree->objectSpace->getComparator()(*fs[pv].object, *fs[i].object);
                fs[i].distance = d;
            }
        }

        sort(fs.begin(), fs.end());

        int childrenSize = iobj.vptree->internalChildrenSize;
        int cid = childrenSize - 1;
        int cms = (fsize * cid) / childrenSize;

        // divide the objects into child clusters.
        fs[fsize - 1].clusterID = cid;
        for (int i = fsize - 2; i >= 0; i--) {
            if (i < cms && cid > 0) {
                if (fs[i].distance != fs[i + 1].distance) {
                    cid--;
                    cms = (fsize * cid) / childrenSize;
                }
            }
            fs[i].clusterID = cid;
        }

        if (cid != 0) {
            // the required number of child nodes could not be acquired
            std::stringstream msg;
            msg
                    << "LeafNode::splitObjects: Too many same distances. Reduce internal children size for the tree index or not use the tree index."
                    << std::endl;
            msg << "  internalChildrenSize=" << childrenSize << std::endl;
            msg << "  # of the children=" << (childrenSize - cid) << std::endl;
            msg << "  Size=" << fsize <<std:: endl;
            msg << "  pivot=" << pv << std::endl;
            msg << "  cluster id=" << cid << std::endl;
            msg << "  Show distances for debug." << std::endl;
            for (int i = 0; i < fsize; i++) {
                msg << "  " << fs[i].id << ":" << fs[i].distance << std::endl;
                msg << "  ";
                PersistentObject &po = *fs[i].object;
                iobj.vptree->objectSpace->show(msg, po);
                msg << std::endl;
            }
            if (fs[fsize - 1].clusterID == cid) {
                msg << "LeafNode::splitObjects: All of the object distances are the same!" << std::endl;;
                POLARIS_THROW_EX(msg.str());
            } else {
                std::cerr << msg.str() << std::endl;
                std::cerr << "LeafNode::splitObjects: Anyway, continue..." << std::endl;
                // sift the cluster IDs to start from 0 to continue.
                for (int i = 0; i < fsize; i++) {
                    fs[i].clusterID -= cid;
                }
            }
        }

        long long *pivots = new long long[childrenSize];
        for (int i = 0; i < childrenSize; i++) {
            pivots[i] = -1;
        }

        // find the boundaries for the subspaces
        for (int i = 0; i < fsize; i++) {
            if (pivots[fs[i].clusterID] == -1) {
                pivots[fs[i].clusterID] = i;
                fs[i].leafDistance = Object::Pivot;
            } else {
                distance_t d = iobj.vptree->objectSpace->getComparator()(*fs[pivots[fs[i].clusterID]].object,
                                                                       *fs[i].object);
                fs[i].leafDistance = d;
            }
        }
        delete[] pivots;

        return;
    }

    void
    LeafNode::removeObject(size_t id, size_t replaceId) {

        size_t fsize = getObjectSize();
        size_t idx;
        if (replaceId != 0) {
            for (idx = 0; idx < fsize; idx++) {
                if (getObjectIDs()[idx].id == replaceId) {
                    std::cerr << " Warning. found the same ID as the replaced ID. " << id << ":" << replaceId
                              << std::endl;
                    std::cerr << "          ignore it, if normalized distance." << std::endl;
                    replaceId = 0;
                    break;
                }
            }
        }
        for (idx = 0; idx < fsize; idx++) {
            if (getObjectIDs()[idx].id == id) {
                if (replaceId != 0) {
                    getObjectIDs()[idx].id = replaceId;
                    return;
                } else {
                    break;
                }
            }
        }
        if (idx == fsize) {
            if (pivot == 0) {
                POLARIS_THROW_EX("LeafNode::removeObject: Internal error!. the pivot is illegal.");
            }
            std::stringstream msg;
            msg << "VpTree::Leaf::remove: Warning. Cannot find the specified object. ID=" << id << "," << replaceId
                << " idx=" << idx << " If the same objects were inserted into the index, ignore this message.";
            POLARIS_THROW_EX(msg.str());
        }

#ifdef NGT_NODE_USE_VECTOR
        for (; idx < objectIDs.size() - 1; idx++) {
          getObjectIDs()[idx] = getObjectIDs()[idx + 1];
        }
        objectIDs.pop_back();
#else
        objectSize--;
        for (; idx < objectSize; idx++) {
            getObjectIDs()[idx] = getObjectIDs()[idx + 1];
        }
#endif

        return;
    }


    bool InternalNode::verify(Repository<InternalNode> &internalNodes, Repository<LeafNode> &leafNodes) {
        size_t isize = internalNodes.size();
        size_t lsize = leafNodes.size();
        bool valid = true;
        for (size_t i = 0; i < childrenSize; i++) {
            size_t nid = getChildren()[i].getID();
            ID::Type type = getChildren()[i].getType();
            size_t size = type == ID::Leaf ? lsize : isize;
            if (nid >= size) {
                std::cerr << "Error! Internal children node id is too big." << nid << ":" << size << std::endl;
                valid = false;
            }
            try {
                if (type == ID::Leaf) {
                    leafNodes.get(nid);
                } else {
                    internalNodes.get(nid);
                }
            } catch (...) {
                std::cerr << "Error! Cannot get the node. " << ((type == ID::Leaf) ? "Leaf" : "Internal") << std::endl;
                valid = false;
            }
        }
        return valid;
    }


    bool LeafNode::verify(size_t nobjs, std::vector <uint8_t> &status) {
        bool valid = true;
        for (size_t i = 0; i < objectSize; i++) {
            size_t nid = getObjectIDs()[i].id;
            if (nid > nobjs) {
                std::cerr << "Error! Object id is too big. " << nid << ":" << nobjs << std::endl;
                valid = false;
                continue;
            }
            status[nid] |= 0x04;
        }
        return valid;
    }
}  // namespace polaris