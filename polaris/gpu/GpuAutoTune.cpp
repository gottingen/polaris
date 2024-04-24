/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <polaris/gpu/GpuAutoTune.h>
#include <typeinfo>

#include <polaris/index_pre_transform.h>
#include <polaris/index_replicas.h>
#include <polaris/index_shards.h>
#include <polaris/index_shards_ivf.h>

#include <polaris/gpu/GpuIndex.h>
#include <polaris/gpu/GpuIndexFlat.h>
#include <polaris/gpu/GpuIndexIVFFlat.h>
#include <polaris/gpu/GpuIndexIVFPQ.h>
#include <polaris/gpu/GpuIndexIVFScalarQuantizer.h>
#include <polaris/gpu/impl/IndexUtils.h>
#include <polaris/gpu/utils/DeviceUtils.h>
#include <polaris/impl/faiss_assert.h>

namespace polaris {
namespace gpu {

using namespace ::faiss;

/**********************************************************
 * Parameters to auto-tune on GpuIndex'es
 **********************************************************/

#define DC(classname) auto ix = dynamic_cast<const classname*>(index)

void GpuParameterSpace::initialize(const Index* index) {
    if (DC(IndexPreTransform)) {
        initialize(ix->index);
        return;
    }
    if (DC(IndexShardsIVF)) {
        ParameterSpace::initialize(index);
        return;
    }
    if (DC(IndexReplicas)) {
        if (ix->count() == 0)
            return;
        index = ix->at(0);
    }
    if (DC(IndexShards)) {
        if (ix->count() == 0)
            return;
        index = ix->at(0);
    }
    if (DC(GpuIndexIVF)) {
        ParameterRange& pr = add_range("nprobe");
        for (int i = 0; i < 12; i++) {
            size_t nprobe = 1 << i;
            if (nprobe >= ix->getNumLists() || nprobe > getMaxKSelection())
                break;
            pr.values.push_back(nprobe);
        }

        ParameterSpace ivf_pspace;
        ivf_pspace.initialize(ix->quantizer);

        for (const ParameterRange& p : ivf_pspace.parameter_ranges) {
            ParameterRange& pr = add_range("quantizer_" + p.name);
            pr.values = p.values;
        }
    }
    // not sure we should call the parent initializer
}

#undef DC
// non-const version
#define DC(classname) auto* ix = dynamic_cast<classname*>(index)

void GpuParameterSpace::set_index_parameter(
        Index* index,
        const std::string& name,
        double val) const {
    if (DC(IndexReplicas)) {
        for (int i = 0; i < ix->count(); i++)
            set_index_parameter(ix->at(i), name, val);
        return;
    }
    if (name == "nprobe") {
        if (DC(GpuIndexIVF)) {
            ix->nprobe = size_t(val);
            return;
        }
    }
    if (name == "use_precomputed_table") {
        if (DC(GpuIndexIVFPQ)) {
            ix->setPrecomputedCodes(bool(val));
            return;
        }
    }

    if (name.find("quantizer_") == 0) {
        if (DC(GpuIndexIVF)) {
            std::string sub_name = name.substr(strlen("quantizer_"));
            set_index_parameter(ix->quantizer, sub_name, val);
            return;
        }
    }

    // maybe normal index parameters apply?
    ParameterSpace::set_index_parameter(index, name, val);
}

} // namespace gpu
} // namespace polaris
