/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <polaris/gpu/GpuIndexIVF.h>
#include <polaris/impl/ScalarQuantizer.h>

#include <memory>

namespace polaris {
struct IndexIVFFlat;
}

namespace polaris {
namespace gpu {

class IVFFlat;
class GpuIndexFlat;

struct GpuIndexIVFFlatConfig : public GpuIndexIVFConfig {
    /// Use the alternative memory layout for the IVF lists
    /// (currently the default)
    bool interleavedLayout = true;
};

/// Wrapper around the GPU implementation that looks like
/// polaris::IndexIVFFlat
class GpuIndexIVFFlat : public GpuIndexIVF {
   public:
    /// Construct from a pre-existing polaris::IndexIVFFlat instance, copying
    /// data over to the given GPU, if the input index is trained.
    GpuIndexIVFFlat(
            GpuResourcesProvider* provider,
            const polaris::IndexIVFFlat* index,
            GpuIndexIVFFlatConfig config = GpuIndexIVFFlatConfig());

    /// Constructs a new instance with an empty flat quantizer; the user
    /// provides the number of IVF lists desired.
    GpuIndexIVFFlat(
            GpuResourcesProvider* provider,
            int dims,
            idx_t nlist,
            polaris::MetricType metric = polaris::METRIC_L2,
            GpuIndexIVFFlatConfig config = GpuIndexIVFFlatConfig());

    /// Constructs a new instance with a provided CPU or GPU coarse quantizer;
    /// the user provides the number of IVF lists desired.
    GpuIndexIVFFlat(
            GpuResourcesProvider* provider,
            Index* coarseQuantizer,
            int dims,
            idx_t nlist,
            polaris::MetricType metric = polaris::METRIC_L2,
            GpuIndexIVFFlatConfig config = GpuIndexIVFFlatConfig());

    ~GpuIndexIVFFlat() override;

    /// Reserve GPU memory in our inverted lists for this number of vectors
    void reserveMemory(size_t numVecs);

    /// Initialize ourselves from the given CPU index; will overwrite
    /// all data in ourselves
    void copyFrom(const polaris::IndexIVFFlat* index);

    /// Copy ourselves to the given CPU index; will overwrite all data
    /// in the index instance
    void copyTo(polaris::IndexIVFFlat* index) const;

    /// After adding vectors, one can call this to reclaim device memory
    /// to exactly the amount needed. Returns space reclaimed in bytes
    size_t reclaimMemory();

    /// Clears out all inverted lists, but retains the coarse centroid
    /// information
    void reset() override;

    /// Should be called if the user ever changes the state of the IVF coarse
    /// quantizer manually (e.g., substitutes a new instance or changes vectors
    /// in the coarse quantizer outside the scope of training)
    void updateQuantizer() override;

    /// Trains the coarse quantizer based on the given vector data
    void train(idx_t n, const float* x) override;

   protected:
    /// Initialize appropriate index
    void setIndex_(
            GpuResources* resources,
            int dim,
            int nlist,
            polaris::MetricType metric,
            float metricArg,
            bool useResidual,
            /// Optional ScalarQuantizer
            polaris::ScalarQuantizer* scalarQ,
            bool interleavedLayout,
            IndicesOptions indicesOptions,
            MemorySpace space);

   protected:
    /// Our configuration options
    const GpuIndexIVFFlatConfig ivfFlatConfig_;

    /// Desired inverted list memory reservation
    size_t reserveMemoryVecs_;

    /// Instance that we own; contains the inverted lists
    std::shared_ptr<IVFFlat> index_;
};

} // namespace gpu
} // namespace polaris
