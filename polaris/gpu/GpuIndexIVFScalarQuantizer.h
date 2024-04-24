/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <polaris/IndexScalarQuantizer.h>
#include <polaris/gpu/GpuIndexIVF.h>
#include <memory>

namespace polaris {
namespace gpu {

class IVFFlat;
class GpuIndexFlat;

struct GpuIndexIVFScalarQuantizerConfig : public GpuIndexIVFConfig {
    /// Use the alternative memory layout for the IVF lists
    /// (currently the default)
    bool interleavedLayout = true;
};

/// Wrapper around the GPU implementation that looks like
/// polaris::IndexIVFScalarQuantizer
class GpuIndexIVFScalarQuantizer : public GpuIndexIVF {
   public:
    /// Construct from a pre-existing polaris::IndexIVFScalarQuantizer instance,
    /// copying data over to the given GPU, if the input index is trained.
    GpuIndexIVFScalarQuantizer(
            GpuResourcesProvider* provider,
            const polaris::IndexIVFScalarQuantizer* index,
            GpuIndexIVFScalarQuantizerConfig config =
                    GpuIndexIVFScalarQuantizerConfig());

    /// Constructs a new instance with an empty flat quantizer; the user
    /// provides the number of IVF lists desired.
    GpuIndexIVFScalarQuantizer(
            GpuResourcesProvider* provider,
            int dims,
            idx_t nlist,
            polaris::ScalarQuantizer::QuantizerType qtype,
            polaris::MetricType metric = MetricType::METRIC_L2,
            bool encodeResidual = true,
            GpuIndexIVFScalarQuantizerConfig config =
                    GpuIndexIVFScalarQuantizerConfig());

    /// Constructs a new instance with a provided CPU or GPU coarse quantizer;
    /// the user provides the number of IVF lists desired.
    GpuIndexIVFScalarQuantizer(
            GpuResourcesProvider* provider,
            Index* coarseQuantizer,
            int dims,
            idx_t nlist,
            polaris::ScalarQuantizer::QuantizerType qtype,
            polaris::MetricType metric = MetricType::METRIC_L2,
            bool encodeResidual = true,
            GpuIndexIVFScalarQuantizerConfig config =
                    GpuIndexIVFScalarQuantizerConfig());

    ~GpuIndexIVFScalarQuantizer() override;

    /// Reserve GPU memory in our inverted lists for this number of vectors
    void reserveMemory(size_t numVecs);

    /// Initialize ourselves from the given CPU index; will overwrite
    /// all data in ourselves
    void copyFrom(const polaris::IndexIVFScalarQuantizer* index);

    /// Copy ourselves to the given CPU index; will overwrite all data
    /// in the index instance
    void copyTo(polaris::IndexIVFScalarQuantizer* index) const;

    /// After adding vectors, one can call this to reclaim device memory
    /// to exactly the amount needed. Returns space reclaimed in bytes
    size_t reclaimMemory();

    /// Clears out all inverted lists, but retains the coarse and scalar
    /// quantizer information
    void reset() override;

    /// Should be called if the user ever changes the state of the IVF coarse
    /// quantizer manually (e.g., substitutes a new instance or changes vectors
    /// in the coarse quantizer outside the scope of training)
    void updateQuantizer() override;

    /// Trains the coarse and scalar quantizer based on the given vector data
    void train(idx_t n, const float* x) override;

   protected:
    /// Validates index SQ parameters
    void verifySQSettings_() const;

    /// Called from train to handle SQ residual training
    void trainResiduals_(idx_t n, const float* x);

   public:
    /// Exposed like the CPU version
    polaris::ScalarQuantizer sq;

    /// Exposed like the CPU version
    bool by_residual;

   protected:
    /// Our configuration options
    const GpuIndexIVFScalarQuantizerConfig ivfSQConfig_;

    /// Desired inverted list memory reservation
    size_t reserveMemoryVecs_;

    /// Instance that we own; contains the inverted list
    std::shared_ptr<IVFFlat> index_;
};

} // namespace gpu
} // namespace polaris
