/******************************************************************************

                                     Turblyze
                           3D incompressible CFD solver
                       Copyright (C) 2025-2026 Mohamed Mousa
                        SPDX-License-Identifier: Apache-2.0

 ------------------------------------------------------------------------------
 * @file HaloExchange.h
 * @brief Ambient parallel halo exchange and cut topology management
 *
 * @details Encapsulates the processor patches (cuts with neighboring ranks)
 * and ghost global indices as an ambient runtime service. Performs
 * non-blocking MPI halo exchanges of cell-centered fields across partition
 * boundaries. In serial runs, all exchange operations are immediate no-ops.
 * Zero MPI headers are exposed to callers.
 *****************************************************************************/

#pragma once

// ********************************** Headers *********************************

// Standard library headers
#include <initializer_list>

// Project headers
#include "Integer.h"
#include "CellData.h"
#include "ProcessorPatch.h"

// ******************************* namespace Halo *****************************

namespace Halo
{
    /// Set the partition's cut metadata and ghost global indices
    void init(ProcessorPatchList patches, IndexList ghostGlobalIndices = {});

    /// Reset the partition metadata (e.g. at end of run or test teardown)
    void reset() noexcept;

    /// Whether this rank has no neighbor cuts (e.g. serial run)
    [[nodiscard]] bool empty() noexcept;

    /// Number of processor patches (neighbor cuts)
    [[nodiscard]] Count numPatches() noexcept;

    /// Processor patch cuts
    [[nodiscard]] const ProcessorPatchList& processorPatches() noexcept;

    /// Global cell index of each ghost cell (for PETSc matrix assembly)
    [[nodiscard]] const IndexList& ghostGlobalIndices() noexcept;

    /// Exchange halo values across all processor patches (no-op if empty)
    void exchange(std::initializer_list<ScalarField*> fields);
    void exchange(std::initializer_list<VectorField*> fields);
    void exchange(std::initializer_list<TensorField*> fields);

    /// Convenience single-field overloads
    inline void exchange(ScalarField& field) { exchange({&field}); }
    inline void exchange(VectorField& field) { exchange({&field}); }
    inline void exchange(TensorField& field) { exchange({&field}); }

} // namespace Halo

// Backward-compatible free-function alias
template<typename T>
inline void exchangeHalos(std::initializer_list<CellData<T>*> fields)
{
    Halo::exchange(fields);
}
