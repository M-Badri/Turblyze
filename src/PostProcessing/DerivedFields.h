/******************************************************************************

                                     Turblyze
                           3D incompressible CFD solver
                       Copyright (C) 2025-2026 Mohamed Mousa
                        SPDX-License-Identifier: Apache-2.0

 ------------------------------------------------------------------------------
 * @file DerivedFields.h
 * @brief Cell-centered derived scalar fields for post-processing
 *
 * @details Field math utilities. Produces scalar fields (magnitudes,
 * Q-criterion, strain rate) from velocity or gradient vector fields. 
 * Intended for use ahead of VTK export so it can write the derived
 * quantities into output files.
 *****************************************************************************/

#pragma once

// ********************************** Headers *********************************

// Project headers
#include "CellData.h"
#include "Mesh.h"

// ******************************* namespace VTK ******************************

namespace VTK
{

/// Compute velocity magnitude field from velocity components
[[nodiscard]] ScalarField velocityMagnitude
(
    const Mesh& mesh,
    const ScalarField& Ux,
    const ScalarField& Uy,
    const ScalarField& Uz
);

/// Compute vorticity magnitude field from vorticity vector field
[[nodiscard]] ScalarField vorticityMagnitude
(
    const Mesh& mesh,
    const VectorField& vorticity
);

/// Compute Q-criterion for vortex identification
/// Q = 0.5 * (||Omega||^2 - ||S||^2)
[[nodiscard]] ScalarField QCriterion
(
    const Mesh& mesh,
    const VectorField& gradUx,
    const VectorField& gradUy,
    const VectorField& gradUz
);

/// Compute strain rate magnitude = sqrt(2 * S_ij * S_ij)
[[nodiscard]] ScalarField strainRateMagnitude
(
    const Mesh& mesh,
    const VectorField& gradUx,
    const VectorField& gradUy,
    const VectorField& gradUz
);

} // namespace VTK
