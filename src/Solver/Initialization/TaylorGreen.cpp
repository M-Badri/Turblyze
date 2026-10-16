/******************************************************************************

                                     Turblyze
                           3D incompressible CFD solver
                       Copyright (C) 2025-2026 Mohamed Mousa
                        SPDX-License-Identifier: Apache-2.0

 ------------------------------------------------------------------------------
 * @file TaylorGreen.cpp
 * @brief Implementation of Taylor-Green vortex field initialization
 *****************************************************************************/

// ********************************** Headers *********************************

// Implementation header
#include "TaylorGreen.h"

// Standard library headers
#include <cmath>

// Project headers
#include "HaloExchange.h"

// ************************* namespace Initialization *************************

namespace Initialization
{

// *************************** Initialization Method **************************

void TaylorGreen::initialize
(
    const Mesh& mesh,
    BoundaryConditions& /*bc*/,
    ScalarField& Ux,
    ScalarField& Uy,
    ScalarField& Uz,
    ScalarField& p
) const
{
    const Count numCells = mesh.numDomainCells();
    const auto& cells = mesh.cells();
    const Scalar pFactor = rho_ * U0_ * U0_ / S(16.0);

    for (Index cellIdx = 0; cellIdx < numCells; ++cellIdx)
    {
        const Vector center = cells[cellIdx].centroid();
        const Scalar x = center.x();
        const Scalar y = center.y();
        const Scalar z = center.z();

        Ux[cellIdx] =
            U0_ * std::sin(x / L_) * std::cos(y / L_) * std::cos(z / L_);
        Uy[cellIdx] =
            -U0_ * std::cos(x / L_) * std::sin(y / L_) * std::cos(z / L_);
        Uz[cellIdx] = S(0.0);

        p[cellIdx] =
            p0_ + pFactor
          * (std::cos(S(2.0) * x / L_) + std::cos(S(2.0) * y / L_))
          * (std::cos(S(2.0) * z / L_) + S(2.0));
    }

    // exchange halos across partition cuts
    Halo::exchange({&Ux, &Uy, &Uz, &p});
}

} // namespace Initialization
