/******************************************************************************

                                     Turblyze
                           3D incompressible CFD solver
                       Copyright (C) 2025-2026 Mohamed Mousa
                        SPDX-License-Identifier: Apache-2.0

 ------------------------------------------------------------------------------
 * @file WALE.cpp
 * @brief Implementation of the WALE sub-grid scale turbulence model
 *****************************************************************************/

// ********************************** Headers *********************************

// Implementation header
#include "WALE.h"

// Standard library headers
#include <algorithm>
#include <cmath>

// ************************* Special Member Functions *************************

WALE::WALE
(
    const Mesh& mesh,
    const BoundaryConditions& bc,
    Scalar nu,
    Scalar Cw,
    Scalar Ck,
    bool debug
)
:
    LES{mesh, bc, nu, debug},
    Cw_{Cw},
    Ck_{Ck}
{}

// ***************************** Protected Methods ****************************

Scalar WALE::SdMagnitudeSquared(const Tensor& gradU) noexcept
{
    const Tensor gradU2 = dot(gradU, gradU);
    const Scalar third = gradU2.trace() / S(3.0);
    const Tensor symmG2 = gradU2.symm();
    const Tensor Sd
    {
        symmG2.xx() - third, symmG2.xy(),         symmG2.xz(),
        symmG2.yx(),         symmG2.yy() - third, symmG2.yz(),
        symmG2.zx(),         symmG2.zy(),         symmG2.zz() - third
    };

    return Sd.magnitudeSquared();
}


void WALE::updateSgsViscosity(const TensorField& gradU)
{
    const Count numOwned = mesh_.numDomainCells();

    for (Index cellIdx = 0; cellIdx < numOwned; ++cellIdx)
    {
        const Tensor& g = gradU[cellIdx];
        const Scalar magSqrS = g.symm().magnitudeSquared();
        const Scalar magSqrSd = SdMagnitudeSquared(g);
        const Scalar lw = Cw_ * delta_[cellIdx];

        const Scalar denom =
            std::pow(magSqrS, S(2.5))
          + std::pow(magSqrSd, S(1.25))
          + vSmallValue;

        nut_[cellIdx] = lw * lw * std::pow(magSqrSd, S(1.5)) / denom;
    }
}


void WALE::updateSgsKineticEnergy(const TensorField& /*gradU*/)
{
    const Count numOwned = mesh_.numDomainCells();

    for (Index cellIdx = 0; cellIdx < numOwned; ++cellIdx)
    {
        const Scalar denom = std::max(Ck_ * delta_[cellIdx], vSmallValue);
        const Scalar vSgs = nut_[cellIdx] / denom;
        kSgs_[cellIdx] = vSgs * vSgs;
    }
}
