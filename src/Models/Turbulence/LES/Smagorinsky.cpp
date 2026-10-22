/******************************************************************************

                                     Turblyze
                           3D incompressible CFD solver
                       Copyright (C) 2025-2026 Mohamed Mousa
                        SPDX-License-Identifier: Apache-2.0

 ------------------------------------------------------------------------------
 * @file Smagorinsky.cpp
 * @brief Implementation of the classical Smagorinsky SGS model
 *****************************************************************************/

// ********************************** Headers *********************************

// Implementation header
#include "Smagorinsky.h"

// Standard library headers
#include <algorithm>
#include <cmath>

// ************************* Special Member Functions *************************

Smagorinsky::Smagorinsky
(
    const Mesh& mesh,
    const BoundaryConditions& bc,
    Scalar nu,
    Scalar Cs,
    Scalar Ck,
    bool vanDriest,
    Scalar Aplus,
    Scalar kappa,
    bool debug
)
:
    LES{mesh, bc, nu, debug},
    Cs_{Cs},
    Ck_{Ck},
    vanDriest_{vanDriest},
    Aplus_{Aplus},
    kappa_{kappa},
    deltaEff_{delta_}
{}

// ***************************** Protected Methods ****************************

void Smagorinsky::updateSgsViscosity(const TensorField& gradU)
{
    const Count numOwned = mesh_.numDomainCells();

    for (Index cellIdx = 0; cellIdx < numOwned; ++cellIdx)
    {
        const Scalar magS = strainRateMagnitude(gradU[cellIdx]);
        const Scalar d = delta_[cellIdx];

        Scalar ls = Cs_ * d;

        if (vanDriest_)
        {
            // Local-shear wall coordinate: u_tau ~ sqrt(nu * |S|)
            const Scalar y = wallDistance_[cellIdx];
            const Scalar uTau = std::sqrt(nu_ * magS);
            const Scalar yPlus = (nu_ > S(0.0)) ? (y * uTau / nu_) : S(0.0);
            const Scalar fd = S(1.0) - std::exp(-yPlus / Aplus_);
            const Scalar lsWall = kappa_ * y * fd;

            ls = std::min(ls, lsWall);
        }

        deltaEff_[cellIdx] = (Cs_ > S(0.0)) ? (ls / Cs_) : d;
        nut_[cellIdx] = ls * ls * magS;
    }
}


void Smagorinsky::updateSgsKineticEnergy(const TensorField& gradU)
{
    const Count numOwned = mesh_.numDomainCells();

    for (Index cellIdx = 0; cellIdx < numOwned; ++cellIdx)
    {
        const Scalar magS = strainRateMagnitude(gradU[cellIdx]);
        const Scalar dEff = deltaEff_[cellIdx];

        // Yoshizawa equilibrium SGS TKE: k_sgs = Ck * delta_eff^2 * |S|^2
        kSgs_[cellIdx] = Ck_ * dEff * dEff * magS * magS;
    }
}
