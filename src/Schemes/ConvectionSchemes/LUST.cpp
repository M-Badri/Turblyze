/******************************************************************************

                                     Turblyze
                           3D incompressible CFD solver
                       Copyright (C) 2025-2026 Mohamed Mousa
                        SPDX-License-Identifier: Apache-2.0

 ------------------------------------------------------------------------------
 * @file LUST.cpp
 * @brief Implementation of the LUST convection scheme
 *****************************************************************************/

// ********************************** Headers *********************************

// Implementation header
#include "LUST.h"

// Project headers
#include "Mesh.h"
#include "LinearInterpolation.h"

// ****************************** Public Methods ******************************

Scalar LUST::correction
(
    const Mesh& m,
    const Face& f,
    const ScalarField& phi,
    const Vector& gradPhiP,
    const Vector& gradPhiN,
    Scalar flowRate
) const
{
    // 1. Central Difference deferred correction:
    const Scalar phiFaceCentral = interpolateToFace(m, f, phi);

    const Index upwindCell =
        (flowRate >= S(0.0)) ? f.ownerCell() : f.neighborCell().value();

    const Scalar phiFaceUDS = phi[upwindCell];
    const Scalar corrCDS = flowRate * (phiFaceCentral - phiFaceUDS);

    // 2. Second-Order Linear Upwind deferred correction:
    const Scalar gradientProjection =
        (flowRate >= S(0.0))
      ? dot(gradPhiP, m.dPf(f))
      : dot(gradPhiN, m.dNf(f));

    const Scalar corrLU = flowRate * gradientProjection;

    // 3. Blend: alpha * CDS + (1 - alpha) * LUD
    return alpha_ * corrCDS + (S(1.0) - alpha_) * corrLU;
}
