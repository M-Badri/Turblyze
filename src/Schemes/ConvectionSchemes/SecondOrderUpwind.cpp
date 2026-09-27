/******************************************************************************

                                     Turblyze
                           3D incompressible CFD solver
                       Copyright (C) 2025-2026 Mohamed Mousa
                        SPDX-License-Identifier: Apache-2.0

 ------------------------------------------------------------------------------
 * @file SecondOrderUpwind.cpp
 * @brief Implementation of the second-order upwind convection scheme
 *****************************************************************************/

// ********************************** Headers *********************************

#include "SecondOrderUpwind.h"
#include "Mesh.h"

// ****************************** Public Methods ******************************

Scalar SecondOrderUpwind::correction
(
    const Mesh& m,
    const Face& f,
    const ScalarField& /*phi*/,
    const Vector& gradPhiP,
    const Vector& gradPhiN,
    Scalar flowRate
) const
{
    // Deferred correction: flowRate * grad(phi)_upwind dot d_upwind_to_face
    const Scalar gradientProjection =
        (flowRate >= S(0.0))
      ? dot(gradPhiP, m.dPf(f))
      : dot(gradPhiN, m.dNf(f));

    return flowRate * gradientProjection;
}
