/******************************************************************************

                                     Turblyze
                           3D incompressible CFD solver
                       Copyright (C) 2025-2026 Mohamed Mousa
                        SPDX-License-Identifier: Apache-2.0

 ------------------------------------------------------------------------------
 * @file DerivedFields.cpp
 * @brief Implementation of derived cell-centered scalar fields
 *****************************************************************************/

// ********************************** Headers *********************************

// Implementation header
#include "DerivedFields.h"

// Standard library headers
#include <cmath>

// Project headers
#include "Mesh.h"
#include "Tensor.h"

// ******************************* namespace VTK ******************************

namespace VTK
{

ScalarField velocityMagnitude
(
    const Mesh& mesh,
    const ScalarField& Ux,
    const ScalarField& Uy,
    const ScalarField& Uz
)
{
    ScalarField result(mesh);
    for (Index cellIdx = 0; cellIdx < mesh.numCells(); ++cellIdx)
    {
        result[cellIdx] = std::sqrt
        (
            Ux[cellIdx] * Ux[cellIdx]
          + Uy[cellIdx] * Uy[cellIdx]
          + Uz[cellIdx] * Uz[cellIdx]
        );
    }
    return result;
}

ScalarField vorticityMagnitude(const Mesh& mesh, const VectorField& vorticity)
{
    ScalarField result(mesh);
    for (Index cellIdx = 0; cellIdx < mesh.numCells(); ++cellIdx)
    {
        result[cellIdx] = magnitude(vorticity[cellIdx]);
    }
    return result;
}

ScalarField QCriterion
(
    const Mesh& mesh,
    const VectorField& gradUx,
    const VectorField& gradUy,
    const VectorField& gradUz
)
{
    ScalarField qCriterion(mesh);

    for (Index cellIdx = 0; cellIdx < mesh.numCells(); ++cellIdx)
    {
        // Q = 0.5 * (||Omega||^2 - ||S||^2)
        const Tensor gradU =
            tensorFromRows(gradUx[cellIdx], gradUy[cellIdx], gradUz[cellIdx]);

        const Scalar sMagSq = gradU.symm().magnitudeSquared();
        const Scalar oMagSq = gradU.skew().magnitudeSquared();

        qCriterion[cellIdx] = S(0.5) * (oMagSq - sMagSq);
    }

    return qCriterion;
}

ScalarField strainRateMagnitude
(
    const Mesh& mesh,
    const VectorField& gradUx,
    const VectorField& gradUy,
    const VectorField& gradUz
)
{
    ScalarField strainRateMag(mesh);

    for (Index cellIdx = 0; cellIdx < mesh.numCells(); ++cellIdx)
    {
        // Strain rate magnitude = sqrt(2 * S_ij * S_ij)
        const Tensor gradU = tensorFromRows
        (
            gradUx[cellIdx],
            gradUy[cellIdx],
            gradUz[cellIdx]
        );

        const Scalar symmMagSq = gradU.symm().magnitudeSquared();
        strainRateMag[cellIdx] = std::sqrt(S(2.0) * symmMagSq);
    }

    return strainRateMag;
}

} // namespace VTK
