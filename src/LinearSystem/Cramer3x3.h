/******************************************************************************

                                     Turblyze
                           3D incompressible CFD solver
                       Copyright (C) 2025-2026 Mohamed Mousa
                        SPDX-License-Identifier: Apache-2.0

 ------------------------------------------------------------------------------
 * @file Cramer3x3.h
 * @brief Cramer's rule implementation for 3x3 linear systems
 *
 * @details Solves A * x = b for a dense 3x3 system in row-major layout:
 * A[0..2] = row 0, A[3..5] = row 1, A[6..8] = row 2.
 * Computes the nine cofactors, reusing three of them for the determinant,
 * then forms adj(A) * b in one pass.
 *****************************************************************************/

#pragma once

// ********************************** Headers *********************************

// Project headers
#include "Scalar.h"

// ********************************** Solver **********************************

/// Solve a dense 3x3 linear system A * x = b via Cramer's rule
inline void solve3x3
(
    const Scalar* __restrict__ A,
    const Scalar* __restrict__ b,
    Scalar* __restrict__ x
)
{
    // Cofactors of column 0
    const Scalar C00 = A[4] * A[8] - A[5] * A[7];
    const Scalar C10 = A[5] * A[6] - A[3] * A[8];
    const Scalar C20 = A[3] * A[7] - A[4] * A[6];

    const Scalar invDet =
        S(1.0) / (A[0] * C00 + A[1] * C10 + A[2] * C20);

    // Remaining cofactors
    const Scalar C01 = A[2] * A[7] - A[1] * A[8];
    const Scalar C11 = A[0] * A[8] - A[2] * A[6];
    const Scalar C21 = A[1] * A[6] - A[0] * A[7];

    const Scalar C02 = A[1] * A[5] - A[2] * A[4];
    const Scalar C12 = A[2] * A[3] - A[0] * A[5];
    const Scalar C22 = A[0] * A[4] - A[1] * A[3];

    // x = adj(A) * b / det
    x[0] = (C00 * b[0] + C01 * b[1] + C02 * b[2]) * invDet;
    x[1] = (C10 * b[0] + C11 * b[1] + C12 * b[2]) * invDet;
    x[2] = (C20 * b[0] + C21 * b[1] + C22 * b[2]) * invDet;
}
