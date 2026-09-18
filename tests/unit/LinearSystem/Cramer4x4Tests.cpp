/******************************************************************************

                                     Turblyze
                           3D incompressible CFD solver
                       Copyright (C) 2025-2026 Mohamed Mousa
                        SPDX-License-Identifier: Apache-2.0

 ------------------------------------------------------------------------------
 * @file Cramer4x4Tests.cpp
 * @brief Unit tests for the 4x4 Cramer's rule solver
 *****************************************************************************/

// ********************************** Headers *********************************

// External library headers
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

// Standard library headers
#include <cmath>

// Project headers
#include "Cramer4x4.h"
#include "TestTolerances.h"

using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;

// ***************************** Identity Matrix ******************************

TEST_CASE("Cramer4x4 identity matrix", "[linear-system]")
{
    // I * x = b  ⟹  x = b
    const Scalar A[16] =
    {
        S(1.0), S(0.0), S(0.0), S(0.0),
        S(0.0), S(1.0), S(0.0), S(0.0),
        S(0.0), S(0.0), S(1.0), S(0.0),
        S(0.0), S(0.0), S(0.0), S(1.0)
    };
    const Scalar b[4] = {S(2.0), S(3.0), S(5.0), S(7.0)};
    Scalar x[4] = {};

    solve4x4(A, b, x);

    REQUIRE(x[0] == S(2.0));
    REQUIRE(x[1] == S(3.0));
    REQUIRE(x[2] == S(5.0));
    REQUIRE(x[3] == S(7.0));
}

// ***************************** Diagonal Matrix ******************************

TEST_CASE("Cramer4x4 diagonal matrix", "[linear-system]")
{
    // diag(2,4,5,10) * x = {6, 20, 15, 100}  ⟹  x = {3, 5, 3, 10}
    const Scalar A[16] =
    {
        S(2.0), S(0.0), S(0.0), S(0.0),
        S(0.0), S(4.0), S(0.0), S(0.0),
        S(0.0), S(0.0), S(5.0), S(0.0),
        S(0.0), S(0.0), S(0.0), S(10.0)
    };
    const Scalar b[4] = {S(6.0), S(20.0), S(15.0), S(100.0)};
    Scalar x[4] = {};

    solve4x4(A, b, x);

    REQUIRE(x[0] == S(3.0));
    REQUIRE(x[1] == S(5.0));
    REQUIRE(x[2] == S(3.0));
    REQUIRE(x[3] == S(10.0));
}

// ************************* Dense Well-Conditioned ***************************

TEST_CASE("Cramer4x4 dense system", "[linear-system]")
{
    // Hand-picked so x = {1, 2, 3, 4} exactly
    //   row 0: 2*1 + 1*2 + 1*3 + 0*4 = 7
    //   row 1: 1*1 + 3*2 + 2*3 + 1*4 = 17
    //   row 2: 0*1 + 1*2 + 2*3 + 3*4 = 20
    //   row 3: 1*1 + 0*2 + 1*3 + 2*4 = 12
    const Scalar A[16] =
    {
        S(2.0), S(1.0), S(1.0), S(0.0),
        S(1.0), S(3.0), S(2.0), S(1.0),
        S(0.0), S(1.0), S(2.0), S(3.0),
        S(1.0), S(0.0), S(1.0), S(2.0)
    };
    const Scalar b[4] = {S(7.0), S(17.0), S(20.0), S(12.0)};
    Scalar x[4] = {};

    solve4x4(A, b, x);

    REQUIRE_THAT(x[0], WithinRel(S(1.0), TestTolerances::relTight));
    REQUIRE_THAT(x[1], WithinRel(S(2.0), TestTolerances::relTight));
    REQUIRE_THAT(x[2], WithinRel(S(3.0), TestTolerances::relTight));
    REQUIRE_THAT(x[3], WithinRel(S(4.0), TestTolerances::relTight));
}

// ************************** Diagonal Dominance ******************************

TEST_CASE("Cramer4x4 diagonally dominant matrix", "[linear-system]")
{
    // Typical of implicit CFD coefficient matrices
    const Scalar A[16] =
    {
        S(10.0), S(-1.0), S( 2.0), S( 0.0),
        S(-1.0), S(11.0), S(-1.0), S( 3.0),
        S( 2.0), S(-1.0), S(10.0), S(-1.0),
        S( 0.0), S( 3.0), S(-1.0), S( 8.0)
    };

    // Construct b from known solution x = {1, 2, -1, 1}
    const Scalar b[4] =
    {
        S(10.0)*S(1.0) + S(-1.0)*S(2.0) + S(2.0)*S(-1.0) + S(0.0)*S(1.0),
        S(-1.0)*S(1.0) + S(11.0)*S(2.0) + S(-1.0)*S(-1.0) + S(3.0)*S(1.0),
        S(2.0)*S(1.0)  + S(-1.0)*S(2.0) + S(10.0)*S(-1.0) + S(-1.0)*S(1.0),
        S(0.0)*S(1.0)  + S(3.0)*S(2.0)  + S(-1.0)*S(-1.0) + S(8.0)*S(1.0)
    };
    Scalar x[4] = {};

    solve4x4(A, b, x);

    REQUIRE_THAT(x[0], WithinRel(S( 1.0), TestTolerances::relTight));
    REQUIRE_THAT(x[1], WithinRel(S( 2.0), TestTolerances::relTight));
    REQUIRE_THAT(x[2], WithinRel(S(-1.0), TestTolerances::relTight));
    REQUIRE_THAT(x[3], WithinRel(S( 1.0), TestTolerances::relTight));
}

// **************************** Negative Entries ******************************

TEST_CASE("Cramer4x4 mixed signs", "[linear-system]")
{
    // Antisymmetric off-diagonals with known solution x = {-2, 3, -5, 7}
    const Scalar A[16] =
    {
        S( 4.0), S(-2.0), S( 1.0), S(-3.0),
        S( 2.0), S( 5.0), S(-1.0), S( 2.0),
        S(-1.0), S( 3.0), S( 6.0), S(-2.0),
        S( 3.0), S(-1.0), S( 2.0), S( 7.0)
    };

    const Scalar xRef[4] = {S(-2.0), S(3.0), S(-5.0), S(7.0)};
    const Scalar b[4] =
    {
        A[0]*xRef[0]  + A[1]*xRef[1]  + A[2]*xRef[2]  + A[3]*xRef[3],
        A[4]*xRef[0]  + A[5]*xRef[1]  + A[6]*xRef[2]  + A[7]*xRef[3],
        A[8]*xRef[0]  + A[9]*xRef[1]  + A[10]*xRef[2] + A[11]*xRef[3],
        A[12]*xRef[0] + A[13]*xRef[1] + A[14]*xRef[2] + A[15]*xRef[3]
    };
    Scalar x[4] = {};

    solve4x4(A, b, x);

    REQUIRE_THAT(x[0], WithinRel(S(-2.0), TestTolerances::relTight));
    REQUIRE_THAT(x[1], WithinRel(S( 3.0), TestTolerances::relTight));
    REQUIRE_THAT(x[2], WithinRel(S(-5.0), TestTolerances::relTight));
    REQUIRE_THAT(x[3], WithinRel(S( 7.0), TestTolerances::relTight));
}

// ****************************** Small Entries *******************************

TEST_CASE("Cramer4x4 small off-diagonals", "[linear-system]")
{
    // Near-diagonal with tiny off-diagonals, x = {1, 1, 1, 1}
    const Scalar eps = S(1.0e-4);
    const Scalar A[16] =
    {
        S(1.0), eps,    eps,    S(0.0),
        eps,    S(1.0), eps,    eps,
        eps,    eps,    S(1.0), eps,
        S(0.0), eps,    eps,    S(1.0)
    };
    const Scalar b[4] =
    {
        S(1.0) + S(2.0) * eps,
        S(1.0) + S(3.0) * eps,
        S(1.0) + S(3.0) * eps,
        S(1.0) + S(2.0) * eps
    };
    Scalar x[4] = {};

    solve4x4(A, b, x);

    REQUIRE_THAT(x[0], WithinRel(S(1.0), TestTolerances::relTight));
    REQUIRE_THAT(x[1], WithinRel(S(1.0), TestTolerances::relTight));
    REQUIRE_THAT(x[2], WithinRel(S(1.0), TestTolerances::relTight));
    REQUIRE_THAT(x[3], WithinRel(S(1.0), TestTolerances::relTight));
}

// **************************** Large Entries *********************************

TEST_CASE("Cramer4x4 large coefficients", "[linear-system]")
{
    // Scaled diagonal, x = {1, 1, 1, 1}
    const Scalar big = S(1.0e6);
    const Scalar A[16] =
    {
        big,    S(1.0), S(0.0), S(0.0),
        S(0.0), big,    S(1.0), S(0.0),
        S(0.0), S(0.0), big,    S(1.0),
        S(1.0), S(0.0), S(0.0), big
    };
    const Scalar b[4] =
    {
        big + S(1.0),
        big + S(1.0),
        big + S(1.0),
        big + S(1.0)
    };
    Scalar x[4] = {};

    solve4x4(A, b, x);

    REQUIRE_THAT(x[0], WithinRel(S(1.0), TestTolerances::relTight));
    REQUIRE_THAT(x[1], WithinRel(S(1.0), TestTolerances::relTight));
    REQUIRE_THAT(x[2], WithinRel(S(1.0), TestTolerances::relTight));
    REQUIRE_THAT(x[3], WithinRel(S(1.0), TestTolerances::relTight));
}

// ************************** Zero Right-Hand Side ****************************

TEST_CASE("Cramer4x4 zero RHS", "[linear-system]")
{
    // Non-singular A with b = 0  ⟹  x = 0
    const Scalar A[16] =
    {
        S(5.0), S(1.0), S(2.0), S(1.0),
        S(1.0), S(6.0), S(1.0), S(2.0),
        S(2.0), S(1.0), S(7.0), S(1.0),
        S(1.0), S(2.0), S(1.0), S(8.0)
    };
    const Scalar b[4] = {S(0.0), S(0.0), S(0.0), S(0.0)};
    Scalar x[4] = {};

    solve4x4(A, b, x);

    REQUIRE_THAT(x[0], WithinAbs(S(0.0), TestTolerances::absTight));
    REQUIRE_THAT(x[1], WithinAbs(S(0.0), TestTolerances::absTight));
    REQUIRE_THAT(x[2], WithinAbs(S(0.0), TestTolerances::absTight));
    REQUIRE_THAT(x[3], WithinAbs(S(0.0), TestTolerances::absTight));
}
