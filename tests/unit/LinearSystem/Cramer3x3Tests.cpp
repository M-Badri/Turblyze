/******************************************************************************

                                     Turblyze
                           3D incompressible CFD solver
                       Copyright (C) 2025-2026 Mohamed Mousa
                        SPDX-License-Identifier: Apache-2.0

 ------------------------------------------------------------------------------
 * @file Cramer3x3Tests.cpp
 * @brief Unit tests for the 3x3 Cramer's rule solver
 *****************************************************************************/

// ********************************** Headers *********************************

// External library headers
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

// Project headers
#include "Cramer3x3.h"
#include "TestTolerances.h"

using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;

// ***************************** Identity Matrix ******************************

TEST_CASE("Cramer3x3 identity matrix", "[linear-system]")
{
    // I * x = b  ->  x = b
    const Scalar A[9] =
    {
        S(1.0), S(0.0), S(0.0),
        S(0.0), S(1.0), S(0.0),
        S(0.0), S(0.0), S(1.0)
    };
    const Scalar b[3] = {S(2.0), S(3.0), S(5.0)};
    Scalar x[3] = {};

    solve3x3(A, b, x);

    REQUIRE(x[0] == S(2.0));
    REQUIRE(x[1] == S(3.0));
    REQUIRE(x[2] == S(5.0));
}

// ***************************** Diagonal Matrix ******************************

TEST_CASE("Cramer3x3 diagonal matrix", "[linear-system]")
{
    // diag(2, 4, 5) * x = {6, 20, 15}  ->  x = {3, 5, 3}
    const Scalar A[9] =
    {
        S(2.0), S(0.0), S(0.0),
        S(0.0), S(4.0), S(0.0),
        S(0.0), S(0.0), S(5.0)
    };
    const Scalar b[3] = {S(6.0), S(20.0), S(15.0)};
    Scalar x[3] = {};

    solve3x3(A, b, x);

    REQUIRE(x[0] == S(3.0));
    REQUIRE(x[1] == S(5.0));
    REQUIRE(x[2] == S(3.0));
}

// ************************* Dense Well-Conditioned ***************************

TEST_CASE("Cramer3x3 dense system", "[linear-system]")
{
    // row 0: 2*1 + 1*2 + 1*3 = 7
    // row 1: 1*1 + 3*2 + 2*3 = 13
    // row 2: 0*1 + 1*2 + 2*3 = 8
    // A * x = b  ->  x = {1, 2, 3}
    const Scalar A[9] =
    {
        S(2.0), S(1.0), S(1.0),
        S(1.0), S(3.0), S(2.0),
        S(0.0), S(1.0), S(2.0)
    };
    const Scalar b[3] = {S(7.0), S(13.0), S(8.0)};
    Scalar x[3] = {};

    solve3x3(A, b, x);

    REQUIRE_THAT(x[0], WithinRel(S(1.0), TestTolerances::relTight));
    REQUIRE_THAT(x[1], WithinRel(S(2.0), TestTolerances::relTight));
    REQUIRE_THAT(x[2], WithinRel(S(3.0), TestTolerances::relTight));
}

// ************************** Diagonal Dominance ******************************

TEST_CASE("Cramer3x3 diagonally dominant matrix", "[linear-system]")
{
    // row 0: 10*1 + -1*2 + 2*-1 = 6
    // row 1: -1*1 + 11*2 + -1*-1 = 20
    // row 2: 2*1 + -1*2 + 10*-1 = 15
    // A * x = b  ->  x = {1, 2, -1} -> known solution
    const Scalar A[9] =
    {
        S(10.0), S(-1.0), S( 2.0),
        S(-1.0), S(11.0), S(-1.0),
        S( 2.0), S(-1.0), S(10.0)
    };

    // Construct b from known solution x = {1, 2, -1}
    const Scalar b[3] =
    {
        S(10.0)*S(1.0) + S(-1.0)*S(2.0) + S( 2.0)*S(-1.0),
        S(-1.0)*S(1.0) + S(11.0)*S(2.0) + S(-1.0)*S(-1.0),
        S( 2.0)*S(1.0) + S(-1.0)*S(2.0) + S(10.0)*S(-1.0)
    };
    Scalar x[3] = {};

    solve3x3(A, b, x);

    REQUIRE_THAT(x[0], WithinRel(S( 1.0), TestTolerances::relTight));
    REQUIRE_THAT(x[1], WithinRel(S( 2.0), TestTolerances::relTight));
    REQUIRE_THAT(x[2], WithinRel(S(-1.0), TestTolerances::relTight));
}

// **************************** Negative Entries ******************************

TEST_CASE("Cramer3x3 mixed signs", "[linear-system]")
{
    // Known solution x = {-2, 3, -5}
    const Scalar A[9] =
    {
        S( 4.0), S(-2.0), S( 1.0),
        S( 2.0), S( 5.0), S(-1.0),
        S(-1.0), S( 3.0), S( 6.0)
    };

    const Scalar xRef[3] = {S(-2.0), S(3.0), S(-5.0)};
    const Scalar b[3] =
    {
        A[0]*xRef[0] + A[1]*xRef[1] + A[2]*xRef[2],
        A[3]*xRef[0] + A[4]*xRef[1] + A[5]*xRef[2],
        A[6]*xRef[0] + A[7]*xRef[1] + A[8]*xRef[2]
    };
    Scalar x[3] = {};

    solve3x3(A, b, x);

    REQUIRE_THAT(x[0], WithinRel(S(-2.0), TestTolerances::relTight));
    REQUIRE_THAT(x[1], WithinRel(S( 3.0), TestTolerances::relTight));
    REQUIRE_THAT(x[2], WithinRel(S(-5.0), TestTolerances::relTight));
}

// ****************************** Small Entries *******************************

TEST_CASE("Cramer3x3 small off-diagonals", "[linear-system]")
{
    // Near-identity with tiny off-diagonals, x = {1, 1, 1}
    const Scalar eps = smallValue;
    const Scalar A[9] =
    {
        S(1.0), eps,    eps,
        eps,    S(1.0), eps,
        eps,    eps,    S(1.0)
    };
    const Scalar b[3] =
    {
        S(1.0) + S(2.0) * eps,
        S(1.0) + S(2.0) * eps,
        S(1.0) + S(2.0) * eps
    };
    Scalar x[3] = {};

    solve3x3(A, b, x);

    REQUIRE_THAT(x[0], WithinRel(S(1.0), TestTolerances::relTight));
    REQUIRE_THAT(x[1], WithinRel(S(1.0), TestTolerances::relTight));
    REQUIRE_THAT(x[2], WithinRel(S(1.0), TestTolerances::relTight));
}

// **************************** Large Entries *********************************

TEST_CASE("Cramer3x3 large coefficients", "[linear-system]")
{
    // Scaled diagonal, x = {1, 1, 1}
    const Scalar big = S(1.0e6);
    const Scalar A[9] =
    {
        big,    S(1.0), S(0.0),
        S(0.0), big,    S(1.0),
        S(1.0), S(0.0), big
    };
    const Scalar b[3] =
    {
        big + S(1.0),
        big + S(1.0),
        big + S(1.0)
    };
    Scalar x[3] = {};

    solve3x3(A, b, x);

    REQUIRE_THAT(x[0], WithinRel(S(1.0), TestTolerances::relTight));
    REQUIRE_THAT(x[1], WithinRel(S(1.0), TestTolerances::relTight));
    REQUIRE_THAT(x[2], WithinRel(S(1.0), TestTolerances::relTight));
}

// ************************** Zero Right-Hand Side ****************************

TEST_CASE("Cramer3x3 zero RHS", "[linear-system]")
{
    // Non-singular A with b = 0  ->  x = 0
    const Scalar A[9] =
    {
        S(5.0), S(1.0), S(2.0),
        S(1.0), S(6.0), S(1.0),
        S(2.0), S(1.0), S(7.0)
    };
    const Scalar b[3] = {S(0.0), S(0.0), S(0.0)};
    Scalar x[3] = {};

    solve3x3(A, b, x);

    REQUIRE_THAT(x[0], WithinAbs(S(0.0), TestTolerances::absTight));
    REQUIRE_THAT(x[1], WithinAbs(S(0.0), TestTolerances::absTight));
    REQUIRE_THAT(x[2], WithinAbs(S(0.0), TestTolerances::absTight));
}
