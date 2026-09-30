/******************************************************************************

                                     Turblyze
                           3D incompressible CFD solver
                       Copyright (C) 2025-2026 Mohamed Mousa
                        SPDX-License-Identifier: Apache-2.0

 ------------------------------------------------------------------------------
 * @file HaloExchangeTests.cpp
 * @brief Unit tests for Halo::exchange non-blocking communication
 *
 * @details Uses the 1D chain fixture (makeDecomposedChainMesh) so topology is
 * minimal (one owned cell, one or two ghost stubs) and received values are
 * checked analytically. After exchange, ghost cell p must hold
 * the neighbour rank's owned cell value.
 *
 * Runs only under mpirun (at least two ranks). Each test verifies that the
 * receive lands in the correct ghost slot and that nothing is clobbered.
 *****************************************************************************/

// ********************************** Headers *********************************

// External library headers
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

// Project headers
#include "Comm.h"
#include "HaloExchange.h"
#include "MeshFixtures.h"
#include "Field.h"
#include "TestTolerances.h"

// ****************************** Internal Helpers ****************************

namespace
{

using Catch::Matchers::WithinRel;

/// Known test value: unique per rank, separated across ranks by 1
[[nodiscard]] Scalar chainValue(Scalar base, Index globalId) noexcept
{
    return base + S(globalId);
}

} // namespace

// **************************** Scalar Ghost Fill ****************************

TEST_CASE("exchangeHalos fills scalar ghosts from the neighbour", "[mpi][parallel]")
{
    if (!Comm::parallelRun())
    {
        SKIP("halo exchange is not available for a single rank");
    }

    const DecomposedChainMesh chain;
    const Mesh& mesh = chain.mesh();

    ScalarField phi(mesh);
    phi[0] = chainValue(S(100.0), Comm::myProcessorNum());

    Halo::exchange({&phi});

    for (const ProcessorPatch& patch : Halo::processorPatches())
    {
        REQUIRE_THAT
        (
            phi[patch.ghostFirstCell()],
            WithinRel
            (
                chainValue(S(100.0),
                patch.neighborRank()),
                TestTolerances::relTight
            )
        );
    }
}

// **************************** Vector Ghost Fill ****************************

TEST_CASE
(
    "exchangeHalos fills vector ghosts from the neighbour",
    "[mpi][parallel]"
)
{
    if (!Comm::parallelRun())
    {
        SKIP("halo exchange is a no-op on a single rank");
    }

    const DecomposedChainMesh chain;
    const Mesh& mesh = chain.mesh();

    VectorField velocity(mesh);
    velocity[0] =
        Vector(chainValue(S(100.0), Comm::myProcessorNum()), S(0.0), S(0.0));

    Halo::exchange({&velocity});

    for (const ProcessorPatch& patch : Halo::processorPatches())
    {
        const Vector& ghost = velocity[patch.ghostFirstCell()];
        REQUIRE_THAT
        (
            ghost.x(),
            WithinRel
            (
                chainValue(S(100.0),
                patch.neighborRank()),
                TestTolerances::relTight
            )
        );
        REQUIRE(ghost.y() == S(0.0));
        REQUIRE(ghost.z() == S(0.0));
    }
}

// ************************** Batched Two-Field Fill *************************

TEST_CASE("exchangeHalos fills two batched fields at once", "[mpi][parallel]")
{
    if (!Comm::parallelRun())
    {
        SKIP("halo exchange is a no-op on a single rank");
    }

    const DecomposedChainMesh chain;
    const Mesh& mesh = chain.mesh();

    const Index rank = Comm::myProcessorNum();
    ScalarField a(mesh);
    ScalarField b(mesh);
    a[0] = chainValue(S(100.0), rank);
    b[0] = chainValue(S(200.0), rank);

    // One message per neighbour carries both fields, laid out [a ghosts|b ghosts]
    Halo::exchange({&a, &b});

    for (const ProcessorPatch& patch : Halo::processorPatches())
    {
        REQUIRE_THAT
        (
            a[patch.ghostFirstCell()],
            WithinRel
            (
                chainValue(S(100.0),
                patch.neighborRank()),
                TestTolerances::relTight
            )
        );
        REQUIRE_THAT
        (
            b[patch.ghostFirstCell()],
            WithinRel
            (
                chainValue(S(200.0),
                patch.neighborRank()),
                TestTolerances::relTight
            )
        );
    }
}