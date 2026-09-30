/******************************************************************************

                                     Turblyze
                           3D incompressible CFD solver
                       Copyright (C) 2025-2026 Mohamed Mousa
                        SPDX-License-Identifier: Apache-2.0

 ------------------------------------------------------------------------------
 * @file MeshCountTests.cpp
 * @brief Unit tests for the static cell/face counts that size field containers
 *****************************************************************************/

// ********************************** Headers *********************************

// External library headers
#include <catch2/catch_test_macros.hpp>

// Project headers
#include "MeshFixtures.h"
#include "Mesh.h"
#include "CellData.h"
#include "Vector.h"

// ****************************** Mesh Counts *********************************

TEST_CASE("Mesh counts report correct dimensions", "[mesh]")
{
    const TestMesh box(2, 2, 2);

    // 2x2x2 = 8 cells; 12 internal + 24 boundary faces = 36
    REQUIRE(box.mesh().numCells() == 8);
    REQUIRE(box.mesh().numFaces() == 36);
    REQUIRE(box.mesh().numDomainCells() == box.mesh().numCells());
    REQUIRE(box.mesh().numHaloCells() == 0);
    REQUIRE(!box.mesh().isDecomposed());
}

// ************************ Multiple Coexisting Meshes ************************

TEST_CASE("Multiple meshes coexist simultaneously", "[mesh]")
{
    const TestMesh box1(2, 2, 2);
    const TestMesh box2(3, 1, 1);

    REQUIRE(box1.mesh().numCells() == 8);
    REQUIRE(box1.mesh().numFaces() == 36);

    REQUIRE(box2.mesh().numCells() == 3);
    REQUIRE(box2.mesh().numFaces() == 16);
}

// ******************************* Field Sizing *******************************

TEST_CASE("Field containers size from the mesh", "[mesh]")
{
    const TestMesh box(2, 1, 1);

    const ScalarField s(box.mesh());
    REQUIRE(s.size() == box.mesh().numCells());

    const VectorField v(box.mesh(), Vector(S(1.0), S(0.0), S(0.0)));
    REQUIRE(v.size() == box.mesh().numCells());
    REQUIRE(v[0] == Vector(S(1.0), S(0.0), S(0.0)));
}