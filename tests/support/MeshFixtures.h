/******************************************************************************

                                     Turblyze
                           3D incompressible CFD solver
                       Copyright (C) 2025-2026 Mohamed Mousa
                        SPDX-License-Identifier: Apache-2.0

 ------------------------------------------------------------------------------
 * @file MeshFixtures.h
 * @brief Programmatic hex-box meshes for unit tests
 *****************************************************************************/

#pragma once

// ********************************** Headers *********************************

// Project headers
#include "Mesh.h"
#include "MeshCreator.h"
#include "Scalar.h"
#include "Integer.h"

// ********************************** BoxPatch ********************************

// The six axis-aligned boundary patches of a hex box, in emission order
namespace BoxPatch
{

inline const Name xMin = "xMin";
inline const Name xMax = "xMax";
inline const Name yMin = "yMin";
inline const Name yMax = "yMax";
inline const Name zMin = "zMin";
inline const Name zMax = "zMax";

} // namespace BoxPatch

// **************************** Fixture Factories *****************************

/// Build a structured hex box of nx*ny*nz unit-spacing cells at the origin.
[[nodiscard]] Mesh makeHexBoxMesh
(
    Count nx,
    Count ny,
    Count nz,
    Scalar spacing = S(1.0)
);

/// Build this rank's submesh of a 1D cell chain
[[nodiscard]] Mesh makeDecomposedChainMesh();

/// Build this rank's submesh of the same hex box, partitioned by METIS
[[nodiscard]] Mesh makeDecomposedHexBoxMesh
(
    Count nx,
    Count ny,
    Count nz,
    Scalar spacing = S(1.0)
);

// ****************************** class TestMesh ******************************

class TestMesh
{
public:

// ************************* Special Member Functions *************************

    /// Build a hex box
    TestMesh(Count nx, Count ny, Count nz, Scalar spacing = S(1.0))
    :
        mesh_(makeHexBoxMesh(nx, ny, nz, spacing))
    {}

    /// Not copyable
    TestMesh(const TestMesh&) = delete;
    TestMesh& operator=(const TestMesh&) = delete;

    /// Not movable
    TestMesh(TestMesh&&) = delete;
    TestMesh& operator=(TestMesh&&) = delete;

    /// Destructor
    ~TestMesh() noexcept = default;

// ***************************** Accessor Methods *****************************

    /// The owned mesh
    [[nodiscard]] const Mesh& mesh() const noexcept
    {
        return mesh_;
    }

// ****************************** Private Members *****************************

private:

    /// The single populated mesh
    Mesh mesh_;
};

// ************************* class DecomposedChainMesh ************************

class DecomposedChainMesh
{
public:

// ************************* Special Member Functions *************************

    /// Build this rank's chain submesh
    DecomposedChainMesh()
    :
        mesh_(makeDecomposedChainMesh())
    {}

    /// Not copyable
    DecomposedChainMesh(const DecomposedChainMesh&) = delete;
    DecomposedChainMesh& operator=(const DecomposedChainMesh&) = delete;

    /// Not movable
    DecomposedChainMesh(DecomposedChainMesh&&) = delete;
    DecomposedChainMesh& operator=(DecomposedChainMesh&&) = delete;

    /// Destructor
    ~DecomposedChainMesh() noexcept = default;

// ***************************** Accessor Methods *****************************

    /// The owned submesh
    [[nodiscard]] const Mesh& mesh() const noexcept
    {
        return mesh_;
    }

// ****************************** Private Members *****************************

private:

    Mesh mesh_;
};

// ************************** class DecomposedBoxMesh *************************

class DecomposedBoxMesh
{
public:

// ************************* Special Member Functions *************************

    /// Build this rank's share of a hex box
    DecomposedBoxMesh(Count nx, Count ny, Count nz, Scalar spacing = S(1.0))
    :
        mesh_(makeDecomposedHexBoxMesh(nx, ny, nz, spacing))
    {}

    /// Not copyable
    DecomposedBoxMesh(const DecomposedBoxMesh&) = delete;
    DecomposedBoxMesh& operator=(const DecomposedBoxMesh&) = delete;

    /// Not movable
    DecomposedBoxMesh(DecomposedBoxMesh&&) = delete;
    DecomposedBoxMesh& operator=(DecomposedBoxMesh&&) = delete;

    /// Destructor
    ~DecomposedBoxMesh() noexcept = default;

// ***************************** Accessor Methods *****************************

    /// The owned submesh
    [[nodiscard]] const Mesh& mesh() const noexcept
    {
        return mesh_;
    }

// ****************************** Private Members *****************************

private:

    Mesh mesh_;
};