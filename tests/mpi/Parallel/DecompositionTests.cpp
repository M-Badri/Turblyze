/******************************************************************************

                                     Turblyze
                           3D incompressible CFD solver
                       Copyright (C) 2025-2026 Mohamed Mousa
                        SPDX-License-Identifier: Apache-2.0

 ------------------------------------------------------------------------------
 * @file DecompositionTests.cpp
 * @brief Multi-rank tests validating mesh decomposition and inter-rank cuts
 *****************************************************************************/

// ********************************** Headers *********************************

// Standard library headers
#include <algorithm>
#include <cmath>
#include <vector>

// External library headers
#include <catch2/catch_test_macros.hpp>
#include <mpi.h>

// Project headers
#include "BoundaryConditions.h"
#include "Comm.h"
#include "Field.h"
#include "HaloExchange.h"
#include "LeastSquares.h"
#include "MPIScalarType.h"
#include "MeshFixtures.h"
#include "Reduce.h"
#include "Scalar.h"
#include "TestTolerances.h"

// ***************************** Internal Helpers *****************************

namespace
{

constexpr int exchangeTag = 42;

/// Swap scalar payloads across one cut; the two sides may differ in size
[[nodiscard]] std::vector<Scalar> exchangeWithNeighbor
(
    const ProcessorPatch& patch,
    const std::vector<Scalar>& sendBuffer,
    Count recvCount
)
{
    std::vector<Scalar> recvBuffer(recvCount);

    MPI_Sendrecv
    (
        sendBuffer.data(),
        static_cast<int>(sendBuffer.size()),
        MPIScalarType(),
        static_cast<int>(patch.neighborRank()),
        exchangeTag,
        recvBuffer.data(),
        static_cast<int>(recvBuffer.size()),
        MPIScalarType(),
        static_cast<int>(patch.neighborRank()),
        exchangeTag,
        MPI_COMM_WORLD,
        MPI_STATUS_IGNORE
    );

    return recvBuffer;
}

} // namespace

// ****************************** Cell Accounting *****************************

TEST_CASE("Decomposition preserves total cell count", "[mpi][parallel]")
{
    constexpr Count nx = 8;
    constexpr Count ny = 2;
    constexpr Count nz = 2;
    constexpr Count expectedTotal = nx * ny * nz;

    const DecomposedBoxMesh box(nx, ny, nz);
    const Mesh& mesh = box.mesh();

    const Count domainSum = globalSum(mesh.numDomainCells());
    REQUIRE(domainSum == expectedTotal);
    REQUIRE(mesh.numDomainCells() > 0);
}

// **************************** Ghost Cell Geometry ***************************

TEST_CASE
(
    "Ghost cell geometry bit-matches the remote owner rank",
    "[mpi][parallel]"
)
{
    if (!Comm::parallelRun())
    {
        SKIP("Decomposition cut checks require at least 2 ranks");
    }

    const DecomposedBoxMesh box(8, 2, 2);
    const Mesh& mesh = box.mesh();
    const CellList& cells = mesh.cells();

    for (const ProcessorPatch& patch : Halo::processorPatches())
    {
        std::vector<Scalar> sendBuffer;
        sendBuffer.reserve(4 * patch.sendCellIndices().size());

        for (const Index cellIdx : patch.sendCellIndices())
        {
            const Vector& centroid = cells[cellIdx].centroid();
            sendBuffer.push_back(centroid.x());
            sendBuffer.push_back(centroid.y());
            sendBuffer.push_back(centroid.z());
            sendBuffer.push_back(cells[cellIdx].volume());
        }

        const std::vector<Scalar> received =
            exchangeWithNeighbor
            (
                patch,
                sendBuffer,
                4 * patch.ghostCellCount()
            );

        for (Index i = 0; i < patch.ghostCellCount(); ++i)
        {
            const Cell& ghost = cells[patch.ghostFirstCell() + i];

            const bool match =
                received[4 * i]     == ghost.centroid().x()
             && received[4 * i + 1] == ghost.centroid().y()
             && received[4 * i + 2] == ghost.centroid().z()
             && received[4 * i + 3] == ghost.volume();

            REQUIRE(match);
        }
    }
}

// ***************************** Cut Face Geometry ****************************

TEST_CASE
(
    "Cut face geometry bit-matches across partition cuts",
    "[mpi][parallel]"
)
{
    if (!Comm::parallelRun())
    {
        SKIP("Decomposition cut checks require at least 2 ranks");
    }

    const DecomposedBoxMesh box(8, 2, 2);
    const Mesh& mesh = box.mesh();
    const FaceList& faces = mesh.faces();

    for (const ProcessorPatch& patch : Halo::processorPatches())
    {
        std::vector<Scalar> sendBuffer;
        sendBuffer.reserve(7 * patch.numFaces());

        for
        (
            Index faceIdx = patch.firstFaceIdx();
            faceIdx <= patch.lastFaceIdx();
            ++faceIdx
        )
        {
            const Face& face = faces[faceIdx];
            const Vector& centroid = face.centroid();
            const Vector& normal = face.normal();

            sendBuffer.push_back(centroid.x());
            sendBuffer.push_back(centroid.y());
            sendBuffer.push_back(centroid.z());
            sendBuffer.push_back(normal.x());
            sendBuffer.push_back(normal.y());
            sendBuffer.push_back(normal.z());
            sendBuffer.push_back(face.contactArea());
        }

        const std::vector<Scalar> received =
            exchangeWithNeighbor
            (
                patch,
                sendBuffer,
                7 * patch.numFaces()
            );

        for (Index i = 0; i < patch.numFaces(); ++i)
        {
            const Face& face = faces[patch.firstFaceIdx() + i];

            const bool match =
                received[7 * i]     == face.centroid().x()
             && received[7 * i + 1] == face.centroid().y()
             && received[7 * i + 2] == face.centroid().z()
             && received[7 * i + 3] == face.normal().x()
             && received[7 * i + 4] == face.normal().y()
             && received[7 * i + 5] == face.normal().z()
             && received[7 * i + 6] == face.contactArea();

            REQUIRE(match);
        }
    }
}

// *************************** Gradient Across Cuts ***************************

TEST_CASE
(
    "Least-squares gradient across partition cuts reproduces linear fields",
    "[mpi][parallel]"
)
{
    if (!Comm::parallelRun())
    {
        SKIP("Decomposition cut checks require at least 2 ranks");
    }

    // 8x4x4 ensures interior cells adjacent to partition cuts do not touch
    // physical boundaries (where an empty BoundaryConditions would fail)
    const DecomposedBoxMesh box(8, 4, 4);
    const Mesh& mesh = box.mesh();

    const Scalar a = S(1.5);
    const Scalar b = S(2.5);
    const Scalar c = S(-3.5);
    const Scalar d = S(4.2);

    ScalarField phi(mesh);
    const CellList& cells = mesh.cells();
    const Count numCells = mesh.numCells();

    for (Index cellIdx = 0; cellIdx < numCells; ++cellIdx)
    {
        const Vector& x = cells[cellIdx].centroid();
        phi[cellIdx] = a * x.x() + b * x.y() + c * x.z() + d;
    }

    const BoundaryConditions bc;
    const LeastSquares leastSquares(mesh, bc);

    std::vector<bool> tested(mesh.numDomainCells(), false);
    const FaceList& faces = mesh.faces();

    Scalar maxError = S(0.0);
    Count testedCells = 0;

    for (const ProcessorPatch& patch : Halo::processorPatches())
    {
        for
        (
            Index faceIdx = patch.firstFaceIdx();
            faceIdx <= patch.lastFaceIdx();
            ++faceIdx
        )
        {
            const Face& face = faces[faceIdx];
            const Index neighborIdx = face.neighborCell().value();

            const Index ownedSide =
                face.ownerCell() < mesh.numDomainCells()
              ? face.ownerCell()
              : neighborIdx;

            if (tested[ownedSide])
            {
                continue;
            }

            tested[ownedSide] = true;

            bool touchesBoundary = false;
            for (const Index cellFaceIdx : cells[ownedSide].faceIndices())
            {
                if (faces[cellFaceIdx].isBoundary())
                {
                    touchesBoundary = true;
                    break;
                }
            }

            if (touchesBoundary)
            {
                continue;
            }

            const Vector gradient =
                leastSquares.cellGradient(Field::p, phi, ownedSide);

            maxError =
                std::max
                (
                    maxError,
                    std::max
                    (
                        std::abs(gradient.x() - a),
                        std::max
                        (
                            std::abs(gradient.y() - b),
                            std::abs(gradient.z() - c)
                        )
                    )
                );

            ++testedCells;
        }
    }

    const Scalar globalError = globalMax(maxError);
    const Count globalTested = globalSum(testedCells);

    REQUIRE(globalTested > 0);
    REQUIRE(globalError <= std::sqrt(smallValue));
    REQUIRE(globalError <= TestTolerances::absOperator);
}
