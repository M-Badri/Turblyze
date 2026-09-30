/******************************************************************************

                                     Turblyze
                           3D incompressible CFD solver
                       Copyright (C) 2025-2026 Mohamed Mousa
                        SPDX-License-Identifier: Apache-2.0

 ------------------------------------------------------------------------------
 * @file MeshCreator.cpp
 * @brief Mesh read and geometry-preparation phase, serial or distributed
 *****************************************************************************/

// ********************************** Headers *********************************

// Implementation header
#include "MeshCreator.h"

// Standard library headers
#include <format>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

// Project headers
#include "Logger.h"
#include "Comm.h"
#include "ErrorHandler.h"
#include "MeshChecker.h"
#include "MeshDecomposer.h"
#include "MeshDistributor.h"
#include "MeshReader.h"
#include "Reduce.h"
#include "SubmeshData.h"
#include "HaloExchange.h"

// ****************************** Internal Helpers ****************************

namespace
{

/// Compute face and domain-cell geometry; ghost stubs already carry theirs
void prepareGeometry
(
    FaceList& faces,
    CellList& cells,
    const NodeList& nodes,
    Count numDomainCells,
    bool debug
)
{
    std::vector<FaceIntegrals> faceIntegrals(faces.size());

    for (Index faceIdx = 0; faceIdx < faces.size(); ++faceIdx)
    {
        faceIntegrals[faceIdx] =
            faces[faceIdx].geometricProperties(nodes);
    }
    if (debug)
    {
        std::cout
            << "Geometric properties calculated for faces." << '\n';
    }

    for (Index cellIdx = 0; cellIdx < numDomainCells; ++cellIdx)
    {
        cells[cellIdx].geometricProperties(faceIntegrals);
    }
    if (debug)
    {
        std::cout
            << "Geometric properties calculated for cells." << '\n';
    }
}


/// Read the complete mesh from file, prepare it, optionally check it
Mesh parseMesh(const CaseConfiguration& config)
{
    Halo::reset();

    MeshReader meshReader(config.meshFile);

    auto nodes = meshReader.moveNodes();
    auto faces = meshReader.moveFaces();
    auto cells = meshReader.moveCells();
    auto patches = meshReader.moveBoundaryPatches();

    std::cout << std::format
    (
        "Mesh Loaded: {} nodes, {} faces, {} cells.\n",
        nodes.size(), faces.size(), cells.size()
    );

    MeshCreator::linkBoundaryFaces(faces, patches);
    prepareGeometry(faces, cells, nodes, cells.size(), config.debug);

    Mesh mesh
    (
        std::move(nodes),
        std::move(faces),
        std::move(cells),
        std::move(patches)
    );

    if (config.checkQuality)
    {
        MeshChecker::check(mesh);
    }

    return mesh;
}


/// Rebuild one rank's Mesh from its flat submesh block
[[nodiscard]] Mesh buildSubmesh(SubmeshData block, bool debug)
{
    const Count numFaces = block.faceOwner.size();
    const Count numGhosts = block.ghostVolumes.size();

    // Nodes
    NodeList nodes;
    nodes.reserve(block.nodeCoords.size() / 3);

    for (Index i = 0; i < block.nodeCoords.size(); i += 3)
    {
        nodes.emplace_back
        (
            block.nodeCoords[i],
            block.nodeCoords[i + 1],
            block.nodeCoords[i + 2]
        );
    }

    // Faces: node lists from CSR, owner/neighbor from flat arrays
    FaceList faces;
    faces.reserve(numFaces);

    for (Index f = 0; f < numFaces; ++f)
    {
        IndexList faceNodeIndices
        (
            block.faceNodes.data() + block.faceNodeOffsets[f],
            block.faceNodes.data() + block.faceNodeOffsets[f + 1]
        );

        if (block.faceNeighbor[f] != SubmeshData::noNeighbor)
        {
            faces.emplace_back
            (
                f,
                std::move(faceNodeIndices),
                block.faceOwner[f],
                block.faceNeighbor[f]
            );
        }
        else
        {
            faces.emplace_back
            (
                f,
                std::move(faceNodeIndices),
                block.faceOwner[f]
            );
        }
    }

    // Cells: owned cells first (with face lists and signs from CSR)
    CellList cells;
    cells.reserve(block.numOwnedCells + numGhosts);

    for (Index c = 0; c < block.numOwnedCells; ++c)
    {
        IndexList cellFaceIndices
        (
            block.cellFaces.data() + block.cellFaceOffsets[c],
            block.cellFaces.data() + block.cellFaceOffsets[c + 1]
        );

        // Precompute neighbor-cell indices from face ownership
        IndexList cellNeighborIndices;
        cellNeighborIndices.reserve(cellFaceIndices.size());

        for (const Index faceIdx : cellFaceIndices)
        {
            const Face& face = faces[faceIdx];

            if (face.ownerCell() == c)
            {
                if (face.neighborCell().has_value())
                {
                    cellNeighborIndices.push_back(face.neighborCell().value());
                }
            }
            else
            {
                cellNeighborIndices.push_back(face.ownerCell());
            }
        }

        std::vector<int8_t> signs
        (
            block.cellFaceSigns.data() + block.cellFaceOffsets[c],
            block.cellFaceSigns.data() + block.cellFaceOffsets[c + 1]
        );

        Cell cell
        (
            c,
            std::move(cellFaceIndices),
            std::move(cellNeighborIndices),
            std::move(signs)
        );

        cells.push_back(std::move(cell));
    }

    // Ghost cells: append stubs carrying their complete-mesh geometry
    for (Index g = 0; g < numGhosts; ++g)
    {
        Cell ghost;
        ghost.setIdx(block.numOwnedCells + g);
        ghost.setGeometry
        (
            Vector
            (
                block.ghostCentroids[3 * g],
                block.ghostCentroids[3 * g + 1],
                block.ghostCentroids[3 * g + 2]
            ),
            block.ghostVolumes[g]
        );

        cells.push_back(std::move(ghost));
    }

    // Physical patches
    PatchList patches;
    patches.reserve(block.patchNames.size() + block.procNeighborRanks.size());

    for (Index p = 0; p < block.patchNames.size(); ++p)
    {
        BoundaryPatch patch
        (
            block.patchZoneIds[p],
            block.patchFirstFace[p],
            block.patchLastFace[p]
        );
        patch.setName(block.patchNames[p]);
        patch.setType(static_cast<PatchType>(block.patchTypes[p]));
        patches.push_back(std::move(patch));
    }

    // Processor patches: boundary patches and cut metadata
    ProcessorPatchList processorPatches;
    processorPatches.reserve(block.procNeighborRanks.size());

    for (Index p = 0; p < block.procNeighborRanks.size(); ++p)
    {
        BoundaryPatch patch
        (
            0,
            block.procFirstFace[p],
            block.procLastFace[p]
        );
        patch.setName
        (
            std::format
            (
                "processor{}to{}",
                Comm::myProcessorNum(),
                block.procNeighborRanks[p]
            )
        );
        patch.setType(PatchType::processor);
        patches.push_back(std::move(patch));

        processorPatches.emplace_back
        (
            block.procNeighborRanks[p],
            block.procFirstFace[p],
            block.procLastFace[p],
            IndexList
            (
                block.procSendCells.data() + block.procSendOffsets[p],
                block.procSendCells.data() + block.procSendOffsets[p + 1]
            ),
            block.numOwnedCells + block.procGhostOffsets[p],
            block.procGhostOffsets[p + 1] - block.procGhostOffsets[p]
        );
    }

    MeshCreator::linkBoundaryFaces(faces, patches);
    prepareGeometry(faces, cells, nodes, block.numOwnedCells, debug);

    Mesh mesh
    (
        std::move(nodes),
        std::move(faces),
        std::move(cells),
        std::move(patches),
        numGhosts
    );

    Halo::init
    (
        std::move(processorPatches),
        std::move(block.ghostGlobalIds)
    );

    return mesh;
}

} // namespace

// *************************** namespace MeshCreator **************************

namespace MeshCreator
{

void linkBoundaryFaces(FaceList& faces, const PatchList& patches)
{
    for (const auto& patch : patches)
    {
        for
        (
            Index faceIdx = patch.firstFaceIdx();
            faceIdx <= patch.lastFaceIdx();
            ++faceIdx
        )
        {
            faces[faceIdx].setPatch(patch);
        }
    }
}


Mesh decomposeAndDistribute(Mesh completeMesh, bool debug)
{
    std::vector<SubmeshData> blocks;

    if (Comm::master())
    {
        {
            const MeshDecomposer decomposer
            (
                completeMesh,
                Comm::numProcessors()
            );

            blocks = decomposer.decompose();
        }

        // The complete mesh is released; the local submesh takes its place
        completeMesh = Mesh();
    }

    SubmeshData block = MeshDistributor::distribute(std::move(blocks));

    const Count totalCellCount = block.totalCellCount;

    Mesh mesh = buildSubmesh(std::move(block), debug);

    if (globalSum(mesh.numDomainCells()) != totalCellCount)
    {
        FatalError("Cell count mismatch after decomposition");
    }

    return mesh;
}


Mesh create(const CaseConfiguration& config)
{
    std::cout << '\n';
    Logger::sectionHeader("Reading and Preparing Mesh");

    if (!Comm::parallelRun())
    {
        return parseMesh(config);
    }

    // The master rank reads and partitions, then ships the submeshes
    Mesh completeMesh = Comm::master() ? parseMesh(config) : Mesh();

    if (Comm::master())
    {
        std::cout << std::format
        (
            "Decomposing into {} submeshes (METIS).\n",
            Comm::numProcessors()
        );
    }

    return decomposeAndDistribute(std::move(completeMesh), config.debug);
}

} // namespace MeshCreator
