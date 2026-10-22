/******************************************************************************

                                     Turblyze
                           3D incompressible CFD solver
                       Copyright (C) 2025-2026 Mohamed Mousa
                        SPDX-License-Identifier: Apache-2.0

 ------------------------------------------------------------------------------
 * @file TurbulenceModel.cpp
 * @brief Runtime selection and shared helpers for turbulence models
 *****************************************************************************/

// ********************************** Headers *********************************

// Implementation header
#include "TurbulenceModel.h"

// Standard library headers
#include <algorithm>
#include <cmath>

// Project headers
#include "BoundaryConditions.h"
#include "BoundaryPatch.h"
#include "ConvectionScheme.h"
#include "GradientScheme.h"
#include "HaloExchange.h"
#include "LinearSolvers.h"
#include "Mesh.h"
#include "Reduce.h"
#include "RuntimeSelection.h"
#include "kOmegaSST.h"

// **************************** Runtime Selection ****************************

std::unique_ptr<TurbulenceModel> TurbulenceModel::create
(
    const Name& modelName,
    const Mesh& mesh,
    const BoundaryConditions& bc,
    const TimeScheme& timeScheme,
    const GradientScheme& gradScheme,
    const ConvectionScheme& kScheme,
    LinearSolver& kSolver,
    const ConvectionScheme& omegaScheme,
    LinearSolver& omegaSolver,
    Scalar deltaT,
    Scalar nu,
    Scalar initialK,
    Scalar initialOmega,
    Scalar alphaK,
    Scalar alphaOmega,
    bool roughWall,
    bool debug
)
{
    if (modelName == "kOmegaSST")
    {
        return std::make_unique<kOmegaSST>
        (
            mesh,
            bc,
            timeScheme,
            gradScheme,
            kScheme,
            kSolver,
            omegaScheme,
            omegaSolver,
            deltaT,
            nu,
            initialK,
            initialOmega,
            alphaK,
            alphaOmega,
            roughWall,
            debug
        );
    }

    RuntimeSelection::unknownSelection
    (
        "turbulence model",
        Name(modelName),
        availableModels()
    );
}


NameList TurbulenceModel::availableModels()
{
    return {"Laminar", "kOmegaSST", "Smagorinsky"};
}

// ***************************** Protected Methods ****************************

bool TurbulenceModel::computeWallDistance
(
    const Mesh& mesh,
    const BoundaryConditions& bc,
    ScalarField& wallDistance
)
{
    bool converged = false;
    wallDistance.setAll(S(1e10));
    VectorField nearestWallPoint{mesh, Vector{S(1e15), S(1e15), S(1e15)}};

    // Seed wall-adjacent cells with the distance to each wall centroid
    for (const auto& patch : mesh.patches())
    {
        if (patch.type() == PatchType::processor)
        {
            continue;
        }

        const auto* patchBc = bc.find(patch.name(), Field::Ux);
        if (patchBc == nullptr || !patchBc->isWall())
        {
            continue;
        }

        for
        (
            Index faceIdx = patch.firstFaceIdx();
            faceIdx <= patch.lastFaceIdx();
            ++faceIdx
        )
        {
            const auto& face = mesh.faces()[faceIdx];
            const Index cellIdx = face.ownerCell();
            const Vector cellCenter = mesh.cells()[cellIdx].centroid();
            const Vector faceCenter = face.centroid();
            const Vector normal = face.normal();
            const Vector cellToFace = faceCenter - cellCenter;
            const Scalar dist = std::abs(dot(cellToFace, normal));

            if (dist < wallDistance[cellIdx])
            {
                wallDistance[cellIdx] = dist;
                nearestWallPoint[cellIdx] = faceCenter;
            }
        }
    }

    // Iterative propagation
    const Count maxIterations = 100;
    const Scalar tolerance = smallValue;

    for (Index iter = 0; iter < maxIterations; ++iter)
    {
        Halo::exchange({&wallDistance});
        Halo::exchange({&nearestWallPoint});

        Scalar maxChange = S(0.0);

        for (const auto& face : mesh.faces())
        {
            if (face.isBoundary())
            {
                continue;
            }

            const Index owner = face.ownerCell();
            const Index neighbor = face.neighborCell().value();
            const Vector ownerCenter = mesh.cells()[owner].centroid();
            const Vector neighborCenter = mesh.cells()[neighbor].centroid();

            // Try to improve owner using neighbor's nearest wall point
            {
                const Vector candidatePoint = nearestWallPoint[neighbor];
                const Scalar newDist =
                    magnitude(ownerCenter - candidatePoint);

                if (newDist < wallDistance[owner])
                {
                    const Scalar change = wallDistance[owner] - newDist;
                    maxChange = std::max(maxChange, change);
                    wallDistance[owner] = newDist;
                    nearestWallPoint[owner] = candidatePoint;
                }
            }

            // Try to improve neighbor using owner's nearest wall point
            {
                const Vector candidatePoint = nearestWallPoint[owner];
                const Scalar newDist =
                    magnitude(neighborCenter - candidatePoint);

                if (newDist < wallDistance[neighbor])
                {
                    const Scalar change = wallDistance[neighbor] - newDist;
                    maxChange = std::max(maxChange, change);
                    wallDistance[neighbor] = newDist;
                    nearestWallPoint[neighbor] = candidatePoint;
                }
            }
        }

        // Parallel invariant: collective must be reached by every rank
        if (globalMax(maxChange) < tolerance)
        {
            converged = true;
            break;
        }
    }

    Halo::exchange({&wallDistance});

    return converged;
}
