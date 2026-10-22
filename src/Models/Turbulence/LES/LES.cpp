/******************************************************************************

                                     Turblyze
                           3D incompressible CFD solver
                       Copyright (C) 2025-2026 Mohamed Mousa
                        SPDX-License-Identifier: Apache-2.0

 ------------------------------------------------------------------------------
 * @file LES.cpp
 * @brief Base class implementation for LES sub-grid scale models
 *****************************************************************************/

// ********************************** Headers *********************************

// Implementation header
#include "LES.h"

// Standard library headers
#include <algorithm>
#include <cmath>
#include <memory>

// Project headers
#include "BoundaryPatch.h"
#include "HaloExchange.h"
#include "RuntimeSelection.h"
#include "Smagorinsky.h"

// ************************* Special Member Functions *************************

LES::LES
(
    const Mesh& m,
    const BoundaryConditions& bc,
    Scalar nu,
    bool debug
)
:
    mesh_{m},
    bc_{bc},
    nu_{nu},
    debug_{debug}
{
    computeFilterWidth();
    wallDistanceConverged_ =
        computeWallDistance(mesh_, bc_, wallDistance_);
}

// ***************************** Runtime Selection ****************************

std::unique_ptr<LES> LES::create
(
    const Name& modelName,
    const Mesh& mesh,
    const BoundaryConditions& bc,
    Scalar nu,
    Scalar Cs,
    Scalar Ck,
    bool vanDriest,
    Scalar Aplus,
    Scalar kappa,
    bool debug
)
{
    if (modelName == "Smagorinsky")
    {
        return std::make_unique<Smagorinsky>
        (
            mesh,
            bc,
            nu,
            Cs,
            Ck,
            vanDriest,
            Aplus,
            kappa,
            debug
        );
    }

    RuntimeSelection::unknownSelection
    (
        "LES model",
        modelName,
        availableModels()
    );
}


NameList LES::availableModels()
{
    return {"Smagorinsky"};
}

// ***************************** Turbulence Solve *****************************

void LES::solve
(
    const ScalarField& /*Ux*/,
    const ScalarField& /*Uy*/,
    const ScalarField& /*Uz*/,
    const FaceFluxField& /*flowRateFace*/,
    const TensorField& gradU
)
{
    updateSgsViscosity(gradU);
    updateSgsKineticEnergy(gradU);

    // halo exchange across partition cuts
    Halo::exchange({&nut_, &kSgs_});
}

// ************************** Shared Accessor Methods *************************

FaceData<Scalar> LES::wallShearStress
(
    const ScalarField& Ux,
    const ScalarField& Uy,
    const ScalarField& Uz
) const
{
    FaceData<Scalar> shearStress(mesh_, S(0.0));

    for (const auto& patch : mesh_.patches())
    {
        if (patch.type() == PatchType::processor)
        {
            continue;
        }

        const auto* bc = bc_.find(patch.name(), Field::Ux);
        if (bc == nullptr || !bc->isWall())
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
            const auto& face = mesh_.faces()[faceIdx];
            const Index cellIdx = face.ownerCell();
            const Vector& normal = face.normal();
            const Vector cellVelocity(Ux[cellIdx], Uy[cellIdx], Uz[cellIdx]);
            const Scalar normalVelocity = dot(cellVelocity, normal);
            const Vector tangentVelocity =
                cellVelocity - normalVelocity * normal;
            const Scalar wallDist =
                std::max
                (
                    std::abs(dot(mesh_.dPf(face), normal)),
                    vSmallValue
                );

            const Scalar nuEff = nu_ + nut_[cellIdx];
            shearStress[face.idx()] =
                nuEff * magnitude(tangentVelocity) / wallDist;
        }
    }

    return shearStress;
}

// ****************************** Private Methods *****************************

void LES::computeFilterWidth()
{
    const Count numOwned = mesh_.numDomainCells();
    const auto& cells = mesh_.cells();

    for (Index cellIdx = 0; cellIdx < numOwned; ++cellIdx)
    {
        const Scalar vol = cells[cellIdx].volume();
        delta_[cellIdx] = std::cbrt(std::max(vol, S(0.0)));
    }

    Halo::exchange({&delta_});
}
