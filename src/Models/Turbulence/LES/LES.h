/******************************************************************************

                                     Turblyze
                           3D incompressible CFD solver
                       Copyright (C) 2025-2026 Mohamed Mousa
                        SPDX-License-Identifier: Apache-2.0

 ------------------------------------------------------------------------------
 * @file LES.h
 * @brief Base class for Large Eddy Simulation (LES) sub-grid scale models
 *
 * @details The LES base class provides the foundation for scale-resolving
 * sub-grid scale (SGS) models. It manages the sub-grid eddy viscosity (nut),
 * sub-grid kinetic energy (k_sgs), cell filter width (delta = V^(1/3)),
 * wall distance calculation, and MPI halo exchange.
 *****************************************************************************/

#pragma once

// ********************************** Headers *********************************

// Standard library headers
#include <algorithm>
#include <cmath>
#include <memory>

// Project headers
#include "TurbulenceModel.h"
#include "Scalar.h"
#include "Vector.h"
#include "Tensor.h"
#include "CellData.h"
#include "FaceData.h"
#include "Mesh.h"
#include "BoundaryConditions.h"

// ********************************* class LES ********************************

class LES : public TurbulenceModel
{
public:

// ************************* Special Member Functions *************************

    /// Constructor
    LES
    (
        const Mesh& m,
        const BoundaryConditions& bc,
        Scalar nu,
        bool debug = false
    );

    /// Copy constructor and assignment - Not copyable (non-copyable base)
    LES(const LES&) = delete;
    LES& operator=(const LES&) = delete;

    /// Move constructor and assignment - Not movable (non-copyable base)
    LES(LES&&) = delete;
    LES& operator=(LES&&) = delete;

    /// Destructor
    ~LES() noexcept override = default;

// ***************************** Runtime Selection ****************************

    /// Construct the LES model selected by name
    [[nodiscard]] static std::unique_ptr<LES> create
    (
        const Name& modelName,
        const Mesh& mesh,
        const BoundaryConditions& bc,
        Scalar nu,
        Scalar Cs,
        Scalar Ck,
        Scalar Cw,
        bool vanDriest,
        Scalar Aplus,
        Scalar kappa,
        bool debug
    );

    /// Names of available LES models
    [[nodiscard]] static NameList availableModels();

// ***************************** Turbulence Solve *****************************

    /// Solve LES
    void solve
    (
        const ScalarField& Ux,
        const ScalarField& Uy,
        const ScalarField& Uz,
        const FaceFluxField& flowRateFace,
        const TensorField& gradU
    ) override;

// ************************** Shared Accessor Methods *************************

    /// An LES model is turbulent flow
    [[nodiscard]] bool isTurbulent() const noexcept override
    {
        return true;
    }

    /// Turbulent sub-grid kinematic viscosity
    [[nodiscard]] const ScalarField&
    turbulentViscosity() const noexcept override
    {
        return nut_;
    }

    /// Sub-grid scale kinetic energy
    [[nodiscard]] const ScalarField& kSgs() const noexcept
    {
        return kSgs_;
    }

    /// Cell filter width
    [[nodiscard]] const ScalarField& filterWidth() const noexcept
    {
        return delta_;
    }

    /// Distance to nearest wall
    [[nodiscard]] const ScalarField& wallDistance() const noexcept
    {
        return wallDistance_;
    }

    /// Whether wall-distance initialization converged
    [[nodiscard]] bool wallDistanceConverged() const noexcept override
    {
        return wallDistanceConverged_;
    }

    /// Wall shear stress magnitude on wall faces
    [[nodiscard]] FaceData<Scalar> wallShearStress
    (
        const ScalarField& Ux,
        const ScalarField& Uy,
        const ScalarField& Uz
    ) const override;

    /// Cell-centered scalar fields exported by LES models
    [[nodiscard]] CellDataPair cellDataOutputs() const override
    {
        return
        {
            {"nut", &nut_},
            {"k_sgs", &kSgs_},
            {"delta", &delta_}
        };
    }


// ***************************** Protected Methods ****************************

protected:

    /// Update model-specific sub-grid scale viscosity nut_
    virtual void updateSgsViscosity(const TensorField& gradU) = 0;

    /// Update model-specific sub-grid scale kinetic energy kSgs_
    virtual void updateSgsKineticEnergy(const TensorField& gradU) = 0;

    /// Evaluate strain rate magnitude |S| = sqrt(2 * S : S)
    [[nodiscard]] static Scalar
    strainRateMagnitude(const Tensor& gradU) noexcept
    {
        return std::sqrt(S(2.0) * gradU.symm().magnitudeSquared());
    }

// ***************************** Protected Members ****************************

protected:

    /// Mesh reference
    const Mesh& mesh_;

    /// Boundary conditions reference
    const BoundaryConditions& bc_;

    /// Laminar kinematic viscosity
    Scalar nu_;

    /// Sub-grid scale eddy viscosity
    ScalarField nut_{mesh_, S(0.0)};

    /// Sub-grid scale turbulent kinetic energy
    ScalarField kSgs_{mesh_, S(0.0)};

    /// Filter width delta = V^(1/3)
    ScalarField delta_{mesh_, S(0.0)};

    /// Wall distance for near-wall damping
    ScalarField wallDistance_{mesh_, S(0.0)};

    /// meshWave wall-distance loop convergence flag
    bool wallDistanceConverged_ = false;

    /// Debug output flag
    bool debug_;

// ****************************** Private Methods *****************************

private:

    /// Compute cell filter width delta = V^(1/3)
    void computeFilterWidth();
};
