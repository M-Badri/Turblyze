/******************************************************************************

                                     Turblyze
                           3D incompressible CFD solver
                       Copyright (C) 2025-2026 Mohamed Mousa
                        SPDX-License-Identifier: Apache-2.0

 ------------------------------------------------------------------------------
 * @file TaylorGreen.h
 * @brief Taylor-Green vortex field initialization
 *
 * @details Implements the 3D Taylor-Green vortex analytical velocity and
 * pressure distribution evaluated at cell centroids:
 * u =  U0 * sin(x/L) * cos(y/L) * cos(z/L)
 * v = -U0 * cos(x/L) * sin(y/L) * cos(z/L)
 * w =  0
 * p =  p0 + (rho * U0^2 / 16) * (cos(2x/L) + cos(2y/L)) * (cos(2z/L) + 2)
 *****************************************************************************/

#pragma once

// ********************************** Headers *********************************

#include "Initializer.h"
#include "Scalar.h"

// ************************* namespace Initialization *************************

namespace Initialization
{

// **************************** class TaylorGreen *****************************

class TaylorGreen final : public Initializer
{
public:

// ************************* Special Member Functions *************************

    /// Constructor
    TaylorGreen
    (
        Scalar U0,
        Scalar L,
        Scalar p0,
        Scalar rho
    ) noexcept
    : 
        U0_{U0},
        L_{L},
        p0_{p0},
        rho_{rho}
    {}

// *************************** Initialization Method **************************

    /// Apply the initial condition to velocity and pressure fields
    void initialize
    (
        const Mesh& mesh,
        BoundaryConditions& bc,
        ScalarField& Ux,
        ScalarField& Uy,
        ScalarField& Uz,
        ScalarField& p
    ) const override;

// ****************************** Private Members *****************************

private:

    /// Velocity scale
    Scalar U0_;

    /// Length scale
    Scalar L_;

    /// Reference pressure
    Scalar p0_;

    /// Fluid density
    Scalar rho_;
};

} // namespace Initialization
