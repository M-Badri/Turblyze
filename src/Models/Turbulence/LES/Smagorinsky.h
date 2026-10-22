/******************************************************************************

                                     Turblyze
                           3D incompressible CFD solver
                       Copyright (C) 2025-2026 Mohamed Mousa
                        SPDX-License-Identifier: Apache-2.0

 ------------------------------------------------------------------------------
 * @file Smagorinsky.h
 * @brief Classical Smagorinsky sub-grid scale turbulence model with wall damping
 *
 * @details This header defines the Smagorinsky class, which implements the
 * classical algebraic eddy-viscosity SGS model:
 * nut = (Cs * delta_eff)^2 * |S|
 * k_sgs = Ck * delta_eff^2 * |S|^2
 * where |S| = sqrt(2 * S_ij * S_ij) is the resolved strain rate magnitude and
 * delta_eff includes optional Van Driest near-wall damping:
 * delta_eff = min(delta, (kappa / Cs) * y * (1 - exp(-y^+ / A+)))
 *****************************************************************************/

#pragma once

// ********************************** Headers *********************************

#include "LES.h"

// ***************************** class Smagorinsky ****************************

class Smagorinsky final : public LES
{
public:

// ****************************** Model Constants *****************************

    /// Structure containing Smagorinsky model constants
    struct ModelConstants
    {
        Scalar Cs;     ///< Smagorinsky constant
        Scalar Ck;     ///< Yoshizawa SGS kinetic energy constant
        Scalar Aplus;  ///< Van Driest damping constant A+
        Scalar kappa;  ///< Von Karman constant
    };

    static constexpr ModelConstants coeffs_
    {
        .Cs    = S(0.165),
        .Ck    = S(0.094),
        .Aplus = S(26.0),
        .kappa = S(0.41)
    };

// ************************* Special Member Functions *************************

    /// Constructor
    Smagorinsky
    (
        const Mesh& mesh,
        const BoundaryConditions& bc,
        Scalar nu,
        Scalar Cs = coeffs_.Cs,
        Scalar Ck = coeffs_.Ck,
        bool vanDriest = true,
        Scalar Aplus = coeffs_.Aplus,
        Scalar kappa = coeffs_.kappa,
        bool debug = false
    );

    /// Copy constructor and assignment - Not copyable (non-copyable base)
    Smagorinsky(const Smagorinsky&) = delete;
    Smagorinsky& operator=(const Smagorinsky&) = delete;

    /// Move constructor and assignment - Not movable (non-copyable base)
    Smagorinsky(Smagorinsky&&) = delete;
    Smagorinsky& operator=(Smagorinsky&&) = delete;

    /// Destructor
    ~Smagorinsky() noexcept override = default;

// ***************************** Accessor Methods *****************************

    /// Smagorinsky constant Cs
    [[nodiscard]] Scalar Cs() const noexcept
    {
        return Cs_;
    }

    /// Yoshizawa SGS kinetic energy constant Ck
    [[nodiscard]] Scalar Ck() const noexcept
    {
        return Ck_;
    }

    /// Whether Van Driest wall damping is enabled
    [[nodiscard]] bool vanDriest() const noexcept
    {
        return vanDriest_;
    }

// ***************************** Protected Methods ****************************

protected:

    /// Update sub-grid scale viscosity nut_ = (Cs * delta_eff)^2 * |S|
    void updateSgsViscosity(const TensorField& gradU) override;

    /// Update sub-grid scale kinetic energy kSgs_ = Ck * delta_eff^2 * |S|^2
    void updateSgsKineticEnergy(const TensorField& gradU) override;

// ****************************** Private Members *****************************

private:

    /// Smagorinsky constant (default: 0.165)
    Scalar Cs_;

    /// Yoshizawa SGS kinetic energy constant (default: 0.094)
    Scalar Ck_;

    /// Enable Van Driest near-wall damping
    bool vanDriest_;

    /// Van Driest damping constant A+ (default: 26.0)
    Scalar Aplus_;

    /// Von Karman constant kappa (default: 0.41)
    Scalar kappa_;

    /// Cached effective filter width deltaEff with wall damping applied
    ScalarField deltaEff_{mesh_, S(0.0)};
};
