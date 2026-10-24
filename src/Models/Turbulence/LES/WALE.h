/******************************************************************************

                                     Turblyze
                           3D incompressible CFD solver
                       Copyright (C) 2025-2026 Mohamed Mousa
                        SPDX-License-Identifier: Apache-2.0

 ------------------------------------------------------------------------------
 * @file WALE.h
 * @brief Wall-Adapting Local Eddy-viscosity (WALE) sub-grid scale model
 *
 * @details The WALE model implements the algebraic SGS model of 
 * Nicoud & Ducros (1999) based on the traceless symmetric part of
 * the squared velocity gradient tensor Sd:
 * nut = (Cw * delta)^2 * (Sd : Sd)^(3/2) / ((S : S)^(5/2) + (Sd : Sd)^(5/4))
 * k_sgs = (nut / (Ck * delta))^2
 *****************************************************************************/

#pragma once

// ********************************** Headers *********************************

#include "LES.h"

// ******************************** class WALE ********************************

class WALE final : public LES
{
public:

// ****************************** Model Constants *****************************

    /// Structure containing WALE model constants
    struct ModelConstants
    {
        Scalar Cw;  ///< WALE model constant
        Scalar Ck;  ///< SGS kinetic energy constant
    };

    static constexpr ModelConstants coeffs_
    {
        .Cw = S(0.325),
        .Ck = S(0.094)
    };

// ************************* Special Member Functions *************************

    /// Constructor
    WALE
    (
        const Mesh& mesh,
        const BoundaryConditions& bc,
        Scalar nu,
        Scalar Cw = coeffs_.Cw,
        Scalar Ck = coeffs_.Ck,
        bool debug = false
    );

    /// Copy constructor and assignment - Not copyable (non-copyable base)
    WALE(const WALE&) = delete;
    WALE& operator=(const WALE&) = delete;

    /// Move constructor and assignment - Not movable (non-copyable base)
    WALE(WALE&&) = delete;
    WALE& operator=(WALE&&) = delete;

    /// Destructor
    ~WALE() noexcept override = default;

// ***************************** Accessor Methods *****************************

    /// WALE model constant Cw
    [[nodiscard]] Scalar Cw() const noexcept
    {
        return Cw_;
    }

    /// SGS kinetic energy constant Ck
    [[nodiscard]] Scalar Ck() const noexcept
    {
        return Ck_;
    }

// ***************************** Protected Methods ****************************

protected:

    /// Update sub-grid scale viscosity nut_ from the WALE operator
    void updateSgsViscosity(const TensorField& gradU) override;

    /// Update sub-grid scale kinetic energy kSgs_ = (nut / (Ck * delta))^2
    void updateSgsKineticEnergy(const TensorField& gradU) override;

    /// Squared magnitude Sd : Sd of the traceless symmetric part of gradU^2
    [[nodiscard]] static Scalar
    SdMagnitudeSquared(const Tensor& gradU) noexcept;

// ****************************** Private Members *****************************

private:

    /// WALE constant (default: 0.325)
    Scalar Cw_;

    /// SGS kinetic energy constant (default: 0.094)
    Scalar Ck_;
};
