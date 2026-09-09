/******************************************************************************

                                     Turblyze
                           3D incompressible CFD solver
                       Copyright (C) 2025-2026 Mohamed Mousa
                        SPDX-License-Identifier: Apache-2.0

 ------------------------------------------------------------------------------
 * @file Vector.h
 * @brief 3D vector class for geometric and mathematical operations in CFD
 *
 * @details This header defines a 3D vector class that serves as the
 * foundation for all vector-based calculations in the CFD solver. The Vector
 * class provides essential mathematical operations required in the finite
 * volume discretization and mesh operations.
 *****************************************************************************/

#pragma once

// ********************************** Headers *********************************

// Standard library headers
#include <cmath>
#include <iosfwd>

// Project headers
#include "Scalar.h"
#include "ErrorHandler.h"

// ******************************* class Vector *******************************

class Vector
{
public:

/// ************************* Special Member Functions ************************

    /// Default constructor
    Vector() noexcept = default;

    /// Construct vector with specified components
    Vector(Scalar xValue, Scalar yValue, Scalar zValue) noexcept
    :
        x_{xValue},
        y_{yValue},
        z_{zValue}
    {}

// ***************************** Accessor Methods *****************************

    /// Get X component
    [[nodiscard]] Scalar x() const noexcept { return x_; }

    /// Get Y component
    [[nodiscard]] Scalar y() const noexcept { return y_; }

    /// Get Z component
    [[nodiscard]] Scalar z() const noexcept { return z_; }

// ***************************** Operator Methods *****************************

    /// Vector addition operator
    Vector operator+(const Vector& other) const noexcept
    {
        return Vector{x_ + other.x_, y_ + other.y_, z_ + other.z_};
    }

    /// Vector subtraction operator
    Vector operator-(const Vector& other) const noexcept
    {
        return Vector{x_ - other.x_, y_ - other.y_, z_ - other.z_};
    }

    /// Scalar multiplication operator
    Vector operator*(Scalar s) const noexcept
    {
        return Vector{x_ * s, y_ * s, z_ * s};
    }

    /// Scalar division operator
    Vector operator/(Scalar s) const noexcept
    {
        return Vector{x_ / s, y_ / s, z_ / s};
    }

    /// Compound addition assignment operator
    Vector& operator+=(const Vector& other) noexcept
    {
        x_ += other.x_;
        y_ += other.y_;
        z_ += other.z_;

        return *this;
    }

    /// Compound subtraction assignment operator
    Vector& operator-=(const Vector& other) noexcept
    {
        x_ -= other.x_;
        y_ -= other.y_;
        z_ -= other.z_;

        return *this;
    }

    /// Compound multiplication assignment operator
    Vector& operator*=(Scalar s) noexcept
    {
        x_ *= s;
        y_ *= s;
        z_ *= s;

        return *this;
    }

    /// Compound division assignment operator
    Vector& operator/=(Scalar s) noexcept
    {
        if (std::abs(s) <= vSmallValue)
        {
            FatalError("Division by zero in Vector::operator/=");
        }

        const Scalar inverse = S(1.0) / s;
        x_ *= inverse;
        y_ *= inverse;
        z_ *= inverse;

        return *this;
    }

    /// Equality comparison operator
    bool operator==(const Vector& other) const noexcept
    {
        return (std::abs(x_ - other.x_) <= vSmallValue)
            && (std::abs(y_ - other.y_) <= vSmallValue)
            && (std::abs(z_ - other.z_) <= vSmallValue);
    }

// ****************************** Private Members *****************************

private:

    /// x, y, z components of the vector
    Scalar x_ = S(0.0);
    Scalar y_ = S(0.0);
    Scalar z_ = S(0.0);
};

// *************************** Non-Member Functions ***************************

/// Compute dot product of two vectors
[[nodiscard]] inline Scalar dot(const Vector& p1, const Vector& p2) noexcept
{
    return p1.x() * p2.x() + p1.y() * p2.y() + p1.z() * p2.z();
}

/// Compute cross product of two vectors
[[nodiscard]] inline Vector cross
(
    const Vector& p1,
    const Vector& p2
) noexcept
{
    return
        Vector
        (
            p1.y() * p2.z() - p1.z() * p2.y(),
            p1.z() * p2.x() - p1.x() * p2.z(),
            p1.x() * p2.y() - p1.y() * p2.x()
        );
}

/// Squared magnitude of a vector
[[nodiscard]] inline Scalar magnitudeSquared(const Vector& v) noexcept
{
    return v.x() * v.x() + v.y() * v.y() + v.z() * v.z();
}

/// Magnitude of a vector
[[nodiscard]] inline Scalar magnitude(const Vector& v) noexcept
{
    return std::sqrt(magnitudeSquared(v));
}

/// Return a normalized copy of a vector
[[nodiscard]] inline Vector normalized(const Vector& v) noexcept
{
    const Scalar mag = magnitude(v);

    if (mag < vSmallValue)
    {
        FatalError("Division by zero in normalized(Vector)");
    }

    const Scalar inverse = S(1.0) / mag;
    return Vector(v.x() * inverse, v.y() * inverse, v.z() * inverse);
}

/// Scalar multiplication operator
inline Vector operator*(Scalar s, const Vector& p) noexcept
{
    return p * s;
}

/// Stream output operator for Vector
std::ostream& operator<<(std::ostream& os, const Vector& p);
