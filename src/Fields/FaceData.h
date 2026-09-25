/******************************************************************************

                                     Turblyze
                           3D incompressible CFD solver
                       Copyright (C) 2025-2026 Mohamed Mousa
                        SPDX-License-Identifier: Apache-2.0

 ------------------------------------------------------------------------------
 * @file FaceData.h
 * @brief Template container for face-centered field data storage
 *****************************************************************************/

#pragma once

// ********************************** Headers *********************************

// Standard library headers
#include <algorithm>
#include <vector>
#include <concepts>

// Project headers
#include "Scalar.h"
#include "Vector.h"
#include "Mesh.h"
#include "Integer.h"

// *************************** concept FaceFieldType **************************

template<typename T>
concept FaceFieldType = std::same_as<T, Scalar> || std::same_as<T, Vector>;

// ****************************** class FaceData ******************************

template<FaceFieldType T>
class FaceData
{
public:

// ************************* Special Member Functions *************************

    /// Construct zero-initialized field
    FaceData()
    :
        internalField_(Mesh::faceCount(), T{})
    {}

    /// Construct field with initial value
    explicit FaceData(const T& initialValue)
    :
        internalField_(Mesh::faceCount(), initialValue)
    {}

// ****************************** Setter Methods ******************************

    /// Set all field values to a given value
    void setAll(const T& value)
    {
        std::fill(internalField_.begin(), internalField_.end(), value);
    }

// ***************************** Accessor Methods *****************************

    /// Get number of faces in the field
    [[nodiscard]] Count size() const noexcept
    {
        return internalField_.size();
    }

// ***************************** Operator Methods *****************************

    /// Unchecked subscript operator
    T& operator[](Index faceIndex) noexcept
    {
        return internalField_[faceIndex];
    }

    /// Unchecked const subscript operator
    const T& operator[](Index faceIndex) const noexcept
    {
        return internalField_[faceIndex];
    }

// ****************************** Private Members *****************************

private:

    /// Face-centered field values
    std::vector<T> internalField_;
};

// ********************************** Aliases *********************************

/// Type alias for scalar face fields (e.g., mass flux)
using FaceFluxField = FaceData<Scalar>;
