/******************************************************************************

                                     Turblyze
                           3D incompressible CFD solver
                       Copyright (C) 2025-2026 Mohamed Mousa
                        SPDX-License-Identifier: Apache-2.0

 ------------------------------------------------------------------------------
 * @file CellData.h
 * @brief Template container for cell-centered field data storage
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
#include "Tensor.h"
#include "Mesh.h"
#include "Integer.h"

// ************************* concept CellFieldType ****************************

template<typename T>
concept CellFieldType =
    std::same_as<T, Scalar>
 || std::same_as<T, Vector>
 || std::same_as<T, Tensor>;

// ****************************** class CellData ******************************

template<CellFieldType T>
class CellData
{
public:

// ************************* Special Member Functions *************************

    /// Construct field sized to mesh with optional initial value
    explicit CellData(const Mesh& mesh, const T& initialValue = T{})
    :
        internalField_(mesh.numCells(), initialValue)
    {}

// ****************************** Setter Methods ******************************

    /// Set all field values to a given value
    void setAll(const T& value)
    {
        std::fill(internalField_.begin(), internalField_.end(), value);
    }

// ***************************** Accessor Methods *****************************

    /// Get number of cells in the field
    [[nodiscard]] Count size() const noexcept
    {
        return internalField_.size();
    }

    /// Get pointer to field storage
    [[nodiscard]] T* data() noexcept
    {
        return internalField_.data();
    }

// ***************************** Operator Methods *****************************

    /// Unchecked subscript operator
    T& operator[](Index cellIndex) noexcept
    {
        return internalField_[cellIndex];
    }

    /// Unchecked const subscript operator
    const T& operator[](Index cellIndex) const noexcept
    {
        return internalField_[cellIndex];
    }

// ****************************** Private Members *****************************

private:

    /// Cell-centered field values
    std::vector<T> internalField_;
};

// ********************************** Aliases *********************************

/// Type alias for general scalar fields
using ScalarField = CellData<Scalar>;

/// Type alias for general vector fields
using VectorField = CellData<Vector>;

/// Type alias for general tensor fields
using TensorField = CellData<Tensor>;
