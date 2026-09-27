/******************************************************************************

                                     Turblyze
                           3D incompressible CFD solver
                       Copyright (C) 2025-2026 Mohamed Mousa
                        SPDX-License-Identifier: Apache-2.0

 ------------------------------------------------------------------------------
 * @file Face.h
 * @brief Represents a face in the computational mesh
 *
 * @details The face is defined by a sequence of nodes (vertices). Each face
 * has an owner cell and may have a neighbor cell (for internal faces). In 
 * addition to connectivity, the face stores geometric properties such as
 * centroid, normal, and area.
 *****************************************************************************/

#pragma once

// ********************************** Headers *********************************

// Standard library headers
#include <optional>
#include <utility>

// Project headers
#include "Integer.h"
#include "Scalar.h"
#include "Vector.h"
#include "BoundaryPatch.h"
#include "MeshContainers.h"

// *************************** Forward Declarations ***************************

class Cell;

// *************************** struct FaceIntegrals ***************************

struct FaceIntegrals
{
    Scalar x2 = S(0.0);
    Scalar y2 = S(0.0);
    Scalar z2 = S(0.0);
    Scalar volume = S(0.0);
};

// ******************************** class Face ********************************

class Face
{
public:

// ************************* Special Member Functions *************************

    /// Default constructor
    Face() = default;

    /// Constructor for internal faces
    Face
    (
        Index faceIdx,
        IndexList nodes,
        Index owner,
        Index neighbor
    )
    :
        idx_(faceIdx),
        nodeIndices_(std::move(nodes)),
        ownerCell_(owner),
        neighborCell_(neighbor)
    {}

    /// Constructor for boundary faces
    Face
    (
        Index faceIdx,
        IndexList nodes,
        Index owner
    )
    :
        idx_(faceIdx),
        nodeIndices_(std::move(nodes)),
        ownerCell_(owner),
        neighborCell_(std::nullopt)
    {}

// ****************************** Setter Methods ******************************

    /// Set face identifier
    void setIdx(Index faceIdx) noexcept
    {
        idx_ = faceIdx;
    }

    /// Set owner cell index
    void setOwnerCell(Index owner) noexcept
    {
        ownerCell_ = owner;
    }

    /// Set neighbor cell index
    void setNeighborCell(Index neighbor) noexcept
    {
        neighborCell_ = neighbor;
    }

    /// Set neighbor cell to null
    void setNeighborCell(std::nullopt_t) noexcept
    {
        neighborCell_ = std::nullopt;
    }

    /// Add node index to face connectivity
    void addNodeIndex(Index nodeIdx)
    {
        nodeIndices_.push_back(nodeIdx);
    }

    /// Clear all node indices
    void clearNodeIndices() noexcept
    {
        nodeIndices_.clear();
    }

    /// Set the boundary patch this face belongs to
    void setPatch(const BoundaryPatch& p) noexcept
    {
        patch_ = &p;
    }

// ***************************** Accessor Methods *****************************

    /// Get face identifier
    [[nodiscard]] Index idx() const noexcept
    {
        return idx_;
    }

    /// Get node connectivity
    [[nodiscard]] const IndexList& nodeIndices() const noexcept
    {
        return nodeIndices_;
    }

    /// Get owner cell index
    [[nodiscard]] Index ownerCell() const noexcept
    {
        return ownerCell_;
    }

    /// Get neighbor cell index
    [[nodiscard]] const std::optional<Index>& neighborCell() const noexcept
    {
        return neighborCell_;
    }

    /// Get face centroid
    [[nodiscard]] const Vector& centroid() const noexcept
    {
        return centroid_;
    }

    /// Get face normal vector
    [[nodiscard]] const Vector& normal() const noexcept
    {
        return normal_;
    }

    /// Get face area for flux calculations
    [[nodiscard]] Scalar projectedArea() const noexcept
    {
        return projectedArea_;
    }

    /// Get face contact area
    [[nodiscard]] Scalar contactArea() const noexcept
    {
        return contactArea_;
    }

    /// Get the boundary patch this face belongs to
    [[nodiscard]] const BoundaryPatch* patch() const noexcept
    {
        return patch_;
    }

    /// Check if this is a boundary face
    [[nodiscard]] bool isBoundary() const noexcept
    {
        return !neighborCell_.has_value();
    }

// ************************ Geometric Property Methods ************************

    /// Calculate Face centroid, normal, area, and second moment integral
    [[nodiscard]] FaceIntegrals geometricProperties
    (
        const NodeList& allNodes
    );

// ****************************** Private Members *****************************

private:

    /// Unique face identifier
    Index idx_ = 0;

    /// Indices of nodes that define this face
    IndexList nodeIndices_;

    /// Index of the owner cell
    Index ownerCell_ = 0;

    /// Index of neighbor cell (nullopt for boundary faces)
    std::optional<Index> neighborCell_;

    /// Face geometric centroid
    Vector centroid_;

    /// Face normal vector (unit vector)
    Vector normal_;

    /// Face area (projected area for flux calculations)
    Scalar projectedArea_ = S(0.0);

    /// Contact area (For shear stress calculations)
    Scalar contactArea_ = S(0.0);

    /// Owning boundary patch (nullptr for internal or unlinked faces)
    const BoundaryPatch* patch_ = nullptr;
};
