/******************************************************************************

                                     Turblyze
                           3D incompressible CFD solver
                       Copyright (C) 2025-2026 Mohamed Mousa
                        SPDX-License-Identifier: Apache-2.0

 ------------------------------------------------------------------------------
 * @file MeshChecker.cpp
 * @brief Mesh quality assessment utilities
 *****************************************************************************/

// ********************************** Headers *********************************

// Implementation header 
#include "MeshChecker.h"

// Standard library headers
#include <algorithm>
#include <cmath>
#include <format>
#include <iostream>
#include <numbers>
#include <string>
#include <vector>

// Project headers
#include "ErrorHandler.h"
#include "Integer.h"
#include "StringTypes.h"

// *************************** namespace MeshChecker **************************

namespace MeshChecker
{

// **************************** Quality Thresholds ****************************

constexpr Scalar minArea = S(1e-12);
constexpr Scalar minVolume = S(1e-30);
constexpr Scalar maxNonOrthThreshold = S(70.0);
constexpr Scalar maxSkewThreshold = S(4.0);
constexpr Scalar maxAspectThreshold = S(100.0);

// ***************************** Internal Helpers *****************************

namespace
{

// Print up to 10 IDs from a list, with truncation indicator
void printIndicesList
(
    const IndexList& indices,
    Name entityName
)
{
    constexpr Count maxDisplay = 10;
    const Count count = std::min(indices.size(), maxDisplay);

    std::string output = (indices.size() <= maxDisplay)
        ? std::format("  {} IDs: ", entityName)
        : std::format("  First {} {} IDs: ", maxDisplay, entityName);

    for (Index i = 0; i < count; ++i)
    {
        if (i > 0)
        {
            output += ", ";
        }

        output += std::format("{}", indices[i]);
    }

    if (indices.size() > maxDisplay)
    {
        output += " ...";
    }

    std::cout << output << '\n';
}


Scalar faceSkewness
(
    const Mesh& mesh,
    const Face& face,
    const Vector& ownerCellCentroid,
    const Vector& neighborCellCentroid
)
{
    const Vector dPf = face.centroid() - ownerCellCentroid;
    const Vector dPN = neighborCellCentroid - ownerCellCentroid;

    const Vector skewnessVector =
        dPf
      - ((dot(face.normal(), dPf))
      / (dot(face.normal(), dPN) + vSmallValue))
      * dPN;

    const Scalar skewnessMag = magnitude(skewnessVector);

    const Vector skewnessDirection =
        skewnessVector / (skewnessMag + smallValue);

    // Characteristic face dimension: empirical approximation
    Scalar faceCharacteristicLength = S(0.2) * magnitude(dPN) + vSmallValue;

    // Refine by finding max vertex extent in skewness direction
    const IndexList& nodeIndices = face.nodeIndices();

    for (Index nodeIdx : nodeIndices)
    {
        const Vector vertexToCentroid =
            mesh.nodes()[nodeIdx] - face.centroid();

        const Scalar projection =
            std::abs(dot(skewnessDirection, vertexToCentroid));

        faceCharacteristicLength =
            std::max(faceCharacteristicLength, projection);
    }

    // Return normalized skewness
    return skewnessMag / faceCharacteristicLength;
}


Scalar boundaryFaceSkewness
(
    const Mesh& mesh,
    const Face& face,
    const Vector& ownerCellCentroid
)
{
    const Vector dPf = face.centroid() - ownerCellCentroid;

    // Virtual dPN for boundary faces
    const Vector dPN = S(2.0) * dot(face.normal(), dPf) * face.normal();

    const Vector skewnessVector =
        dPf - dot(face.normal(), dPf) * face.normal();

    const Scalar skewnessMag = magnitude(skewnessVector);

    const Vector skewnessDirection =
        skewnessVector / (skewnessMag + smallValue);

    // Characteristic face dimension: empirical approximation
    Scalar faceCharacteristicLength = S(0.4) * magnitude(dPN) + vSmallValue;

    // Refine by finding max vertex extent in skewness direction
    const IndexList& nodeIndices = face.nodeIndices();

    for (Index nodeIdx : nodeIndices)
    {
        const Vector vertexToCentroid =
            mesh.nodes()[nodeIdx] - face.centroid();

        const Scalar projection =
            std::abs(dot(skewnessDirection, vertexToCentroid));

        faceCharacteristicLength =
            std::max(faceCharacteristicLength, projection);
    }

    return skewnessMag / faceCharacteristicLength;
}


Scalar cellAspectRatio
(
    const Mesh& mesh,
    const Cell& cell
)
{
    // Accumulate absolute face area components per direction
    Vector sumMagAreaComponents;

    const auto& faceIndices = cell.faceIndices();

    for (Index faceIdx : faceIndices)
    {
        const Face& face = mesh.faces()[faceIdx];
        const Vector areaVec = face.normal() * face.projectedArea();

        sumMagAreaComponents =
            Vector
            (
                sumMagAreaComponents.x() + std::abs(areaVec.x()),
                sumMagAreaComponents.y() + std::abs(areaVec.y()),
                sumMagAreaComponents.z() + std::abs(areaVec.z())
            );
    }

    // Find min and max projected areas
    const Scalar minComponent =
        std::min
        (
            {
                sumMagAreaComponents.x(),
                sumMagAreaComponents.y(),
                sumMagAreaComponents.z()
            }
        );

    const Scalar maxComponent =
        std::max
        (
            {
                sumMagAreaComponents.x(),
                sumMagAreaComponents.y(),
                sumMagAreaComponents.z()
            }
        );

    Scalar directionalAspect = maxComponent / (minComponent + vSmallValue);

    // Add hydraulic aspect ratio for 3D cells
    const Scalar totalSurfaceArea =
        sumMagAreaComponents.x()
      + sumMagAreaComponents.y()
      + sumMagAreaComponents.z();

    const Scalar volume = cell.volume();

    if (volume > vSmallValue)
    {
        // Hydraulic aspect ratio: (1/6) * A / V^(2/3)
        const Scalar hydraulicAspect =
            (S(1.0)/S(6.0)) * totalSurfaceArea
          / std::pow(volume, S(2.0)/S(3.0));

        directionalAspect = std::max(directionalAspect, hydraulicAspect);
    }

    return directionalAspect;
}


bool validateConnectivity(const Mesh& mesh)
{
    bool valid = true;

    for (const auto& face : mesh.faces())
    {
        if (face.ownerCell() >= mesh.numCells())
        {
            Warning
            (
                "Face " + std::to_string(face.idx())
              + " owner cell index "
              + std::to_string(face.ownerCell())
              + " out of range"
            );
            valid = false;
        }

        if
        (
            !face.isBoundary()
         && face.neighborCell().value() >= mesh.numCells())
        {
            Warning
            (
                "Face " + std::to_string(face.idx())
              + " neighbor cell index "
              + std::to_string(face.neighborCell().value())
              + " out of range"
            );
            valid = false;
        }

        for (Index nodeIdx : face.nodeIndices())
        {
            if (nodeIdx >= mesh.numNodes())
            {
                Warning
                (
                    "Face " + std::to_string(face.idx())
                  + " node index " + std::to_string(nodeIdx)
                  + " out of range"
                );
                valid = false;
            }
        }
    }

    for (const auto& cell : mesh.cells())
    {
        for (Index faceIdx : cell.faceIndices())
        {
            if (faceIdx >= mesh.numFaces())
            {
                Warning
                (
                    "Cell " + std::to_string(cell.idx())
                  + " face index " + std::to_string(faceIdx)
                  + " out of range"
                );
                valid = false;
            }
        }
    }

    return valid;
}

}

// ******************************** Mesh Check ********************************

void check(const Mesh& mesh)
{
    std::cout
        << "\n--- Mesh Quality Check ---" << '\n';

    if (mesh.faces().empty() || mesh.cells().empty())
    {
        FatalError("Empty mesh detected");
    }

    if (!validateConnectivity(mesh))
    {
        FatalError("Mesh connectivity validation failed");
    }

    // Face area statistics
    const Scalar firstArea = mesh.faces()[0].projectedArea();
    Scalar minFaceArea = firstArea;
    Scalar maxFaceArea = firstArea;
    Index minFaceIdx = mesh.faces()[0].idx();
    Index maxFaceIdx = mesh.faces()[0].idx();

    // Collect faces with small area
    IndexList smallAreaFaces;

    // Non-orthogonality statistics
    Scalar maxNonOrthogonality = S(0.0);
    Scalar totalCosAngle = S(0.0);
    Count nonOrthCount = 0;
    Index maxNonOrthFaceIdx = 0;
    IndexList severeNonOrthFaces;

    // Skewness statistics
    Scalar maxSkewness = S(0.0);
    Index maxSkewFaceIdx = 0;
    IndexList highSkewFaces;

    // Radian to degree conversion
    constexpr Scalar radToDeg = S(180.0) / std::numbers::pi_v<Scalar>;

    for (const auto& face : mesh.faces())
    {
        const Scalar area = face.projectedArea();
        const Index faceIdx = face.idx();

        // Area statistics
        if (area < minFaceArea)
        {
            minFaceArea = area;
            minFaceIdx = faceIdx;
        }

        if (area > maxFaceArea)
        {
            maxFaceArea = area;
            maxFaceIdx = faceIdx;
        }

        if (area < minArea)
        {
            smallAreaFaces.push_back(faceIdx);
        }

        // Calculate non-orthogonality and skewness
        if (face.isBoundary())
        {
            // Boundary face: calculate skewness only
            const Cell& ownerCell = mesh.cells()[face.ownerCell()];

            const Scalar skew =
                boundaryFaceSkewness(mesh, face, ownerCell.centroid());

            if (skew > maxSkewness)
            {
                maxSkewness = skew;
                maxSkewFaceIdx = faceIdx;
            }
            if (skew > maxSkewThreshold)
            {
                highSkewFaces.push_back(faceIdx);
            }
        }
        else
        {
            // Internal face: calculate both
            const Cell& ownerCell = mesh.cells()[face.ownerCell()];

            const Cell& neighborCell =
                mesh.cells()[face.neighborCell().value()];

            // Non-orthogonality (angle in degrees)
            const Vector dPN =
                neighborCell.centroid() - ownerCell.centroid();
            const Scalar ortho = std::clamp
            (
                dot(dPN, face.normal()) / (magnitude(dPN) + vSmallValue),
                S(-1.0),
                S(1.0)
            );

            const Scalar angleRad = std::acos(ortho);
            const Scalar angleDeg = angleRad * radToDeg;

            totalCosAngle += ortho;
            nonOrthCount++;

            if (angleDeg > maxNonOrthogonality)
            {
                maxNonOrthogonality = angleDeg;
                maxNonOrthFaceIdx = faceIdx;
            }

            if (angleDeg > maxNonOrthThreshold)
            {
                severeNonOrthFaces.push_back(faceIdx);
            }

            // Skewness
            const Scalar skew =
                faceSkewness
                (
                    mesh,
                    face,
                    ownerCell.centroid(),
                    neighborCell.centroid()
                );

            if (skew > maxSkewness)
            {
                maxSkewness = skew;
                maxSkewFaceIdx = faceIdx;
            }

            if (skew > maxSkewThreshold)
            {
                highSkewFaces.push_back(faceIdx);
            }
        }
    }

    // Calculate average non-orthogonality
    Scalar avgNonOrthogonality = S(0.0);
    if (nonOrthCount > 0)
    {
        avgNonOrthogonality =
            std::acos(totalCosAngle / S(nonOrthCount)) * radToDeg;
    }

    // Cell volume and aspect ratio statistics
    Scalar minCellVolume = mesh.cells()[0].volume();
    Scalar maxCellVolume = mesh.cells()[0].volume();
    Index minCellIdx = mesh.cells()[0].idx();
    Index maxCellIdx = mesh.cells()[0].idx();

    Scalar maxAspectRatio = S(0.0);
    Index maxAspectCellIdx = 0;
    IndexList highAspectCells;

    IndexList smallVolumeCells;
    IndexList invertedCells;

    for (const auto& cell : mesh.cells())
    {
        // Volume statistics
        if (cell.volume() < minCellVolume)
        {
            minCellVolume = cell.volume();
            minCellIdx = cell.idx();
        }

        if (cell.volume() > maxCellVolume)
        {
            maxCellVolume = cell.volume();
            maxCellIdx = cell.idx();
        }

        // Check for inverted or small cells
        if (cell.volume() < S(0.0))
        {
            invertedCells.push_back(cell.idx());
        }
        else if (cell.volume() < minVolume)
        {
            smallVolumeCells.push_back(cell.idx());
        }

        // Calculate aspect ratio
        const Scalar aspectRatio = cellAspectRatio(mesh, cell);
        if (aspectRatio > maxAspectRatio)
        {
            maxAspectRatio = aspectRatio;
            maxAspectCellIdx = cell.idx();
        }

        // High aspect ratio threshold
        if (aspectRatio > maxAspectThreshold)
        {
            highAspectCells.push_back(cell.idx());
        }
    }

    std::cout << std::format
    (
        "\nFace Area Statistics:\n"
        "  Minimum area: {:.6e} m² (face {})\n"
        "  Maximum area: {:.6e} m² (face {})\n",
        minFaceArea, minFaceIdx, maxFaceArea, maxFaceIdx
    );

    std::cout << std::format
    (
        "\nCell Volume Statistics:\n"
        "  Minimum volume: {:.6e} m³ (cell {})\n"
        "  Maximum volume: {:.6e} m³ (cell {})\n",
        minCellVolume, minCellIdx, maxCellVolume, maxCellIdx
    );

    if (!invertedCells.empty())
    {
        FatalError
        (
            std::format
            (
                "{} inverted cells (negative volume) detected",
                invertedCells.size()
            )
        );
    }

    // Non-orthogonality statistics
    std::cout << "\nNon-Orthogonality Statistics:\n";

    if (nonOrthCount > 0)
    {
        std::cout << std::format
        (
            "  Maximum: {:.2f}° (face {})\n"
            "  Average: {:.2f}°\n",
            maxNonOrthogonality, maxNonOrthFaceIdx, avgNonOrthogonality
        );
    }
    else
    {
        std::cout << "  No internal faces to measure\n";
    }

    if (!severeNonOrthFaces.empty())
    {
        Warning
        (
            std::format
            (
                "{} faces with non-orthogonality > {}°",
                severeNonOrthFaces.size(), maxNonOrthThreshold
            )
        );

        printIndicesList(severeNonOrthFaces, "Face");
    }

    // Skewness statistics
    std::cout << std::format
    (
        "\nSkewness Statistics:\n"
        "  Maximum: {:.3f} (face {})\n",
        maxSkewness, maxSkewFaceIdx
    );

    if (!highSkewFaces.empty())
    {
        Warning
        (
            std::format
            (
                "{} faces with skewness > {}",
                highSkewFaces.size(), maxSkewThreshold
            )
        );

        printIndicesList(highSkewFaces, "Face");
    }

    // Aspect ratio statistics
    std::cout << std::format
    (
        "\nAspect Ratio Statistics:\n"
        "  Maximum: {:.1f} (cell {})\n",
        maxAspectRatio, maxAspectCellIdx
    );

    if (!highAspectCells.empty())
    {
        Warning
        (
            std::format
            (
                "{} cells with aspect ratio > {}",
                highAspectCells.size(), maxAspectThreshold
            )
        );

        printIndicesList(highAspectCells, "Cell");
    }

    // Quality warnings for small areas/volumes
    if (!smallAreaFaces.empty())
    {
        std::cout << std::format
        (
            "\nQuality Check - Small Face Areas:\n"
            "  Found {} faces with area < {:.0e} m²\n",
            smallAreaFaces.size(), minArea
        );

        printIndicesList(smallAreaFaces, "Face");
    }

    if (!smallVolumeCells.empty())
    {
        std::cout << std::format
        (
            "\nQuality Check - Small Cell Volumes:\n"
            "  Found {} cells with volume < {:.0e} m³\n",
            smallVolumeCells.size(), minVolume
        );

        printIndicesList(smallVolumeCells, "Cell");
    }

    // Overall mesh quality summary
    std::cout << "\n--- Mesh Quality Summary ---\n";

    bool goodQuality = true;

    if (maxNonOrthogonality > maxNonOrthThreshold)
    {
        Warning
        (
            std::format
            (
                "Non-orthogonality exceeds {}° threshold",
                maxNonOrthThreshold
            )
        );
        goodQuality = false;
    }

    if (maxSkewness > maxSkewThreshold)
    {
        Warning
        (
            std::format
            (
                "Skewness exceeds {} threshold",
                maxSkewThreshold
            )
        );
        goodQuality = false;
    }

    if (maxAspectRatio > maxAspectThreshold)
    {
        Warning
        (
            std::format
            (
                "Aspect ratio exceeds {} threshold",
                maxAspectThreshold
            )
        );
        goodQuality = false;
    }

    if
    (
        smallAreaFaces.empty()
     && smallVolumeCells.empty()
     && invertedCells.empty()
     && goodQuality
    )
    {
        std::cout << "DONE: All mesh quality metrics within acceptable ranges\n";
    }
}

} // namespace MeshChecker
