/******************************************************************************

                                     Turblyze
                           3D incompressible CFD solver
                       Copyright (C) 2025-2026 Mohamed Mousa
                        SPDX-License-Identifier: Apache-2.0

 ------------------------------------------------------------------------------
 * @file MeshCreator.h
 * @brief Mesh read, partition, and geometry-preparation
 *****************************************************************************/

#pragma once

// ********************************** Headers *********************************

// Project headers
#include "CaseConfiguration.h"
#include "Mesh.h"

// *************************** namespace MeshCreator **************************

namespace MeshCreator
{

/// Link boundary faces to their owning boundary patches
void linkBoundaryFaces(FaceList& faces, const PatchList& patches);

/// Read, prepare, and optionally quality-check the configured mesh
[[nodiscard]] Mesh create(const CaseConfiguration& config);

/// Partition the master's complete mesh and rebuild this rank's submesh
[[nodiscard]] Mesh decomposeAndDistribute(Mesh completeMesh, bool debug = false);

} // namespace MeshCreator
