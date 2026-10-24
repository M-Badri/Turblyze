#!/usr/bin/env python3
"""
generateMesh.py
---------------
Generates a 3D structured Cartesian hexahedral mesh in Fluent (.msh) ASCII format
for the Taylor-Green Vortex on the impermeable symmetry box [0, pi * L]^3.

Domain: [0, pi * L]^3 (default L = 1.0)
Boundary Patches (all type symmetry):
  - xMin (x = 0), xMax (x = pi * L)
  - yMin (y = 0), yMax (y = pi * L)
  - zMin (z = 0), zMax (z = pi * L)

Zero external dependencies (pure Python standard library).
"""

import argparse
import math
import os
import sys


def parseArgs():
    parser = argparse.ArgumentParser(
        description="Generate 3D Cartesian .msh mesh for Taylor-Green Vortex [0, pi*L]^3."
    )
    parser.add_argument(
        "--n",
        type=int,
        default=128,
        help="Number of cells per coordinate direction (default: 128, giving 128^3 = 2,097,152 cells)",
    )
    parser.add_argument(
        "--L",
        type=float,
        default=1.0,
        help="Characteristic length scale L (default: 1.0, domain length = pi * L)",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="inputFiles/taylorGreen128.msh",
        help="Output path for the .msh file (default: inputFiles/taylorGreen128.msh)",
    )
    return parser.parse_args()


def toHex(val: int) -> str:
    """Format an integer as a lowercase hex string without '0x' prefix."""
    return format(val, "x")


def generateMesh(n: int, L: float, outPath: str):
    nx = ny = nz = n
    lx = ly = lz = math.pi * L
    dx = lx / nx
    dy = ly / ny
    dz = lz / nz

    numCells = nx * ny * nz
    numNodes = (nx + 1) * (ny + 1) * (nz + 1)

    print(f"Generating Taylor-Green mesh on [0, {lx:.6f}]^3:")
    print(f"  Resolution: {nx} x {ny} x {nz} cells ({numCells:,} total)")
    print(f"  Nodes: {nx + 1} x {ny + 1} x {nz + 1} ({numNodes:,} total)")
    print(f"  Grid spacing (filter width): dx = dy = dz = {dx:.6e} m")
    print(f"  Writing to: {outPath}")

    outDir = os.path.dirname(outPath)
    if outDir:
        os.makedirs(outDir, exist_ok=True)

    # 1-based node index for (i, j, k) where i in [0, nx], j in [0, ny], k in [0, nz]
    def nodeId(i: int, j: int, k: int) -> int:
        return 1 + i + (nx + 1) * (j + (ny + 1) * k)

    # 1-based cell index for (i, j, k) where i in [0, nx-1], j in [0, ny-1], k in [0, nz-1]
    def cellId(i: int, j: int, k: int) -> int:
        return 1 + i + nx * (j + ny * k)

    # Face counts
    numInternalX = (nx - 1) * ny * nz
    numInternalY = nx * (ny - 1) * nz
    numInternalZ = nx * ny * (nz - 1)
    numInternalFaces = numInternalX + numInternalY + numInternalZ

    numBndXMin = ny * nz
    numBndXMax = ny * nz
    numBndYMin = nx * nz
    numBndYMax = nx * nz
    numBndZMin = nx * ny
    numBndZMax = nx * ny
    numBndFaces = (
        numBndXMin
        + numBndXMax
        + numBndYMin
        + numBndYMax
        + numBndZMin
        + numBndZMax
    )

    totalFaces = numInternalFaces + numBndFaces

    intStart = 1
    intEnd = numInternalFaces

    xMinStart = intEnd + 1
    xMinEnd = xMinStart + numBndXMin - 1

    xMaxStart = xMinEnd + 1
    xMaxEnd = xMaxStart + numBndXMax - 1

    yMinStart = xMaxEnd + 1
    yMinEnd = yMinStart + numBndYMin - 1

    yMaxStart = yMinEnd + 1
    yMaxEnd = yMaxStart + numBndYMax - 1

    zMinStart = yMaxEnd + 1
    zMinEnd = zMinStart + numBndZMin - 1

    zMaxStart = zMinEnd + 1
    zMaxEnd = zMaxStart + numBndZMax - 1

    assert zMaxEnd == totalFaces, f"Face count mismatch: {zMaxEnd} != {totalFaces}"

    with open(outPath, "w", encoding="ascii") as f:
        # 1. Header & Dimensions
        f.write('(0 "Taylor-Green Vortex [0, pi*L]^3 hexahedral mesh")\n')
        f.write("(2 3)\n")

        # 2. Total Declarations
        f.write(f"(10 (0 1 {toHex(numNodes)} 0))\n")
        f.write(f"(12 (0 1 {toHex(numCells)} 0))\n")
        f.write(f"(13 (0 1 {toHex(totalFaces)} 0))\n")

        # 3. Nodes Section (Zone 1)
        f.write(f"(10 (1 1 {toHex(numNodes)} 1 3)(\n")
        for k in range(nz + 1):
            zk = k * dz
            for j in range(ny + 1):
                yj = j * dy
                for i in range(nx + 1):
                    xi = i * dx
                    f.write(f"{xi:.10e} {yj:.10e} {zk:.10e}\n")
        f.write("))\n")

        # 4. Cells Section (Zone 1, uniform hex type 4)
        f.write(f"(12 (1 1 {toHex(numCells)} 1 4))\n")

        # 5. Internal Faces Section (Zone 2)
        f.write(f"(13 (2 {toHex(intStart)} {toHex(intEnd)} 2 4)(\n")

        # x-normal internal faces at planes i = 1 .. nx-1 (+x winding, owner i-1, neighbor i)
        for k in range(nz):
            for j in range(ny):
                for i in range(1, nx):
                    owner = cellId(i - 1, j, k)
                    neigh = cellId(i, j, k)
                    n1 = nodeId(i, j, k)
                    n2 = nodeId(i, j, k + 1)
                    n3 = nodeId(i, j + 1, k + 1)
                    n4 = nodeId(i, j + 1, k)
                    f.write(
                        f"{toHex(n1)} {toHex(n2)} {toHex(n3)} {toHex(n4)} "
                        f"{toHex(owner)} {toHex(neigh)}\n"
                    )

        # y-normal internal faces at planes j = 1 .. ny-1 (+y winding, owner j-1, neighbor j)
        for k in range(nz):
            for j in range(1, ny):
                for i in range(nx):
                    owner = cellId(i, j - 1, k)
                    neigh = cellId(i, j, k)
                    n1 = nodeId(i, j, k)
                    n2 = nodeId(i + 1, j, k)
                    n3 = nodeId(i + 1, j, k + 1)
                    n4 = nodeId(i, j, k + 1)
                    f.write(
                        f"{toHex(n1)} {toHex(n2)} {toHex(n3)} {toHex(n4)} "
                        f"{toHex(owner)} {toHex(neigh)}\n"
                    )

        # z-normal internal faces at planes k = 1 .. nz-1 (+z winding, owner k-1, neighbor k)
        for k in range(1, nz):
            for j in range(ny):
                for i in range(nx):
                    owner = cellId(i, j, k - 1)
                    neigh = cellId(i, j, k)
                    n1 = nodeId(i, j, k)
                    n2 = nodeId(i, j + 1, k)
                    n3 = nodeId(i + 1, j + 1, k)
                    n4 = nodeId(i + 1, j, k)
                    f.write(
                        f"{toHex(n1)} {toHex(n2)} {toHex(n3)} {toHex(n4)} "
                        f"{toHex(owner)} {toHex(neigh)}\n"
                    )
        f.write("))\n")

        # 6. Boundary Faces
        # Zone 3: xMin (i = 0). Normal points -x (outward from cell (0, j, k))
        f.write(f"(13 (3 {toHex(xMinStart)} {toHex(xMinEnd)} 3 4)(\n")
        i = 0
        for k in range(nz):
            for j in range(ny):
                owner = cellId(i, j, k)
                n1 = nodeId(i, j + 1, k)
                n2 = nodeId(i, j + 1, k + 1)
                n3 = nodeId(i, j, k + 1)
                n4 = nodeId(i, j, k)
                f.write(
                    f"{toHex(n1)} {toHex(n2)} {toHex(n3)} {toHex(n4)} "
                    f"{toHex(owner)} 0\n"
                )
        f.write("))\n")

        # Zone 4: xMax (i = nx). Normal points +x (outward from cell (nx-1, j, k))
        f.write(f"(13 (4 {toHex(xMaxStart)} {toHex(xMaxEnd)} 3 4)(\n")
        i = nx
        for k in range(nz):
            for j in range(ny):
                owner = cellId(nx - 1, j, k)
                n1 = nodeId(i, j, k)
                n2 = nodeId(i, j, k + 1)
                n3 = nodeId(i, j + 1, k + 1)
                n4 = nodeId(i, j + 1, k)
                f.write(
                    f"{toHex(n1)} {toHex(n2)} {toHex(n3)} {toHex(n4)} "
                    f"{toHex(owner)} 0\n"
                )
        f.write("))\n")

        # Zone 5: yMin (j = 0). Normal points -y (outward from cell (i, 0, k))
        f.write(f"(13 (5 {toHex(yMinStart)} {toHex(yMinEnd)} 3 4)(\n")
        j = 0
        for k in range(nz):
            for i in range(nx):
                owner = cellId(i, j, k)
                n1 = nodeId(i, 0, k + 1)
                n2 = nodeId(i + 1, 0, k + 1)
                n3 = nodeId(i + 1, 0, k)
                n4 = nodeId(i, 0, k)
                f.write(
                    f"{toHex(n1)} {toHex(n2)} {toHex(n3)} {toHex(n4)} "
                    f"{toHex(owner)} 0\n"
                )
        f.write("))\n")

        # Zone 6: yMax (j = ny). Normal points +y (outward from cell (i, ny-1, k))
        f.write(f"(13 (6 {toHex(yMaxStart)} {toHex(yMaxEnd)} 3 4)(\n")
        j = ny
        for k in range(nz):
            for i in range(nx):
                owner = cellId(i, ny - 1, k)
                n1 = nodeId(i, ny, k)
                n2 = nodeId(i + 1, ny, k)
                n3 = nodeId(i + 1, ny, k + 1)
                n4 = nodeId(i, ny, k + 1)
                f.write(
                    f"{toHex(n1)} {toHex(n2)} {toHex(n3)} {toHex(n4)} "
                    f"{toHex(owner)} 0\n"
                )
        f.write("))\n")

        # Zone 7: zMin (k = 0). Normal points -z (outward from cell (i, j, 0))
        f.write(f"(13 (7 {toHex(zMinStart)} {toHex(zMinEnd)} 3 4)(\n")
        k = 0
        for j in range(ny):
            for i in range(nx):
                owner = cellId(i, j, k)
                n1 = nodeId(i + 1, j, 0)
                n2 = nodeId(i + 1, j + 1, 0)
                n3 = nodeId(i, j + 1, 0)
                n4 = nodeId(i, j, 0)
                f.write(
                    f"{toHex(n1)} {toHex(n2)} {toHex(n3)} {toHex(n4)} "
                    f"{toHex(owner)} 0\n"
                )
        f.write("))\n")

        # Zone 8: zMax (k = nz). Normal points +z (outward from cell (i, j, nz-1))
        f.write(f"(13 (8 {toHex(zMaxStart)} {toHex(zMaxEnd)} 3 4)(\n")
        k = nz
        for j in range(ny):
            for i in range(nx):
                owner = cellId(i, j, nz - 1)
                n1 = nodeId(i, j, nz)
                n2 = nodeId(i, j + 1, nz)
                n3 = nodeId(i + 1, j + 1, nz)
                n4 = nodeId(i + 1, j, nz)
                f.write(
                    f"{toHex(n1)} {toHex(n2)} {toHex(n3)} {toHex(n4)} "
                    f"{toHex(owner)} 0\n"
                )
        f.write("))\n")

        # 7. Boundary Zone Names & Types (Section 45)
        f.write("(45 (3 symmetry xMin)())\n")
        f.write("(45 (4 symmetry xMax)())\n")
        f.write("(45 (5 symmetry yMin)())\n")
        f.write("(45 (6 symmetry yMax)())\n")
        f.write("(45 (7 symmetry zMin)())\n")
        f.write("(45 (8 symmetry zMax)())\n")

    print(f"Mesh written successfully to {outPath}.")


if __name__ == "__main__":
    args = parseArgs()
    generateMesh(args.n, args.L, args.out)
