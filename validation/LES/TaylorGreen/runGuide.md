<!--
SPDX-FileCopyrightText: 2025-2026 Mohamed Mousa
SPDX-License-Identifier: Apache-2.0
-->

# Reproducing The Taylor-Green Vortex LES Verification

Commands assume the Turblyze executable is built in `build.nosync/`.

## 1. Build Solver

```bash
cmake -S . -B build.nosync -DCMAKE_BUILD_TYPE=Release
cmake --build build.nosync -j
```

## 2. Generate Hexahedral Mesh ($128^3$)

Generate the $[0, \pi L]^3$ mesh with symmetry patches:

```bash
python3 validation/LES/TaylorGreen/scripts/generateMesh.py --n 128 --out inputFiles/taylorGreen128.msh
```

*(To run a faster turnaround test, you can specify `--n 64` or `--n 32`)*.

## 3. Run Simulation

Run from the `build.nosync/` directory:

```bash
cd build.nosync
./Turblyze ../validation/LES/TaylorGreen/TaylorGreenCase
```

The output solution is written to `outputFiles.nosync/taylorGreen.vtkhdf`.

## 4. Extract Results & Compare with DNS

Run the post-processing script:

```bash
python3 validation/LES/TaylorGreen/scripts/analyzeTaylorGreen.py \
    --vtkhdf outputFiles.nosync/taylorGreen.vtkhdf \
    --dns validation/LES/TaylorGreen/reference/dnsRe1600.csv \
    --outdir validation/LES/TaylorGreen/results
```

This generates:
- `validation/LES/TaylorGreen/results/taylorGreenLESResults.csv`
- `validation/LES/TaylorGreen/results/dnsComparisonSummary.md`
- `validation/LES/TaylorGreen/results/tkeComparison.svg`
- `validation/LES/TaylorGreen/results/dissipationComparison.svg`
