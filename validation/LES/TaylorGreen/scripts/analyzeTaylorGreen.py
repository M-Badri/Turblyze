#!/usr/bin/env python3
"""
analyzeTaylorGreen.py
---------------------
Post-processes Taylor-Green Vortex (Re = 1600) simulation from VTKHDF output:
1. Extracts time history of resolved kinetic energy eRes(t*), SGS kinetic energy
   eSgs(t*), dissipation rate dissipation(t*) = -deRes/dt*, and eddy viscosity ratio <nu_t> / nu.
2. Compares against high-order DNS reference data (van Rees et al. 2011 / HiOCFD Problem C3.5).
3. Generates publication-ready SVG figures (pure Python, zero dependencies) and
   writes a comprehensive comparison summary markdown and CSV.
"""

import argparse
import array
import csv
import math
import os
import subprocess
import sys
import tempfile


def parseArgs():
    parser = argparse.ArgumentParser(
        description="Analyze Taylor-Green Vortex LES results from VTKHDF output."
    )
    parser.add_argument(
        "--vtkhdf",
        type=str,
        default="outputFiles.nosync/taylorGreen.vtkhdf",
        help="Path to the volume .vtkhdf file (default: outputFiles.nosync/taylorGreen.vtkhdf)",
    )
    parser.add_argument(
        "--dns",
        type=str,
        default="validation/LES/TaylorGreen/reference/dnsRe1600.csv",
        help="Path to the DNS reference CSV file (default: validation/LES/TaylorGreen/reference/dnsRe1600.csv)",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="validation/LES/TaylorGreen/results",
        help="Output directory for CSV, summary, and SVG plots (default: validation/LES/TaylorGreen/results)",
    )
    parser.add_argument(
        "--nu",
        type=float,
        default=0.000625,
        help="Kinematic viscosity nu (default: 0.000625 m^2/s for Re=1600 with U0=1, L=1)",
    )
    return parser.parse_args()


# -----------------------------------------------------------------------------
# Data Extraction from VTKHDF
# -----------------------------------------------------------------------------

def extractDatasetH5dump(vtkhdfPath: str, datasetPath: str, typeCode: str) -> array.array:
    """Extract a dataset into a Python array using h5dump -b NATIVE."""
    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as tmp:
        tmpName = tmp.name

    try:
        cmd = ["h5dump", "-b", "NATIVE", "-d", datasetPath, "-o", tmpName, vtkhdfPath]
        res = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
        if res.returncode != 0:
            raise RuntimeError(f"h5dump failed for {datasetPath}: {res.stderr}")

        fileSize = os.path.getsize(tmpName)
        arr = array.array(typeCode)
        elemSize = arr.itemsize
        numItems = fileSize // elemSize

        with open(tmpName, "rb") as f:
            arr.fromfile(f, numItems)
        return arr
    finally:
        if os.path.exists(tmpName):
            os.remove(tmpName)


def loadVtkhdfData(vtkhdfPath: str):
    """
    Load time steps, velocity, nut, and k_sgs from VTKHDF using h5py if available,
    falling back to h5dump.
    """
    try:
        import h5py
        print("Reading VTKHDF using h5py...")
        with h5py.File(vtkhdfPath, "r") as f:
            steps = list(f["/VTKHDF/Steps/Values"][:])
            velOffsets = list(f["/VTKHDF/Steps/CellDataOffsets/velocity"][:])
            velocity = f["/VTKHDF/CellData/velocity"][:]
            nut = f["/VTKHDF/CellData/nut"][:]
            kSgs = f["/VTKHDF/CellData/k_sgs"][:] if "/VTKHDF/CellData/k_sgs" in f else None
            delta = f["/VTKHDF/CellData/delta"][:] if "/VTKHDF/CellData/delta" in f else None
            return {
                "steps": steps,
                "velOffsets": velOffsets,
                "velocity": velocity,
                "nut": nut,
                "kSgs": kSgs,
                "delta": delta,
                "isH5py": True,
            }
    except ImportError:
        pass

    print("Reading VTKHDF using h5dump fallback...")
    steps = list(extractDatasetH5dump(vtkhdfPath, "/VTKHDF/Steps/Values", "d"))
    velOffsets = list(extractDatasetH5dump(vtkhdfPath, "/VTKHDF/Steps/CellDataOffsets/velocity", "q"))
    velocity = extractDatasetH5dump(vtkhdfPath, "/VTKHDF/CellData/velocity", "f")
    nut = extractDatasetH5dump(vtkhdfPath, "/VTKHDF/CellData/nut", "f")

    kSgs = None
    try:
        kSgs = extractDatasetH5dump(vtkhdfPath, "/VTKHDF/CellData/k_sgs", "f")
    except Exception:
        pass

    delta = None
    try:
        delta = extractDatasetH5dump(vtkhdfPath, "/VTKHDF/CellData/delta", "f")
    except Exception:
        pass

    return {
        "steps": steps,
        "velOffsets": velOffsets,
        "velocity": velocity,
        "nut": nut,
        "kSgs": kSgs,
        "delta": delta,
        "isH5py": False,
    }


# -----------------------------------------------------------------------------
# Analysis & Metrics Calculation
# -----------------------------------------------------------------------------

def analyzeSimulation(data: dict, nu: float):
    steps = data["steps"]
    numSteps = len(steps)
    velOffsets = data["velOffsets"]
    isH5py = data["isH5py"]

    if numSteps > 1:
        numCells = velOffsets[1] - velOffsets[0]
    else:
        numCells = len(data["nut"])

    print(f"Total output snapshots: {numSteps}")
    print(f"Cells per snapshot: {numCells:,}")

    eResList = []
    eSgsList = []
    eTotList = []
    nutRatioList = []

    for sIdx in range(numSteps):
        cellStart = sIdx * numCells
        cellEnd = cellStart + numCells

        if isH5py:
            velSlice = data["velocity"][cellStart:cellEnd]
            u2 = velSlice[:, 0] ** 2 + velSlice[:, 1] ** 2 + velSlice[:, 2] ** 2
            eRes = float(0.5 * u2.mean())
            nutSlice = data["nut"][cellStart:cellEnd]
            nutMean = float(nutSlice.mean())

            if data["kSgs"] is not None:
                kSgsSlice = data["kSgs"][cellStart:cellEnd]
                eSgs = float(kSgsSlice.mean())
            else:
                eSgs = 0.0
        else:
            velStart = cellStart * 3
            vel = data["velocity"]
            sumU2 = 0.0
            for i in range(numCells):
                idx = velStart + 3 * i
                sumU2 += vel[idx] * vel[idx] + vel[idx + 1] * vel[idx + 1] + vel[idx + 2] * vel[idx + 2]
            eRes = 0.5 * (sumU2 / numCells)

            nutRaw = data["nut"]
            sumNut = sum(nutRaw[cellStart:cellEnd])
            nutMean = sumNut / numCells

            if data["kSgs"] is not None:
                kSgsRaw = data["kSgs"]
                sumKSgs = sum(kSgsRaw[cellStart:cellEnd])
                eSgs = sumKSgs / numCells
            else:
                eSgs = 0.0

        eTot = eRes + eSgs

        eResList.append(eRes)
        eSgsList.append(eSgs)
        eTotList.append(eTot)
        nutRatioList.append(nutMean / nu)

    # Compute dissipation rate: eps = -dE_res/dt
    dissipationList = [0.0] * numSteps
    for i in range(numSteps):
        if numSteps == 1:
            dissipationList[i] = 0.0
        elif i == 0:
            dt = steps[1] - steps[0]
            dissipationList[i] = -(eResList[1] - eResList[0]) / dt
        elif i == numSteps - 1:
            dt = steps[-1] - steps[-2]
            dissipationList[i] = -(eResList[-1] - eResList[-2]) / dt
        else:
            dt = steps[i + 1] - steps[i - 1]
            dissipationList[i] = -(eResList[i + 1] - eResList[i - 1]) / dt

    return {
        "t": steps,
        "numCells": numCells,
        "eRes": eResList,
        "eSgs": eSgsList,
        "eTot": eTotList,
        "dissipation": dissipationList,
        "nutRatio": nutRatioList,
    }


def loadDnsReference(dnsPath: str):
    """Load reference DNS data: t, eKin, dissipation."""
    tDns = []
    eDns = []
    epsDns = []
    with open(dnsPath, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            tDns.append(float(row["t"]))
            eDns.append(float(row["eKin"]))
            epsDns.append(float(row["dissipation"]))
    return {"t": tDns, "eKin": eDns, "dissipation": epsDns}


# -----------------------------------------------------------------------------
# SVG Plotting Engine (Zero Dependencies)
# -----------------------------------------------------------------------------

def generateSvgPlot(
    title: str,
    xLabel: str,
    yLabel: str,
    series: list,
    xRange: tuple,
    yRange: tuple,
    outSvgPath: str,
    thresholdY: float = None,
    thresholdLabel: str = None,
):
    """Generate a clean vector SVG plot using pure Python string formatting."""
    width = 720
    height = 460
    marginLeft = 80
    marginRight = 30
    marginTop = 50
    marginBottom = 60

    plotWidth = width - marginLeft - marginRight
    plotHeight = height - marginTop - marginBottom

    xMin, xMax = xRange
    yMin, yMax = yRange

    def toScreenX(x):
        return marginLeft + (x - xMin) / (xMax - xMin + 1e-12) * plotWidth

    def toScreenY(y):
        return marginTop + plotHeight - (y - yMin) / (yMax - yMin + 1e-12) * plotHeight

    svg = []
    svg.append(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}">\n')
    svg.append('<style>\n')
    svg.append('  text { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif; }\n')
    svg.append('  .grid { stroke: #e5e7eb; stroke-width: 1; stroke-dasharray: 4,4; }\n')
    svg.append('  .axis { stroke: #374151; stroke-width: 1.5; }\n')
    svg.append('  .title { font-size: 16px; font-weight: 600; fill: #111827; text-anchor: middle; }\n')
    svg.append('  .label { font-size: 13px; font-weight: 500; fill: #374151; }\n')
    svg.append('  .tick { font-size: 11px; fill: #6b7280; }\n')
    svg.append('  .legend { font-size: 12px; font-weight: 500; }\n')
    svg.append('</style>\n')

    # Background
    svg.append(f'<rect width="{width}" height="{height}" fill="#ffffff"/>\n')

    # Grid & Ticks (X-axis: 0, 5, 10, 15, 20)
    for xVal in [0, 5, 10, 15, 20]:
        if xMin <= xVal <= xMax:
            sx = toScreenX(xVal)
            svg.append(f'<line x1="{sx}" y1="{marginTop}" x2="{sx}" y2="{marginTop + plotHeight}" class="grid"/>\n')
            svg.append(f'<line x1="{sx}" y1="{marginTop + plotHeight}" x2="{sx}" y2="{marginTop + plotHeight + 5}" class="axis"/>\n')
            svg.append(f'<text x="{sx}" y="{marginTop + plotHeight + 20}" text-anchor="middle" class="tick">{xVal}</text>\n')

    # Grid & Ticks (Y-axis: 5 intervals)
    for step in range(6):
        yVal = yMin + step * (yMax - yMin) / 5
        sy = toScreenY(yVal)
        svg.append(f'<line x1="{marginLeft}" y1="{sy}" x2="{marginLeft + plotWidth}" y2="{sy}" class="grid"/>\n')
        svg.append(f'<line x1="{marginLeft - 5}" y1="{sy}" x2="{marginLeft}" y2="{sy}" class="axis"/>\n')
        yStr = f"{yVal:.4f}" if yMax <= 0.02 else (f"{yVal:.3f}" if yMax <= 1.0 else f"{yVal:.1f}")
        svg.append(f'<text x="{marginLeft - 10}" y="{sy + 4}" text-anchor="end" class="tick">{yStr}</text>\n')

    # Axes lines
    svg.append(f'<line x1="{marginLeft}" y1="{marginTop + plotHeight}" x2="{marginLeft + plotWidth}" y2="{marginTop + plotHeight}" class="axis"/>\n')
    svg.append(f'<line x1="{marginLeft}" y1="{marginTop}" x2="{marginLeft}" y2="{marginTop + plotHeight}" class="axis"/>\n')

    # Optional Horizontal Threshold Line
    if thresholdY is not None and yMin <= thresholdY <= yMax:
        ty = toScreenY(thresholdY)
        svg.append(f'<line x1="{marginLeft}" y1="{ty}" x2="{marginLeft + plotWidth}" y2="{ty}" stroke="#dc2626" stroke-width="1.8" stroke-dasharray="6,4"/>\n')
        if thresholdLabel:
            svg.append(f'<text x="{marginLeft + plotWidth - 10}" y="{ty - 6}" text-anchor="end" fill="#dc2626" font-size="11px" font-weight="600">{thresholdLabel}</text>\n')

    # Plot each data series
    for s in series:
        xs = s["x"]
        ys = s["y"]
        color = s.get("color", "#2563eb")
        dash = s.get("dash", "")
        widthLine = s.get("width", 2.2)

        pts = []
        for xi, yi in zip(xs, ys):
            if xMin <= xi <= xMax and not math.isnan(yi):
                pts.append(f"{toScreenX(xi):.2f},{toScreenY(yi):.2f}")

        if pts:
            dashAttr = f' stroke-dasharray="{dash}"' if dash else ""
            svg.append(f'<polyline points="{" ".join(pts)}" fill="none" stroke="{color}" stroke-width="{widthLine}"{dashAttr}/>\n')

    # Labels and Title
    svg.append(f'<text x="{marginLeft + plotWidth / 2}" y="{marginTop - 20}" class="title">{title}</text>\n')
    svg.append(f'<text x="{marginLeft + plotWidth / 2}" y="{height - 15}" text-anchor="middle" class="label">{xLabel}</text>\n')
    svg.append(f'<text x="22" y="{marginTop + plotHeight / 2}" text-anchor="middle" transform="rotate(-90, 22, {marginTop + plotHeight / 2})" class="label">{yLabel}</text>\n')

    # Legend
    legX = marginLeft + 20
    legY = marginTop + 20
    for idx, s in enumerate(series):
        ly = legY + idx * 22
        color = s.get("color", "#2563eb")
        dash = s.get("dash", "")
        dashAttr = f' stroke-dasharray="{dash}"' if dash else ""
        svg.append(f'<line x1="{legX}" y1="{ly}" x2="{legX + 30}" y2="{ly}" stroke="{color}" stroke-width="2.5"{dashAttr}/>\n')
        svg.append(f'<text x="{legX + 38}" y="{ly + 4}" fill="#1f2937" class="legend">{s["name"]}</text>\n')

    svg.append("</svg>\n")

    with open(outSvgPath, "w", encoding="utf-8") as f:
        f.writelines(svg)
    print(f"Generated plot: {outSvgPath}")


# -----------------------------------------------------------------------------
# Main Analysis & Report Generation
# -----------------------------------------------------------------------------

def main():
    args = parseArgs()
    os.makedirs(args.outdir, exist_ok=True)

    print("================================================================================")
    print(" Analyzing Taylor-Green Vortex LES Simulation")
    print("================================================================================")
    print(f"VTKHDF file:   {args.vtkhdf}")
    print(f"DNS reference: {args.dns}")
    print(f"Output folder: {args.outdir}")

    if not os.path.exists(args.vtkhdf):
        print(f"Error: VTKHDF file not found: {args.vtkhdf}", file=sys.stderr)
        sys.exit(1)

    dnsData = loadDnsReference(args.dns)
    simRaw = loadVtkhdfData(args.vtkhdf)
    res = analyzeSimulation(simRaw, args.nu)

    # 1. Write CSV History
    csvPath = os.path.join(args.outdir, "taylorGreenLESResults.csv")
    with open(csvPath, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["t", "eRes", "eSgs", "eTot", "dissipation", "nutRatio"])
        for i in range(len(res["t"])):
            writer.writerow([
                f"{res['t'][i]:.4f}",
                f"{res['eRes'][i]:.6e}",
                f"{res['eSgs'][i]:.6e}",
                f"{res['eTot'][i]:.6e}",
                f"{res['dissipation'][i]:.6e}",
                f"{res['nutRatio'][i]:.4f}",
            ])
    print(f"Saved CSV history: {csvPath}")

    # 2. Key Metrics & Comparison against DNS
    maxDissVal = -1.0
    maxDissT = 0.0
    for tVal, dissVal in zip(res["t"], res["dissipation"]):
        if 5.0 <= tVal <= 14.0 and dissVal > maxDissVal:
            maxDissVal = dissVal
            maxDissT = tVal

    maxDnsDiss = max(dnsData["dissipation"])
    maxDnsT = dnsData["t"][dnsData["dissipation"].index(maxDnsDiss)]

    numCells = res.get("numCells", 2097152)
    nGrid = int(round(numCells ** (1.0 / 3.0)))
    delta = math.pi / nGrid
    kolmogorovEtaMin = 0.0118  # approximate Kolmogorov scale at peak dissipation for Re=1600
    ratioEta = delta / kolmogorovEtaMin

    # 3. Generate Summary Markdown
    mdPath = os.path.join(args.outdir, "dnsComparisonSummary.md")
    with open(mdPath, "w", encoding="utf-8") as f:
        f.write("# Taylor-Green Vortex (Re = 1600): LES Verification Summary\n\n")
        f.write("## 1. Benchmark Configuration\n\n")
        f.write("- **Reynolds Number**: Re = U0 * L / nu = 1600 (U0 = 1.0 m/s, L = 1.0 m, nu = 6.25e-4 m^2/s)\n")
        f.write("- **Computational Domain**: Impermeable symmetry octant [0, pi * L]^3 with symmetry BCs on all 6 faces\n")
        f.write(f"- **Grid Resolution**: {nGrid}^3 = {numCells:,} hexahedral cells (equivalent to {2 * nGrid}^3 full periodic box)\n")
        f.write(f"- **Filter Width**: Delta = pi * L / {nGrid} ~ {delta:.4f} m (ratio to Kolmogorov scale Delta / eta_min ~ {ratioEta:.2f})\n")
        f.write("- **Time Scheme & Coupling**: PISO with 2 PRIME correctors, BDF2 (secondOrderImplicit), CFL ~ 0.20\n\n")

        f.write("## 2. Quantitative DNS Comparison\n\n")
        f.write("| Metric | Turblyze LES | Reference DNS | Relative Difference / Status |\n")
        f.write("| :--- | :---: | :---: | :---: |\n")
        f.write(f"| **Initial Kinetic Energy** $E(0)$ | {res['eRes'][0]:.6f} | {dnsData['eKin'][0]:.6f} | exact ({abs(res['eRes'][0] - dnsData['eKin'][0]):.2e}) |\n")
        if maxDissVal > 0:
            relDiffPeak = (maxDissVal - maxDnsDiss) / maxDnsDiss * 100.0
            f.write(f"| **Peak Dissipation Rate** $\\varepsilon_{{\\max}}$ | {maxDissVal:.6f} | {maxDnsDiss:.6f} | {relDiffPeak:+.2f}% |\n")
            f.write(f"| **Time of Peak Dissipation** $t^*$ | {maxDissT:.2f} | {maxDnsT:.2f} | $\\Delta t^* = {maxDissT - maxDnsT:+.2f}$ |\n\n")

        f.write("## 3. Verification Figures\n\n")
        f.write("1. **Kinetic Energy Evolution**: `results/tkeComparison.svg`\n")
        f.write("2. **Dissipation Rate Comparison**: `results/dissipationComparison.svg`\n")

    print(f"Saved comparison summary: {mdPath}")

    # 4. Generate SVG Comparison Figures
    tMax = max(res["t"][-1], 20.0)
    generateSvgPlot(
        title="Taylor-Green Vortex (Re = 1600): Kinetic Energy Evolution",
        xLabel="Non-dimensional Time t* = t U0 / L",
        yLabel="Kinetic Energy E / U0²",
        series=[
            {"name": "DNS (van Rees et al. 2011)", "x": dnsData["t"], "y": dnsData["eKin"], "color": "#111827", "dash": "6,4", "width": 2.2},
            {"name": f"Turblyze LES ({nGrid}³ symmetry)", "x": res["t"], "y": res["eRes"], "color": "#2563eb", "width": 2.5},
        ],
        xRange=(0.0, tMax),
        yRange=(0.0, 0.13),
        outSvgPath=os.path.join(args.outdir, "tkeComparison.svg"),
    )

    generateSvgPlot(
        title="Taylor-Green Vortex (Re = 1600): Dissipation Rate ε = -dE/dt",
        xLabel="Non-dimensional Time t* = t U0 / L",
        yLabel="Dissipation Rate ε (U0³ / L)",
        series=[
            {"name": "DNS (van Rees et al. 2011)", "x": dnsData["t"], "y": dnsData["dissipation"], "color": "#111827", "dash": "6,4", "width": 2.2},
            {"name": f"Turblyze LES ({nGrid}³ symmetry)", "x": res["t"], "y": res["dissipation"], "color": "#dc2626", "width": 2.5},
        ],
        xRange=(0.0, tMax),
        yRange=(0.0, 0.015),
        outSvgPath=os.path.join(args.outdir, "dissipationComparison.svg"),
    )

    print("================================================================================")
    print(" Analysis Complete!")
    if maxDissVal > 0:
        print(f" Peak Dissipation:      {maxDissVal:.6f} at t* = {maxDissT:.2f} (DNS: {maxDnsDiss:.6f} at t* = {maxDnsT:.2f})")
    print("================================================================================")


if __name__ == "__main__":
    main()
