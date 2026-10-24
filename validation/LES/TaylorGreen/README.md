<!--
SPDX-FileCopyrightText: 2025-2026 Mohamed Mousa
SPDX-License-Identifier: Apache-2.0
-->

# Validation: 3D Taylor-Green Vortex (Re = 1600)

This directory contains the verification setup and results for Turblyze's Large Eddy Simulation (LES) turbulence model on the **3D Taylor-Green Vortex (TGV)** at $\text{Re} = 1600$.

## 1. Physical Problem & Symmetry Formulation

The Taylor-Green initial condition on $\Omega = [0, \pi L]^3$ is:
$$\begin{aligned}
u(\mathbf{x}, 0) &= U_0 \sin(x/L)\cos(y/L)\cos(z/L) \\
v(\mathbf{x}, 0) &= -U_0 \cos(x/L)\sin(y/L)\cos(z/L) \\
w(\mathbf{x}, 0) &= 0 \\
p(\mathbf{x}, 0) &= p_0 + \frac{\rho U_0^2}{16}\left(\cos(2x/L) + \cos(2y/L)\right)\left(\cos(2z/L) + 2\right)
\end{aligned}$$

Because the six boundary planes ($x, y, z \in \{0, \pi L\}$) are mirror planes where $\mathbf{U} \cdot \mathbf{n} = 0$ and $\partial \mathbf{U}_\tau / \partial n = 0$, the flow is solved on the **impermeable symmetry octant $[0, \pi L]^3$** using `type symmetry;` on all 6 patches (`xMin`, `xMax`, `yMin`, `yMax`, `zMin`, `zMax`).

## 2. Grid Sizing & Resolution

- **Grid Resolution**: $128^3 = 2,097,152$ cells on $[0, \pi L]^3$
- **Equivalence**: Exactly identical filter width $\Delta = \pi L / 128 \approx 0.02454\text{ L}$ to a **$256^3$ grid on the full $[0, 2\pi L]^3$ periodic box**, requiring only $1/8$ the memory and computational runtime.
- **Ratio to Kolmogorov scale**: $\Delta / \eta_{\min} \approx 2.08$ at peak dissipation ($t^* \approx 9$).

## 3. Directory Layout

```text
validation/LES/TaylorGreen/
├── TaylorGreenCase           # Solver configuration file
├── README.md                 # This overview document
├── runGuide.md               # Step-by-step reproduction instructions
├── reference/
│   ├── dnsRe1600.csv         # Tabulated DNS benchmark data (HiOCFD C3.5 / van Rees et al. 2011)
│   └── sources.md            # Literature references (van Rees, HiOCFD, Brachet, DeBonis)
├── scripts/
│   ├── generateMesh.py       # Pure-Python Fluent .msh generator
│   └── analyzeTaylorGreen.py # Post-processor, DNS comparator & SVG plotter
└── results/
    ├── taylorGreenLESResults.csv     # Time series of eRes, eSgs, eTot, dissipation, nutRatio
    ├── dnsComparisonSummary.md       # Quantitative verification report
    ├── tkeComparison.svg             # Kinetic energy vs DNS plot
    └── dissipationComparison.svg     # Dissipation rate vs DNS plot
```

## 4. Key Results

| Metric | Turblyze LES ($128^3$) | Reference DNS | Status |
| :--- | :---: | :---: | :---: |
| **Initial Kinetic Energy** $E(0)$ | $0.125000$ | $0.125000$ | Exact ($< 10^{-9}$ diff) |
| **Peak Dissipation Rate** $\varepsilon_{\max}$ | $0.013129$ | $0.012858$ | $+2.10\%$ error |
| **Peak Dissipation Time** $t^*$ | $8.40$ | $8.97$ | $\Delta t^* = -0.57$ |
