# eitkit — EIT Reconstruction Toolkit

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A modular, research-grade Python package for **2-D Electrical Impedance Tomography (EIT)**:
forward simulation, Jacobian computation, and classical inverse reconstruction.

---

## Features

| Module | What it does |
|---|---|
| `eitkit.mesh` | DistMesh2D circle mesh, electrode placement |
| `eitkit.protocol` | Adjacent drive patterns, measurement pairs, noise |
| `eitkit.forward` | P1 FEM assembler, gap electrode model, adjoint Jacobian |
| `eitkit.inverse` | Tikhonov (L2) + TV/ADMM (L1) reconstruction, L-curve |
| `eitkit.utils` | Phantom shapes, mesh/conductivity/voltage visualisation |

---

## Installation

Install from source in editable mode:

```bash
git clone https://github.com/abdgafartunde/eit-reconstruction-toolkit
cd eit-reconstruction-toolkit
pip install -e ".[dev]"
```

---

## Quick start

```python
import numpy as np
from eitkit.mesh import make_circle_mesh, place_electrodes
from eitkit.protocol import adjacent_pattern, measurement_pairs, add_noise
from eitkit.forward import simulate, compute_jacobian
from eitkit.utils import make_phantom, plot_conductivity
from eitkit.inverse import tikhonov_solve, tv_solve

# 1. Build mesh and place 16 electrodes
mesh = make_circle_mesh(n_electrodes=16, h0=0.08, seed=42)
ec   = place_electrodes(mesh, n_electrodes=16)

# 2. Define a two-inclusion phantom (moderate contrasts for linear regime)
sigma = make_phantom(mesh, inclusions=[
    {"shape": "circle",  "cx":  0.4, "cy":  0.3, "r": 0.22, "sigma": 1.6},
    {"shape": "ellipse", "cx": -0.35, "cy": -0.25, "a": 0.25, "b": 0.14,
     "theta": 0.6, "sigma": 0.5},
], sigma_background=1.0)

# 3. Forward simulation with noise
drive_pairs = adjacent_pattern(16)
meas_pairs  = measurement_pairs(16)
dV = add_noise(simulate(mesh, ec, sigma, drive_pairs, meas_pairs),
               snr_db=45.0, rng=42)

# 4. Jacobian + Tikhonov reconstruction
sigma_ref = np.ones(mesh.n_elements)
J = compute_jacobian(mesh, ec, sigma_ref, drive_pairs, meas_pairs)
dsigma_tik = tikhonov_solve(J, dV, lambda_=1e-7)
sigma_tik  = sigma_ref + dsigma_tik

# 5. TV / ADMM reconstruction (edge-preserving)
dsigma_tv = tv_solve(J, dV, alpha=5e-8, mesh=mesh)
sigma_tv  = sigma_ref + dsigma_tv

plot_conductivity(mesh, sigma_tik, title="Tikhonov reconstruction")
```

See [`examples/01_reconstruction_demo.ipynb`](examples/01_reconstruction_demo.ipynb)
for a complete interactive walkthrough with configurable hyperparameters, preset
phantoms, L-curve analysis, and side-by-side comparisons.

---

## Documentation

- **[THEORY.md](THEORY.md)** — Full mathematical formulation (forward problem,
  FEM discretisation, adjoint Jacobian, Tikhonov, TV/ADMM, DistMesh2D).
- **[examples/01_reconstruction_demo.ipynb](examples/01_reconstruction_demo.ipynb)** —
  Interactive notebook with configurable hyperparameters.
- **[scripts/reconstruct_and_plot.py](scripts/reconstruct_and_plot.py)** —
  Standalone script that generates publication-quality figures.

---

## Running tests

```bash
pytest tests/ -v          # 154 tests
```

