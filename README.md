# eitkit — EIT Reconstruction Toolkit

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE) 
[![DOI](https://zenodo.org/badge/1231635377.svg)](https://doi.org/10.5281/zenodo.21037312)
[![Tests](https://github.com/abdgafartunde/eit-reconstruction-toolkit/actions/workflows/tests.yml/badge.svg)](https://github.com/abdgafartunde/eit-reconstruction-toolkit/actions/workflows/tests.yml)

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

To reproduce the exact environment the tests and notebook were run with, use the
lock file: `uv sync --extra dev`.

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

## Verification

The scripts in [`scripts/`](scripts/) run from any directory. Each one saves a
figure (to `scripts/` by default, or to `--outdir DIR`) and prints a summary;
add `--show` to open the figure in a window.

| Script | What it checks |
|---|---|
| `verify_mesh.py` | Mesh quality, density, and electrode angles |
| `verify_protocol.py` | Drive and measurement patterns, noise model |
| `verify_forward.py` | Forward solutions and difference voltages |
| `verify_jacobian.py` | Adjoint Jacobian against finite differences |
| `verify_utils.py` | Phantoms and plotting helpers |
| `verify_convergence.py` | FEM electrode voltages against the exact disc solution under mesh refinement |
| `reconstruct_and_plot.py` | End-to-end Tikhonov and TV reconstruction |

For point electrodes on the uniform unit disc, the potential has the closed form
$u(x) = -\frac{I}{\pi}\left(\ln|x - y_s| - \ln|x - y_t|\right)$. The FEM
electrode voltages converge to it at roughly second order:

| $h_0$ | Nodes | Boundary nodes | Relative error |
|---|---|---|---|
| 0.20 | 107 | 32 | 6.6 × 10⁻² |
| 0.10 | 389 | 64 | 8.9 × 10⁻³ |
| 0.05 | 1510 | 128 | 1.5 × 10⁻³ |
| 0.025 | 5907 | 256 | 4.2 × 10⁻⁴ |

---

## Running tests

```bash
pytest          # 173 tests
```

The suite includes the convergence check above and a golden-output test that
reruns the demo notebook's pipeline and compares it with
[`tests/golden/demo_notebook.json`](tests/golden/demo_notebook.json).
GitHub Actions runs the tests on Python 3.10 to 3.13.

---

## Citing eitkit

If you use eitkit, please cite the Zenodo archive
([10.5281/zenodo.21037312](https://doi.org/10.5281/zenodo.21037312)); the
citation metadata is in [`CITATION.cff`](CITATION.cff).

