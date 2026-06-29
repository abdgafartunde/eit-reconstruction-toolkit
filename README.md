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
from eitkit.inverse import tikhonov_solve, choose_lambda

# 1. Build mesh and place 16 electrodes
mesh = make_circle_mesh(n_electrodes=16, h0=0.07, seed=42)
ec   = place_electrodes(mesh, n_electrodes=16)

# 2. Define a two-inclusion phantom
sigma = make_phantom(mesh, inclusions=[
    {"shape": "circle", "cx": -0.4, "cy": 0.0, "r": 0.25, "sigma": 3.0},
    {"shape": "circle", "cx":  0.4, "cy": 0.0, "r": 0.25, "sigma": 0.3},
], sigma_background=1.0)

# 3. Forward simulation → noisy difference voltages
drive_pairs = adjacent_pattern(16)
meas_pairs  = measurement_pairs(16)
dV = add_noise(simulate(mesh, ec, sigma, drive_pairs, meas_pairs), snr_db=40)

# 4. Jacobian + L-curve + Tikhonov reconstruction
sigma_ref = np.ones(len(mesh.elements))
J = compute_jacobian(mesh, ec, sigma_ref, drive_pairs, meas_pairs)
lambda_opt, _, _ = choose_lambda(J, dV)
dsigma = tikhonov_solve(J, dV, lambda_=lambda_opt)

plot_conductivity(mesh, sigma_ref + dsigma, title="Tikhonov reconstruction")
```

See [`examples/02_inverse_tikhonov.ipynb`](examples/02_inverse_tikhonov.ipynb)
for a full walkthrough including TV/ADMM reconstruction and a side-by-side comparison.

---

## Examples

| Notebook | Description |
|---|---|
| [`01_forward_simulation.ipynb`](examples/01_forward_simulation.ipynb) | Mesh, phantom, forward solve, Jacobian, voltage map |
| [`02_inverse_tikhonov.ipynb`](examples/02_inverse_tikhonov.ipynb) | L-curve, Tikhonov, TV/ADMM, comparison |

---

## Running tests

```bash
pytest tests/ -v          # 154 tests
```

---

## License

MIT — see [LICENSE](LICENSE).

