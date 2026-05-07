# eitkit

**eitkit** is a modular, research-grade Python package for **2-D Electrical
Impedance Tomography (EIT)** — covering forward simulation, sensitivity
analysis, and classical inverse reconstruction.

## Highlights

- **P1 finite-element forward solver** with gap electrode model
- **Adjoint Jacobian** (L + L solves; FD relative error ≈ 7×10⁻⁷)
- **Tikhonov (L2)** reconstruction with L-curve parameter selection
- **Total Variation (TV / L1)** reconstruction via ADMM
- Five phantom inclusion shapes: circle, ellipse, rectangle, ring, triangle
- 154-test pytest suite

## Installation

```bash
pip install eitkit
```

## Quick example

```python
from eitkit.mesh import make_circle_mesh, place_electrodes
from eitkit.protocol import adjacent_pattern, measurement_pairs
from eitkit.forward import simulate, compute_jacobian
from eitkit.utils import make_phantom
from eitkit.inverse import tikhonov_solve, choose_lambda
import numpy as np

mesh = make_circle_mesh(n_electrodes=16, h0=0.07, seed=42)
ec   = place_electrodes(mesh, 16)

sigma = make_phantom(mesh, inclusions=[
    {"shape": "circle", "cx": -0.4, "cy": 0.0, "r": 0.25, "sigma": 3.0},
    {"shape": "circle", "cx":  0.4, "cy": 0.0, "r": 0.25, "sigma": 0.3},
])
drive_pairs = adjacent_pattern(16)
meas_pairs  = measurement_pairs(16)
dV          = simulate(mesh, ec, sigma, drive_pairs, meas_pairs)

sigma_ref   = np.ones(len(mesh.elements))
J           = compute_jacobian(mesh, ec, sigma_ref, drive_pairs, meas_pairs)
lam, _, _   = choose_lambda(J, dV)
dsigma      = tikhonov_solve(J, dV, lambda_=lam)
```

See [Getting Started](getting_started.md) for a step-by-step walkthrough.
