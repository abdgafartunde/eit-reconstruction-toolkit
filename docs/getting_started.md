# Getting Started

## Installation

Install from PyPI:

```bash
pip install eitkit
```

Install from source with development extras:

```bash
git clone https://github.com/abdgafartunde/eit-reconstruction-toolkit
cd eit-reconstruction-toolkit
pip install -e ".[dev]"
```

Install documentation dependencies:

```bash
pip install -e ".[docs]"
mkdocs serve        # live preview at http://127.0.0.1:8000
```

## Minimal workflow

A complete difference-EIT reconstruction in seven steps:

### 1. Build mesh and electrodes

```python
from eitkit.mesh import make_circle_mesh, place_electrodes

mesh = make_circle_mesh(n_electrodes=16, h0=0.07, seed=42)
ec   = place_electrodes(mesh, n_electrodes=16)
print(f"{len(mesh.elements)} elements, {len(mesh.nodes)} nodes")
```

### 2. Create a phantom

```python
from eitkit.utils import make_phantom

sigma = make_phantom(mesh, inclusions=[
    {"shape": "circle", "cx": -0.4, "cy": 0.0, "r": 0.25, "sigma": 3.0},
    {"shape": "circle", "cx":  0.4, "cy": 0.0, "r": 0.25, "sigma": 0.3},
], sigma_background=1.0)
```

### 3. Define the measurement protocol

```python
from eitkit.protocol import adjacent_pattern, measurement_pairs

drive_pairs = adjacent_pattern(16)   # shape (16, 2)
meas_pairs  = measurement_pairs(16)  # shape (208, 3)
```

### 4. Run the forward solver

```python
import numpy as np
from eitkit.forward import simulate
from eitkit.protocol import add_noise

dV = add_noise(simulate(mesh, ec, sigma, drive_pairs, meas_pairs), snr_db=40)
```

### 5. Compute the Jacobian

```python
from eitkit.forward import compute_jacobian

sigma_ref = np.ones(len(mesh.elements))
J = compute_jacobian(mesh, ec, sigma_ref, drive_pairs, meas_pairs)
# J.shape == (208, n_elements)
```

### 6. Choose regularisation parameter and reconstruct

```python
from eitkit.inverse import tikhonov_solve, choose_lambda

lambda_opt, _, _ = choose_lambda(J, dV)
dsigma = tikhonov_solve(J, dV, lambda_=lambda_opt)
```

Or use Total Variation for sharper boundaries:

```python
from eitkit.inverse import tv_solve, build_gradient_op

D      = build_gradient_op(mesh)
rho    = float(np.linalg.norm(J, 'fro')**2) / D.nnz
dsigma = tv_solve(J, dV, alpha=0.15 * rho, mesh=mesh, rho=rho)
```

### 7. Visualise

```python
from eitkit.utils import plot_conductivity
import matplotlib.pyplot as plt

plot_conductivity(mesh, sigma_ref + dsigma, title="Reconstructed σ")
plt.show()
```

## Running the test suite

```bash
pytest tests/ -v   # 154 tests
```
