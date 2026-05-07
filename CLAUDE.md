# CLAUDE.md — eitkit project context

This file gives AI assistants (Claude, GitHub Copilot, etc.) the context needed to
work effectively in this repository.

---

## Working Principles

- **List subcomponents first, then implement one at a time.**
  Before starting any component, enumerate all subcomponents explicitly.
  Implement and verify each subcomponent before moving to the next.

---

## Project

**`eitkit`** — a modular, research-grade Python package for 2-D
**Electrical Impedance Tomography (EIT)** forward modelling.

- Repository: `abdgafartunde/eit-reconstruction-toolkit`
- Package name: `eitkit`  (installed with `pip install -e ".[dev]"`)
- Python: ≥ 3.10
- Build system: Hatchling (`pyproject.toml`)
- Linter/formatter: Ruff (line-length 88, target py310)
- Test runner: pytest (96 tests, all passing)

---

## Package layout

```
eitkit/
├── __init__.py            # importlib.metadata version; fallback "0.1.0.dev0"
├── mesh/
│   ├── mesh.py            # Mesh dataclass (nodes, elements, boundary_nodes, areas)
│   ├── distmesh2d.py      # DistMesh algorithm: hex-grid seed → spring relaxation → Delaunay
│   ├── electrode_placement.py  # ElectrodeConfig dataclass; place_electrodes()
│   └── __init__.py        # re-exports: make_circle_mesh, Mesh, ElectrodeConfig, place_electrodes
├── protocol/
│   ├── patterns.py        # adjacent_pattern(n) → NDArray[(L,2)]
│   ├── measurement.py     # measurement_pairs(n) → NDArray[(L*(L-3),3)]  cols: [drive,+,−]
│   ├── noise.py           # add_noise(V, snr_db)
│   └── __init__.py
├── forward/
│   ├── fem_assembler.py   # assemble_K(mesh, sigma) → sparse CSR
│   ├── gap_model.py       # Neumann BCs; pick_ground_node(); apply_neumann_bc()
│   ├── solver.py          # solve_forward(K, f, ground) using scipy.sparse.linalg.spsolve
│   ├── simulation.py      # simulate(mesh, ec, sigma, drive_pairs, meas_pairs) → dV (P,)
│   ├── jacobian.py        # compute_jacobian(mesh, ec, sigma, drive_pairs, meas_pairs) → J (P,E)
│   └── __init__.py
└── utils/
    ├── phantoms.py        # make_phantom(mesh, inclusions, sigma_background=1.0)
    ├── visualisation.py   # plot_mesh, plot_conductivity, plot_voltages (2-D heatmap)
    └── __init__.py
inverse/
├── __init__.py            # re-exports: tikhonov_solve, choose_lambda
└── classical/
    ├── tikhonov.py        # tikhonov_solve(J, dV, lambda_, solver) → δσ; choose_lambda(J, dV) → (λ*, residuals, sol_norms)
    ├── tv.py              # build_gradient_op(mesh) → D; tv_solve(J, dV, alpha, mesh) → δσ
    └── __init__.py        # re-exports tikhonov_solve, choose_lambda, build_gradient_op, tv_solve
```

---

## Key design decisions

| Decision | Detail |
|----------|--------|
| **Difference EIT only** | `simulate` returns `δV = V(σ) − V(σ₀)`; reference `σ₀=1` S/m |
| **Adjacent stimulation** | `adjacent_pattern` produces `(i, i+1 mod L)` drive pairs |
| **Gap electrode model** | Neumann BC on single boundary node per electrode; no contact impedance |
| **P1 FEM** | Piecewise-linear basis; element stiffness `K^e = σ_e/(4A_e) * B^T B` |
| **Adjoint Jacobian** | L forward + L adjoint solves; finite-difference relative error ≈ 7e-7 |
| **2-D only** | Unit disk domain; no 3-D mesh support yet |

---

## Important function signatures

```python
# Mesh
mesh = make_circle_mesh(n_electrodes=16, h0=0.07, seed=42)  # → Mesh
ec   = place_electrodes(mesh, n_electrodes=16)               # → ElectrodeConfig

# Protocol
drive_pairs = adjacent_pattern(n_electrodes)    # → NDArray[(L, 2)]
meas_pairs  = measurement_pairs(n_electrodes)   # → NDArray[(L*(L-3), 3)]  [drive_step, +, −]

# Phantom
sigma = make_phantom(mesh, inclusions=[
    {"shape": "circle",    "cx": 0.0, "cy": 0.0, "r": 0.3, "sigma": 3.0},
    {"shape": "ellipse",   "cx": 0.2, "cy": 0.0, "a": 0.3, "b": 0.15, "theta": 30.0, "sigma": 0.5},
    {"shape": "rectangle", "cx": 0.0, "cy": 0.2, "w": 0.4, "h": 0.2, "theta": 0.0,   "sigma": 4.0},
    {"shape": "ring",      "cx": 0.0, "cy": 0.0, "r_inner": 0.2, "r_outer": 0.4, "sigma": 2.0},
    {"shape": "triangle",  "cx": 0.0, "cy": 0.0, "side": 0.4, "theta": 0.0, "sigma": 3.0},
], sigma_background=1.0)                        # → NDArray[(E,)]

# Inverse (Tikhonov)
delta_sigma = tikhonov_solve(J, dV, lambda_=1e-3)                        # → NDArray[(E,)]
delta_sigma = tikhonov_solve(J, dV, lambda_=1e-3, solver="lsqr")        # iterative solver
lambda_opt, residuals, sol_norms = choose_lambda(J, dV, n_points=50)    # L-curve heuristic

# Inverse (TV)
D             = build_gradient_op(mesh)                                  # → sparse (F, E)
delta_sigma   = tv_solve(J, dV, alpha=1e-2, mesh=mesh)                   # → NDArray[(E,)]
delta_sigma   = tv_solve(J, dV, alpha=1e-2, mesh=mesh, rho=1.0,
                         max_iter=200, tol=1e-4)                          # full signature

# Visualisation
fig, ax = plot_mesh(mesh, elec_config=ec, title="...")
fig, ax = plot_conductivity(mesh, sigma, cmap="hot_r", show_mesh=False)
fig, ax = plot_voltages(meas_pairs, dV, n_electrodes=16, highlight_drive=4)
```

---

## ElectrodeConfig attributes

```python
ec.node_indices   # NDArray[int32], shape (L,) — index into mesh.nodes per electrode
ec.angles         # NDArray[float64], shape (L,) — CCW angle from +x axis (radians)
ec.arc_width      # float — Neumann patch half-width in radians (= π/L)
ec.n_electrodes   # int
```

---

## Common pitfalls

- `simulate` and `compute_jacobian` take `sigma` as the **3rd** positional argument,
  **before** `drive_pairs` and `meas_pairs`.
- `measurement_pairs` returns **all** `L*(L-3)` rows as `[drive_step, +elec, −elec]`
  (not grouped by drive step).
- `plot_voltages` expects `len(dV) == n_electrodes * (n_electrodes - 3)` exactly;
  raises `ValueError` otherwise.
- `ElectrodeConfig` uses `.node_indices` (not `.electrode_nodes`).
- `mesh.py` module docstring must be a raw string (`r"""..."""`) due to `\partial` LaTeX.

---

## Development commands

```powershell
# Install in editable mode with dev extras
pip install -e ".[dev]"

# Run all tests
pytest tests/ -v

# Lint + format
ruff check .
ruff format .

# Run verification scripts
python scripts/verify_utils.py
```

---

## Test suite (154 tests)

| File | Tests | Scope |
|------|-------|-------|
| `tests/test_mesh.py` | 25 | Mesh generation, electrode placement |
| `tests/test_protocol.py` | 24 | adjacent_pattern, measurement_pairs, add_noise |
| `tests/test_forward.py` | 19 | FEM assembly, simulate, Jacobian (adjoint vs FD) |
| `tests/test_utils.py` | 28 | 5 phantom shapes, plot_mesh, plot_conductivity, plot_voltages |
| `tests/test_inverse.py` | 32 | tikhonov_solve (direct+lsqr), choose_lambda, input validation, integration |
| `tests/test_tv.py` | 26 | build_gradient_op, tv_solve (ADMM), edge-preservation, input validation, integration |

Session-scoped fixtures in `tests/conftest.py`:
- `circle_mesh` → `make_circle_mesh(16, h0=0.12)`
- `electrodes`  → `place_electrodes(circle_mesh, 16)`

---

## Roadmap (next components)

- ✅ **Component 9** — Tikhonov inverse solver (`eitkit/inverse/classical/tikhonov.py`)
- ✅ **Component 10** — TV regularisation (`eitkit/inverse/classical/tv.py`)
- **Component 11** — `examples/02_inverse_tikhonov.ipynb`
- **Component 12** — PyPI packaging + `docs/` (Sphinx or MkDocs)
