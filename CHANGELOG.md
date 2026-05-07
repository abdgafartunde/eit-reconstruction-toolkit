# Changelog

All notable changes to **eitkit** are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Versioning follows [Semantic Versioning](https://semver.org/).

---

## [Unreleased]

---

## [0.1.0] — 2026-05-07

First public alpha release of **eitkit** — a modular, research-grade Python
package for 2-D Electrical Impedance Tomography (EIT) forward modelling and
classical inverse reconstruction.

### Added

#### Mesh (`eitkit.mesh`)
- `make_circle_mesh(n_electrodes, h0, seed)` — DistMesh2D algorithm: hex-grid
  seed → spring relaxation → Delaunay triangulation of a unit disk.
- `Mesh` dataclass: `nodes`, `elements`, `boundary_nodes`, `areas`.
- `place_electrodes(mesh, n_electrodes)` → `ElectrodeConfig` with
  `node_indices`, `angles`, `arc_width`.

#### Protocol (`eitkit.protocol`)
- `adjacent_pattern(n)` — adjacent drive pairs `(i, i+1 mod L)`.
- `measurement_pairs(n)` — all `L*(L-3)` differential measurement triplets
  `[drive_step, +elec, −elec]`.
- `add_noise(V, snr_db)` — additive Gaussian noise at a specified SNR.

#### Forward solver (`eitkit.forward`)
- `assemble_K(mesh, sigma)` — P1 FEM global stiffness matrix (CSR).
- `apply_neumann_bc` / `pick_ground_node` — gap electrode model.
- `solve_forward(K, f, ground)` — sparse direct solve via `spsolve`.
- `simulate(mesh, ec, sigma, drive_pairs, meas_pairs)` — difference voltages
  δV = V(σ) − V(σ₀), reference σ₀ = 1 S/m.
- `compute_jacobian(mesh, ec, sigma, drive_pairs, meas_pairs)` — adjoint
  Jacobian J ∈ ℝ^(P×E); finite-difference relative error ≈ 7×10⁻⁷.

#### Inverse solvers (`eitkit.inverse`)
- `tikhonov_solve(J, dV, lambda_, solver)` — Tikhonov (L2) regularisation;
  supports `"direct"` (normal equations) and `"lsqr"` iterative solver.
- `choose_lambda(J, dV, lambdas, n_points)` — L-curve corner heuristic via
  maximum discrete curvature.
- `build_gradient_op(mesh)` — element-adjacency gradient operator D (CSR).
- `tv_solve(J, dV, alpha, mesh, rho, max_iter, tol)` — Total Variation (L1)
  regularisation via ADMM.

#### Utilities (`eitkit.utils`)
- `make_phantom(mesh, inclusions, sigma_background)` — five inclusion shapes:
  `circle`, `ellipse`, `rectangle`, `ring`, `triangle`.
- `plot_mesh(mesh, elec_config, title)` — mesh + electrode visualisation.
- `plot_conductivity(mesh, sigma, cmap, show_mesh, ax, vmin, vmax)` —
  2-D filled-triangle conductivity map.
- `plot_voltages(meas_pairs, dV, n_electrodes, highlight_drive)` — voltage
  sinogram heatmap.

#### Examples
- `examples/01_forward_simulation.ipynb` — mesh, phantom, forward solve,
  Jacobian, voltage visualisation.
- `examples/02_inverse_tikhonov.ipynb` — L-curve, Tikhonov reconstruction,
  TV/ADMM reconstruction, side-by-side comparison.

#### Infrastructure
- Hatchling build system (`pyproject.toml`).
- Ruff linter/formatter (line-length 88, target py310).
- pytest test suite: 154 tests across 6 files (all passing).
- GitHub Actions CI workflow.
- MkDocs + Material documentation site.

---

<!-- links -->
[Unreleased]: https://github.com/abdgafartunde/eit-reconstruction-toolkit/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/abdgafartunde/eit-reconstruction-toolkit/releases/tag/v0.1.0
