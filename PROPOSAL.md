# EIT Reconstruction Toolkit — Project Proposal

**Author:** Abd'gafar Tunde Tiamiyu  
**Affiliation:** Postdoctoral Researcher, School of Mathematics, Jilin University  
**Repository:** [abdgafartunde/eit-reconstruction-toolkit](https://github.com/abdgafartunde/eit-reconstruction-toolkit)  
**Date:** May 2026

---

## 1. Vision and Motivation

Electrical Impedance Tomography (EIT) is a low-cost, non-invasive imaging modality that reconstructs the internal conductivity distribution of a body from boundary voltage measurements. Despite its clinical and industrial promise, the EIT community currently lacks a **unified, modern, Python-native toolkit** that spans the full pipeline — from forward simulation to classical and learning-based inversion — in a single, well-documented package.

Existing tools each cover only part of the problem:

| Tool | Language | Forward | Classical Inverse | Learning-based |
|------|----------|---------|-------------------|----------------|
| EIDORS | MATLAB | ✓ FEM | ✓ (several) | ✗ |
| pyEIT | Python | ✓ FEM | ✓ (JAC, BP, GREIT) | ✗ |
| FEIT | Python (notebooks) | ✓ FEM | ✓ (basic) | ✗ |
| KTC 2023 entries | Python/MATLAB | partial | partial | partial |

The goal is to build **`eitkit`** — a modular, extensible, purely Python package analogous to what Devito and deepwave provide for the full-waveform inversion community — that covers the complete EIT pipeline and serves as a research-grade reference implementation for the EIT community.

---

## 2. Design Principles

1. **Modularity** — each component (mesh, forward solver, measurement model, inverse solver) is a standalone, swappable module.
2. **Built from scratch** — no wrapping of existing EIT libraries; direct control over every numerical choice.
3. **Composability** — solvers can be combined (e.g., physics-informed + TV regularisation).
4. **Reproducibility** — every solver ships with unit tests and benchmark problems (GREIT phantoms, KTC datasets).
5. **Packaging** — installable via `pip install eitkit`; follows modern Python packaging conventions (`pyproject.toml`, semantic versioning).
6. **Documentation** — API docs (Sphinx + ReadTheDocs) + tutorial notebooks (Jupyter).

---

## 3. Package Architecture

```
eitkit/
├── mesh/                  # Mesh generation and management
│   ├── distmesh2d.py      # 2D mesh generation (circle, arbitrary)
│   ├── mesh_io.py         # Load/export meshes (Gmsh, VTK)
│   └── electrode_placement.py  # Electrode positioning on boundaries
│
├── forward/               # Forward problem: FEM solver
│   ├── gap_model.py       # Gap electrode model (Phase 1)
│   ├── cem.py             # Complete Electrode Model — deferred to Phase 2
│   ├── fem_assembler.py   # FEM stiffness matrix assembly
│   ├── solver.py          # Linear system solver (direct + iterative)
│   ├── jacobian.py        # Jacobian (sensitivity matrix) computation
│   └── simulation.py      # High-level forward simulation API
│
├── protocol/              # Measurement and excitation protocols
│   ├── patterns.py        # Stimulation patterns (adjacent Phase 1; opposite/cross Phase 2)
│   ├── measurement.py     # Voltage measurement configurations
│   └── noise.py           # Noise models (AWGN, SNR-based)
│
├── inverse/               # Inverse solvers
│   ├── classical/
│   │   ├── gauss_newton.py    # Gauss-Newton / Levenberg-Marquardt
│   │   ├── tikhonov.py        # Tikhonov (L2) regularisation
│   │   ├── tv.py              # Total Variation regularisation
│   │   ├── dbar.py            # D-bar method (2D difference EIT; absolute deferred)
│   │   ├── back_projection.py # Back-projection (BP)
│   │   └── greit.py           # GREIT algorithm
│   │
│   ├── data_driven/
│   │   ├── unet.py            # U-Net for image-to-image reconstruction
│   │   ├── cnn_direct.py      # Direct CNN mapping (voltages → image)
│   │   ├── encoder_decoder.py # Encoder-decoder architectures
│   │   └── training.py        # Training loop, losses, data augmentation
│   │
│   ├── physics_informed/
│   │   ├── pinn.py            # Physics-Informed Neural Network (PINN)
│   │   ├── deep_reg.py        # Deep image prior / deep regularisation
│   │   └── neural_operator.py # Fourier/Graph neural operator approach
│   │
│   └── hybrid/
│       ├── learned_regularizer.py  # Learned + classical regularisation
│       ├── unrolled.py             # Algorithm unrolling (ISTA-Net style)
│       └── score_based.py          # Diffusion / score-based prior
│
├── utils/
│   ├── linear_algebra.py    # Sparse solvers, preconditioning
│   ├── metrics.py           # GREIT figures of merit, SSIM, PSNR
│   ├── phantoms.py          # Synthetic phantoms (circular, anomalies)
│   ├── data_loader.py       # KTC 2023 / KIT4 dataset loaders
│   └── visualisation.py     # 2D/3D conductivity plots
│
├── benchmarks/              # Reproducible benchmark suite
│   ├── greit_phantom.py
│   ├── ktc2023.py           # KTC 2023 difference EIT benchmarks
│   └── kit4_synthetic.py    # KIT4-style synthetic benchmarks
│
├── examples/                # Jupyter notebook tutorials
│   ├── 01_forward_2d.ipynb
│   ├── 02_tikhonov_inverse.ipynb
│   ├── 03_tv_regularisation.ipynb
│   ├── 04_dbar_difference.ipynb
│   ├── 05_unet_reconstruction.ipynb
│   ├── 06_pinn_eit.ipynb
│   └── 07_hybrid_unrolled.ipynb
│
├── tests/
│   ├── test_mesh.py
│   ├── test_forward.py
│   ├── test_jacobian.py
│   ├── test_classical_inverse.py
│   └── test_metrics.py
│
├── docs/
│   ├── conf.py
│   └── source/
│
├── pyproject.toml
├── README.md
├── LICENSE
└── CHANGELOG.md
```

---

## 4. Forward Solver — Details

The forward problem in EIT is:

$$\nabla \cdot (\sigma \nabla u) = 0 \quad \text{in } \Omega$$

**Phase 1 uses the gap electrode model**, where each electrode $e_\ell$ is treated as a Dirichlet or Neumann patch with no contact impedance. This gives:

$$\sigma \frac{\partial u}{\partial \nu} = g_\ell \quad \text{on } e_\ell, \quad \frac{\partial u}{\partial \nu} = 0 \quad \text{elsewhere on } \partial\Omega$$

The **Complete Electrode Model (CEM)** — which includes contact impedances $z_\ell$ and is the physically accurate model — is deferred to Phase 2:

$$u + z_\ell \sigma \frac{\partial u}{\partial \nu} = V_\ell \quad \text{on } e_\ell, \qquad \int_{e_\ell} \sigma \frac{\partial u}{\partial \nu} \, ds = I_\ell$$

### Implementation plan

- **Mesh:** `distmesh`-style quality triangular meshes for 2D circular domains; 3D deferred.
- **FEM assembly:** sparse stiffness matrix $K(\sigma)$ assembled element-by-element using NumPy/SciPy.
- **Solver:** sparse LU factorisation via `scipy.sparse.linalg.spsolve`; CG available as fallback.
- **Stimulation:** adjacent pattern only in Phase 1 (electrodes $\ell$ and $\ell{+}1$ driven); opposite and cross patterns added in Phase 2.
- **Measurement type:** difference EIT — voltage differences between a reference state $\sigma_0$ and a perturbed state $\sigma$; absolute EIT deferred.
- **Jacobian:** computed via the adjoint method — cost is two solves per stimulation pattern, independent of the number of mesh elements.
- **Differentiability:** JAX-based autodiff through the assembler is a future enhancement (Phase 3+); not required for the SciPy-based pipeline.

---

## 5. Inverse Solvers — Details

### 5.1 Classical Methods

| Method | Description |
|--------|-------------|
| **Back-Projection (BP)** | Baseline; direct backprojection of voltages onto the mesh |
| **Gauss-Newton / LM** | Iterative linearised inversion; forms the backbone for most classical EIT |
| **Tikhonov (L2)** | $\min \|F(\sigma)-V\|^2 + \alpha\|\sigma - \sigma_0\|^2$; smooth reconstructions |
| **Total Variation (TV)** | $\min \|F(\sigma)-V\|^2 + \alpha\|\nabla\sigma\|_1$; edge-preserving; split-Bregman solver |
| **D-bar** | Global convergent method for 2D EIT; Phase 1 targets difference EIT; absolute EIT deferred |
| **GREIT** | Linear reconstruction matrix designed for clinical chest EIT |

### 5.2 Data-Driven Methods

- **Direct CNN:** maps voltage vector $\to$ conductivity image. Trained on synthetic phantoms.
- **U-Net:** encoder–decoder with skip connections; treats reconstruction as image segmentation.
- Framework: **PyTorch** (primary), with optional JAX backend.
- Data generation: synthetic phantoms via forward solver + noise model.

### 5.3 Physics-Informed Methods

- **PINN:** neural network $u_\theta(x)$ trained to satisfy the EIT PDE + CEM boundary conditions; $\sigma$ inferred as a secondary output.
- **Deep Image Prior:** untrained network whose architecture encodes a spatial prior on $\sigma$.
- **Neural Operator:** FNO or GNO for fast approximate forward/inverse mapping.

### 5.4 Hybrid Methods

- **Learned regulariser:** replace hand-crafted TV with a learned convex regulariser (e.g., input-convex neural network).
- **Algorithm unrolling:** unroll Gauss-Newton iterations into a deep network; train end-to-end.
- **Score-based prior:** use a diffusion model trained on conductivity maps as a prior for MAP estimation.

---

## 6. Development Phases

### Phase 1 — Foundation (Weeks 1–4)
- [ ] Project scaffolding: `pyproject.toml`, CI (GitHub Actions), pre-commit hooks
- [ ] 2D mesh generator for circular domain (distmesh algorithm, triangular elements)
- [ ] FEM assembler for the conductivity equation (homogeneous $\sigma$ test, verify against analytical solution)
- [ ] Gap electrode model (Neumann BCs on electrode patches)
- [ ] Adjacent stimulation pattern + voltage measurement
- [ ] Difference EIT simulation: compute $\delta V = V(\sigma) - V(\sigma_0)$
- [ ] Unit tests: reciprocity theorem, current conservation, mesh quality
- [ ] Example notebook: `01_forward_2d.ipynb`

### Phase 2 — Classical Inverse Solvers + CEM (Weeks 5–10)
- [ ] Jacobian (sensitivity matrix) via adjoint method
- [ ] Back-projection solver
- [ ] Tikhonov (L2) solver (direct: pseudo-inverse; iterative: CG)
- [ ] Gauss-Newton / Levenberg-Marquardt iterative solver
- [ ] Total Variation solver (split-Bregman / ADMM)
- [ ] GREIT reconstruction matrix
- [ ] GREIT figures of merit (amplitude, position error, shape deformation, ringing)
- [ ] Upgrade to Complete Electrode Model (replace gap model)
- [ ] Non-adjacent stimulation patterns (opposite, cross)
- [ ] D-bar method (2D difference EIT)
- [ ] Benchmark suite: KTC 2023 + KIT4 synthetic data
- [ ] Example notebooks: `02`–`04`

### Phase 3 — Data-Driven Inverse Solvers (Weeks 11–18)
- [ ] Synthetic phantom dataset generator (circular anomalies, random inclusions)
- [ ] Direct CNN reconstruction (voltage vector → conductivity image)
- [ ] U-Net reconstruction
- [ ] Training loop with augmentation + noise injection
- [ ] Evaluation on KTC 2023 and KIT4 datasets
- [ ] Optional: JAX-based differentiable FEM assembler (enables end-to-end gradient flow)
- [ ] Example notebook: `05_unet_reconstruction.ipynb`

### Phase 4 — Physics-Informed and Hybrid (Weeks 19–26)
- [ ] PINN for EIT (PyTorch)
- [ ] Deep Image Prior for EIT
- [ ] Learned regulariser (input-convex neural network)
- [ ] Algorithm unrolling (unrolled Gauss-Newton)
- [ ] Score-based / diffusion model prior (research-level)
- [ ] Example notebooks: `06`, `07`

### Phase 5 — Polish, 3D Extension, and Release (Weeks 27–30)
- [ ] 3D mesh generation (tetrahedral, spherical/cylindrical domain)
- [ ] 3D FEM forward solver + gap model
- [ ] Sphinx documentation + ReadTheDocs
- [ ] PyPI packaging and release (`pip install eitkit`)
- [ ] Comprehensive README, logo, badges
- [ ] Preprint / technical report
- [ ] Absolute EIT (D-bar, 2D) as an extension module

---

## 7. Technical Stack

| Component | Choice | Rationale |
|-----------|--------|-----------|
| Language | Python ≥ 3.10 | Ubiquitous in scientific ML |
| FEM / sparse LA | NumPy, SciPy | Standard; no heavy dependencies; `spsolve` for Phase 1–2 |
| Autodiff forward | JAX (Phase 3+, optional) | Deferred; SciPy pipeline is sufficient through Phase 2 |
| Deep learning | PyTorch | Dominant in scientific ML research |
| Mesh generation | Distmesh 2D (from scratch) | Full control; 3D and Gmsh deferred to Phase 5 |
| Visualisation | Matplotlib | 2D conductivity maps; 3D plots deferred |
| Benchmark data | KTC 2023 + KIT4 synthetic | Both use similar 16-electrode circular setups |
| Testing | pytest | Unit and integration tests for FEM correctness |
| CI | GitHub Actions | Automated testing on push |
| Docs | Sphinx + MyST | Jupyter-compatible documentation |
| Packaging | pyproject.toml (hatchling) | Modern Python packaging standard |

---

## 8. Differentiation from Existing Tools

| Feature | eitkit (this project) | pyEIT | EIDORS |
|---------|----------------------|-------|--------|
| Pure Python package | ✓ | ✓ | ✗ (MATLAB) |
| Built from scratch | ✓ | ✓ | ✓ |
| Complete Electrode Model | ✓ | ✓ | ✓ |
| D-bar solver | ✓ | partial | ✓ |
| Total Variation | ✓ | ✗ | ✓ |
| Data-driven (DL) inverse | ✓ | ✗ | ✗ |
| Physics-informed (PINN) | ✓ | ✗ | ✗ |
| Hybrid methods | ✓ | ✗ | ✗ |
| Differentiable forward (JAX) | ✓ | ✗ | ✗ |
| Algorithm unrolling | ✓ | ✗ | ✗ |
| Modern packaging (pip) | ✓ | ✓ | ✗ |
| KTC 2023 data loader | ✓ | ✗ | ✗ |

---

## 9. Design Decisions (Resolved)

| # | Question | Decision |
|---|----------|----------|
| 1 | Stimulation patterns scope for Phase 1 | **Adjacent only**; opposite and cross patterns added in Phase 2 |
| 2 | Electrode model for Phase 1 | **Gap model** (Neumann BCs); Complete Electrode Model added in Phase 2 |
| 3 | EIT mode | **Difference EIT** throughout Phases 1–3; absolute EIT as a Phase 5 extension |
| 4 | Forward solver linear algebra | **SciPy** (`scipy.sparse`, `spsolve`); JAX autodiff is optional and deferred to Phase 3+ |
| 5 | Benchmark datasets | **KTC 2023 + KIT4** — both use 16-electrode circular setups and are directly comparable |
| 6 | Dimensionality | **2D only** through Phase 4; 3D forward solver added in Phase 5 |
| 7 | Package name | **`eitkit`** confirmed |

---

## 10. Repository Structure (Initial)

```
eitkit/                  ← package source
docs/                    ← documentation
examples/                ← Jupyter notebooks
tests/                   ← test suite
benchmarks/              ← reproducible benchmarks
.github/workflows/       ← CI
pyproject.toml
README.md
PROPOSAL.md              ← this document
LICENSE
CHANGELOG.md
```

---

## References

- Somersalo, Cheney & Isaacson (1992) — Complete electrode model
- Vauhkonen et al. (1998) — Tikhonov regularisation for EIT
- Knudsen et al. (2009) — D-bar method
- Adler & Guardo (1996) — GREIT
- Hamilton & Hauptmann (2018) — Deep learning for EIT
- Liu et al. (2018) — pyEIT: A python based framework for EIT (*SoftwareX*)
- KTC 2023 Challenge — Kuopio Tomography Challenge 2023
