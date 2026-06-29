# eitkit — Theory & Implementation

> A modular Python toolkit for 2-D Electrical Impedance Tomography: forward simulation, Jacobian computation, and classical regularisation-based reconstruction.

---

## 1. Overview

**Electrical Impedance Tomography (EIT)** is a non-invasive imaging modality that reconstructs the interior conductivity distribution $\sigma(x)$ of a domain $\Omega \subset \mathbb{R}^2$ from boundary voltage measurements. The mathematical problem, posed by Calderón (1980), is severely ill-posed: small perturbations in boundary data can induce arbitrarily large errors in any reconstruction (Mandache, 2001).

`eitkit` implements the complete **difference EIT** pipeline in two dimensions:

| Stage | What it does |
|-------|-------------|
| Mesh generation | DistMesh2D triangular mesh on the unit disc |
| Forward problem | P1-FEM solution of the conductivity equation with gap electrode model |
| Jacobian | Sensitivity matrix via the adjoint method |
| Inverse problem | Tikhonov (L²) and Total Variation (L¹/ADMM) regularisation |

The toolkit uses only NumPy, SciPy, and Matplotlib — no external mesh libraries or deep learning frameworks are required for core functionality.

---

## 2. Mathematical Formulation

### 2.1 Forward Problem

Let $\Omega \subset \mathbb{R}^2$ be a bounded domain with Lipschitz boundary $\partial\Omega$. Given a conductivity distribution $\sigma \in L^\infty(\Omega)$ with $\sigma(x) \ge c > 0$, the electric potential $u \in H^1(\Omega)$ satisfies the elliptic boundary value problem

$$
\begin{aligned}
-\nabla \cdot (\sigma \nabla u) &= 0 \quad &&\text{in } \Omega, \\[4pt]
\sigma \frac{\partial u}{\partial n} &= j \quad &&\text{on } \partial\Omega,
\end{aligned}
$$

where $j$ is the applied current density satisfying the conservation condition $\int_{\partial\Omega} j \, ds = 0$. To obtain a unique solution, a ground condition $u(x_0) = 0$ is imposed at one interior node.

### 2.2 Gap Electrode Model

In Phase 1, each of the $L$ electrodes is modelled as a single boundary node (the **gap model**). A stimulation pair $(e^+, e^-)$ injects current $+I$ at electrode $e^+$ and extracts $-I$ at $e^-$. The Neumann load vector $f \in \mathbb{R}^N$ has entries

$$
f_i = \begin{cases}
+I & \text{if node } i \text{ is electrode } e^+, \\[2pt]
-I & \text{if node } i \text{ is electrode } e^-, \\[2pt]
0  & \text{otherwise}.
\end{cases}
$$

### 2.3 Finite Element Discretisation

The domain is triangulated with $M$ linear (P1) elements on $N$ nodes. Writing $u = \sum_{i=1}^N u_i \varphi_i$ where $\varphi_i$ are the P1 hat functions, the weak form yields the linear system

$$
K(\sigma) \, u = f,
$$

where the global stiffness matrix is assembled element-wise:

$$
K = \sum_{e=1}^{M} K^e, \qquad
K^e_{ij} = \frac{\sigma_e}{4 A_e} \bigl( b_i b_j + c_i c_j \bigr), \quad i,j \in \{0,1,2\}.
$$

The element gradient coefficients are defined on the reference triangle with nodes $(x_0,y_0), (x_1,y_1), (x_2,y_2)$ (indices mod 3):

$$
b_i = y_{i+1} - y_{i+2}, \qquad
c_i = x_{i+2} - x_{i+1},
$$

and $A_e$ is the (positive) element area. The Dirichlet ground condition is enforced by zeroing the row and column of the chosen ground node $g$ and setting $K_{gg} = 1$, $f_g = 0$.

### 2.4 Measurement Protocol

**Adjacent drive / adjacent measurement** is the standard protocol. For $L$ electrodes:

- **Drive pairs** ($L$ steps): step $k$ sources current at electrode $k$ and sinks at electrode $(k+1) \bmod L$.
- **Measurement pairs** ($L-3$ per step): voltages are measured across adjacent electrode pairs $(k+2, k+3), (k+3, k+4), \ldots, (k+L-2, k+L-1)$ (all indices modulo $L$).

The total number of measurements per frame is $P = L(L-3)$. For $L = 16$, this gives $P = 208$.

### 2.5 Difference EIT

In practice, absolute voltages are difficult to calibrate. **Difference EIT** measures voltage changes relative to a reference state $\sigma_0$ (typically a uniform background):

$$
\delta V = V(\sigma) - V(\sigma_0).
$$

Linearising around $\sigma_0$ gives the Born approximation

$$
\delta V \approx J(\sigma_0) \, \delta\sigma,
$$

where $J \in \mathbb{R}^{P \times M}$ is the **Jacobian** (sensitivity matrix) and $\delta\sigma = \sigma - \sigma_0 \in \mathbb{R}^M$ is the conductivity perturbation.

### 2.6 Jacobian via the Adjoint Method

The entry $J_{ie} = \partial V_i / \partial \sigma_e$ is computed efficiently using the **adjoint method**. For measurement $i$ associated with drive step $k$ and electrode pair $(e^+, e^-)$,

$$
J_{ie} = -\int_{\Omega_e} \nabla u_k \cdot \nabla \phi_i \, d\Omega,
$$

where $u_k$ is the forward potential for drive step $k$ and $\phi_i$ is the **adjoint potential**: the solution with a unit current injected at $e^+$ and extracted at $e^-$.

For a P1 triangle, the element integral evaluates to

$$
J_{ie} = -\frac{1}{4A_e} \Bigl[ \bigl( \mathbf{b}_e \cdot \mathbf{u}_k^e \bigr) \bigl( \mathbf{b}_e \cdot \boldsymbol{\phi}_i^e \bigr) + \bigl( \mathbf{c}_e \cdot \mathbf{u}_k^e \bigr) \bigl( \mathbf{c}_e \cdot \boldsymbol{\phi}_i^e \bigr) \Bigr],
$$

where $\mathbf{b}_e, \mathbf{c}_e \in \mathbb{R}^3$ are the element gradient coefficient vectors, and $\mathbf{u}_k^e, \boldsymbol{\phi}_i^e \in \mathbb{R}^3$ are the nodal potential values at the three corners of element $e$.

**Adjoint reuse:** For the adjacent protocol, each distinct measurement electrode pair $(e^+, e^-)$ appears in exactly one drive step, so we need $L$ forward solves and at most $L$ adjoint solves. The factorised stiffness matrix is reused for all solves via `scipy.sparse.linalg.splu`.

---

## 3. Inverse Problem & Regularisation

### 3.1 Tikhonov Regularisation (L²)

The classical Tikhonov solution minimises

$$
\delta\hat\sigma = \arg\min_{\delta\sigma} \left\{ \tfrac{1}{2} \| J \,\delta\sigma - \delta V \|_2^2 + \lambda \| \delta\sigma \|_2^2 \right\},
$$

with closed-form solution via the normal equations:

$$
\delta\hat\sigma = (J^T J + \lambda I)^{-1} J^T \delta V.
$$

Two solvers are provided:
- **`direct`**: `numpy.linalg.solve` on the $(M, M)$ normal-equations system — fast for $M \lesssim 10^4$.
- **`lsqr`**: iterative `scipy.sparse.linalg.lsqr` on the augmented system $[J; \sqrt{\lambda} I] \, \delta\sigma = [\delta V; 0]$.

### 3.2 L-Curve Heuristic

The regularisation parameter $\lambda$ is chosen automatically via the L-curve method (Hansen, 1992). The L-curve is the log-log plot of residual norm vs. solution norm:

$$
\mathcal{C}(\lambda) = \bigl( \log\| J \,\delta\hat\sigma(\lambda) - \delta V \|,\; \log\| \delta\hat\sigma(\lambda) \| \bigr).
$$

The optimal $\lambda$ corresponds to the point of maximum curvature (the "corner"), balancing data fit against solution smoothness.

### 3.3 Total Variation Regularisation (L¹ / ADMM)

TV regularisation promotes **piecewise-constant** (edge-preserving) reconstructions:

$$
\delta\hat\sigma = \arg\min_{\delta\sigma} \left\{ \tfrac{1}{2} \| J \,\delta\sigma - \delta V \|_2^2 + \alpha \| D \,\delta\sigma \|_1 \right\},
$$

where $D \in \mathbb{R}^{F \times M}$ is a sparse element-to-element finite difference operator. Each row of $D$ corresponds to a shared interior edge between two elements, with entries $+1$ and $-1$.

The problem is solved via the **Alternating Direction Method of Multipliers (ADMM)** with splitting $z = D\,\delta\sigma$:

- **$\sigma$-update** (quadratic, direct factorisation):
  $$
  \delta\sigma^{k+1} = (J^T J + \rho D^T D)^{-1} \bigl( J^T \delta V + \rho D^T (z^k - u^k) \bigr).
  $$

- **$z$-update** (element-wise soft-thresholding):
  $$
  z^{k+1} = \mathcal{S}_{\alpha/\rho}\bigl( D\,\delta\sigma^{k+1} + u^k \bigr), \qquad
  \mathcal{S}_\kappa(x) = \operatorname{sgn}(x) \max(|x| - \kappa, 0).
  $$

- **$u$-update** (dual ascent):
  $$
  u^{k+1} = u^k + D\,\delta\sigma^{k+1} - z^{k+1}.
  $$

The penalty parameter $\rho$ is auto-scaled as $\rho = \operatorname{tr}(J^T J) \,/\, \operatorname{tr}(D^T D)$ so that the data-fit and regularisation terms in the augmented Lagrangian have commensurate magnitude. Convergence is declared when the relative primal residual falls below a tolerance.

---

## 4. Mesh Generation — DistMesh2D

The triangular mesh is generated by a node-relaxation algorithm inspired by Persson & Strang (2004):

1. **Seeding**: a regular hex-grid point cloud fills the unit disc with target edge length $h_0$.
2. **Electrode nodes**: $L$ nodes are placed exactly on the boundary at uniform angular spacing and are held fixed during relaxation.
3. **Relaxation**: free (interior) nodes are repeatedly moved by spring forces (repulsion when edges are shorter than $h_0$, no attraction when longer). After each displacement step, boundary nodes are re-projected onto the circle. Delaunay re-triangulation is performed each iteration.
4. **Filtering**: triangles whose centroid lies outside the domain are removed.
5. **CCW enforcement**: all triangles are wound counter-clockwise (positive signed area).

The result is a quality unstructured triangular mesh with guaranteed CCW winding, sorted boundary nodes, and no degenerate elements.

---

## 5. Synthetic Phantoms

Piecewise-constant conductivity phantoms are built on the mesh from a uniform background plus geometric inclusions. Supported shapes:

| Shape | Parameters |
|-------|-----------|
| Circle | $(c_x, c_y)$, radius $r$ |
| Ellipse | $(c_x, c_y)$, semi-axes $(a, b)$, rotation $\theta$ |
| Rectangle | $(c_x, c_y)$, half-widths $(w, h)$, rotation $\theta$ |
| Ring | $(c_x, c_y)$, inner radius $r_{\text{in}}$, outer radius $r_{\text{out}}$ |
| Triangle | $(c_x, c_y)$, side length, rotation $\theta$ |

An element is inside an inclusion if its centroid lies within the geometric region. Later inclusions overwrite earlier ones where they overlap.

---

## 6. Implementation Details

### 6.1 Stiffness Matrix Assembly

The assembly is vectorised over elements using NumPy advanced indexing. Element gradient coefficients $b, c$ are pre-computed in a single pass, and the 9 local-global stiffness contributions per element are accumulated via `scipy.sparse.coo_array`. The result is a `csr_array` for efficient sparse linear algebra.

### 6.2 Linear Solves

Once assembled, the Dirichlet-modified stiffness matrix is factorised once via `scipy.sparse.linalg.splu` (supernodal LU). All $L$ drive steps and $Q$ adjoint solves then use cheap triangular back-substitution (`lu.solve`), giving $\mathcal{O}(L \cdot N^2)$ total work rather than $\mathcal{O}(L \cdot N^3)$ for repeated factorisation.

### 6.3 Scaling Considerations

- **Linear regime**: the Born approximation $J\,\delta\sigma \approx \delta V$ requires moderate conductivity contrasts ($|\delta\sigma|/\sigma_0 \lesssim 0.5$). Large contrasts need nonlinear Gauss-Newton iteration (planned for a future release).
- **Mesh density**: with $h_0 = 0.08$, a typical mesh has $M \approx 1100$ elements and $N \approx 570$ nodes. The forward problem is well-conditioned; the inverse problem has $P = 208$ measurements for $M = 1116$ unknowns (5.4× under-determined).

---

## 7. Package Structure

```
eitkit/
├── __init__.py
├── mesh/
│   ├── mesh.py               # Mesh dataclass (immutable, validated)
│   ├── distmesh2d.py         # Circle mesh generator (DistMesh2D)
│   └── electrode_placement.py # Gap-model electrode placement
├── forward/
│   ├── fem_assembler.py      # Vectorised P1 FEM stiffness assembly
│   ├── gap_model.py          # Neumann load vector (point electrodes)
│   ├── solver.py             # Dirichlet BC + sparse LU solver
│   ├── simulation.py         # High-level forward simulation API
│   └── jacobian.py           # Adjoint-method Jacobian computation
├── protocol/
│   ├── patterns.py           # Adjacent-drive stimulation schedule
│   ├── measurement.py        # Voltage measurement pair schedule
│   └── noise.py              # AWGN noise model (SNR-calibrated)
├── inverse/
│   └── classical/
│       ├── tikhonov.py       # Tikhonov (L²) + L-curve heuristic
│       └── tv.py             # TV-ADMM (L¹) + gradient operator
└── utils/
    ├── phantoms.py           # 5 inclusion shapes for test phantoms
    └── visualisation.py      # Mesh/conductivity/voltage plotting
```

---

## 8. References

- **Calderón, A.P.** (1980). On an inverse boundary value problem. *Seminar on Numerical Analysis and its Applications to Continuum Physics*, Rio de Janeiro.
- **Persson, P.-O. & Strang, G.** (2004). A simple mesh generator in MATLAB. *SIAM Review*, 46(2), 329–345.
- **Hansen, P.C.** (1992). Analysis of discrete ill-posed problems by means of the L-curve. *SIAM Review*, 34(4), 561–580.
- **Boyd, S. et al.** (2011). Distributed optimization and statistical learning via the alternating direction method of multipliers. *Foundations and Trends in Machine Learning*, 3(1), 1–122.
- **Holder, D.S.** (2004). *Electrical Impedance Tomography: Methods, History and Applications*. Institute of Physics Publishing.
- **Adler, A. & Lionheart, W.R.B.** (2006). Uses and abuses of EIDORS: an extensible software base for EIT. *Physiological Measurement*, 27(5), S25–S42.
- **Mandache, N.** (2001). Exponential instability in an inverse problem for the Schrödinger equation. *Inverse Problems*, 17(5), 1435–1444.
