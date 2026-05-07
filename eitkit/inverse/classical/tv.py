r"""
Total Variation (TV) regularisation for EIT difference imaging.

Theory
------
TV regularisation promotes **edge-preserving** reconstructions by penalising
the L1 norm of the conductivity gradient rather than the L2 norm used by
Tikhonov.  The optimisation problem is

.. math::

    \delta\hat\sigma = \operatorname{arg\,min}_{\delta\sigma}
        \tfrac{1}{2}\|J\,\delta\sigma - \delta V\|_2^2
        + \alpha \|D\,\delta\sigma\|_1

where :math:`D \in \mathbb{R}^{F \times E}` is the element-to-element finite
difference (gradient) operator built from the dual mesh (shared edges between
adjacent triangles), and :math:`\alpha > 0` is the regularisation strength.

Algorithm — ADMM / split-Bregman
---------------------------------
Introduce splitting :math:`z = D\,\delta\sigma` and solve via ADMM:

1. **σ-update** (quadratic sub-problem — direct solve):

   .. math::

       \delta\sigma^{k+1} = (J^T J + \rho D^T D)^{-1}
           \bigl(J^T \delta V + \rho D^T (z^k - u^k)\bigr)

2. **z-update** (element-wise soft-thresholding / proximal of L1):

   .. math::

       z^{k+1} = \mathcal{S}_{\alpha/\rho}(D\,\delta\sigma^{k+1} + u^k)

3. **u-update** (dual variable / scaled residual):

   .. math::

       u^{k+1} = u^k + D\,\delta\sigma^{k+1} - z^{k+1}

Convergence is checked by primal and dual residual norms.

Public API
----------
build_gradient_op   Sparse element-to-element finite difference matrix D.
tv_solve            Reconstruct δσ via ADMM TV minimisation.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from numpy.typing import NDArray

from eitkit.mesh.mesh import Mesh

__all__ = ["build_gradient_op", "tv_solve"]


# ── Gradient operator ─────────────────────────────────────────────────────────


def build_gradient_op(mesh: Mesh) -> sp.csr_array:
    r"""Build a sparse element-to-element finite difference operator.

    For each pair of triangles that share an edge, the operator inserts
    a row ``[…, +1, …, −1, …]`` (higher-index element minus lower-index
    element).  The resulting matrix ``D`` has shape ``(F, E)`` where
    ``F`` is the number of interior shared edges and ``E = mesh.n_elements``.

    The L1-seminorm ``||D δσ||_1`` is a discrete approximation of the TV
    functional :math:`\int|\nabla\sigma|` on the triangular mesh.

    Parameters
    ----------
    mesh:
        Triangular mesh produced by :func:`~eitkit.mesh.make_circle_mesh`.

    Returns
    -------
    D : sparse CSR array, shape ``(F, E)``
        Signed incidence matrix of the element adjacency graph.
        Each row corresponds to one shared edge; the two non-zero entries
        are ``+1`` and ``-1``.
    """
    elems = mesh.elements  # (E, 3)  int32
    E = mesh.n_elements

    # Build a map: frozenset{node_i, node_j} → list of element indices
    # for all edges in the mesh.
    from collections import defaultdict

    edge_to_elems: dict[tuple[int, int], list[int]] = defaultdict(list)

    for e_idx, tri in enumerate(elems):
        for k in range(3):
            # Canonical edge: (min, max) node pair
            n0, n1 = int(tri[k]), int(tri[(k + 1) % 3])
            key = (min(n0, n1), max(n0, n1))
            edge_to_elems[key].append(e_idx)

    # Collect shared (interior) edges — exactly two elements per edge
    rows, cols, data = [], [], []
    row = 0
    for _key, elems_list in edge_to_elems.items():
        if len(elems_list) == 2:
            e0, e1 = elems_list[0], elems_list[1]
            # Convention: larger index − smaller index → +1, −1
            if e0 > e1:
                e0, e1 = e1, e0
            rows += [row, row]
            cols += [e1, e0]
            data += [1.0, -1.0]
            row += 1

    F = row  # number of shared edges (interior faces in 2-D)
    D = sp.csr_array(
        (data, (rows, cols)),
        shape=(F, E),
        dtype=np.float64,
    )
    return D


# ── ADMM solver ───────────────────────────────────────────────────────────────


def _soft_threshold(x: NDArray[np.float64], kappa: float) -> NDArray[np.float64]:
    """Element-wise soft-thresholding (proximal operator of kappa * ||.||_1)."""
    return np.sign(x) * np.maximum(np.abs(x) - kappa, 0.0)


def tv_solve(
    J: NDArray[np.float64],
    dV: NDArray[np.float64],
    alpha: float,
    mesh: Mesh,
    *,
    rho: float = 1.0,
    max_iter: int = 200,
    tol: float = 1e-4,
    warm_start: NDArray[np.float64] | None = None,
) -> NDArray[np.float64]:
    r"""Total Variation regularised EIT reconstruction via ADMM.

    Minimises

    .. math::

        \tfrac{1}{2}\|J\,\delta\sigma - \delta V\|_2^2
        + \alpha \|D\,\delta\sigma\|_1

    using the Alternating Direction Method of Multipliers (ADMM) with a
    constant penalty parameter ``rho``.

    Parameters
    ----------
    J:
        Jacobian / sensitivity matrix, shape ``(P, E)``.
    dV:
        Difference voltage vector, shape ``(P,)``.
    alpha:
        TV regularisation strength :math:`\alpha > 0`.
        Larger values produce smoother, more homogeneous reconstructions.
    mesh:
        Triangular mesh — needed to build the gradient operator ``D``.
    rho:
        ADMM penalty parameter.  Typical range ``[0.1, 10]``.
        Does not affect the optimal solution, only convergence speed.
    max_iter:
        Maximum number of ADMM iterations.
    tol:
        Convergence tolerance on the relative primal residual
        ``||D δσ − z|| / max(||D δσ||, ||z||, 1)``.
    warm_start:
        Optional initial guess for ``delta_sigma``, shape ``(E,)``.
        Defaults to the zero vector.

    Returns
    -------
    delta_sigma : ndarray, shape ``(E,)``, dtype ``float64``
        Reconstructed conductivity perturbation.

    Raises
    ------
    ValueError
        If ``J`` is not 2-D, ``dV`` length mismatches ``J`` rows, or
        ``alpha`` / ``rho`` are not positive finite scalars.

    Notes
    -----
    The σ-update requires solving one linear system per iteration.  The
    matrix ``H = J^T J + ρ D^T D`` is assembled once and factorised with
    ``scipy.sparse.linalg.splu`` for efficiency.

    References
    ----------
    .. [Boyd2011] Boyd et al. (2011). Distributed Optimization and Statistical
       Learning via the Alternating Direction Method of Multipliers.
       *Foundations and Trends in Machine Learning*, 3(1), 1–122.
    """
    # ── input validation ──────────────────────────────────────────────────
    J = np.asarray(J, dtype=np.float64)
    dV = np.asarray(dV, dtype=np.float64)

    if J.ndim != 2:
        raise ValueError(f"J must be 2-D, got shape {J.shape}")
    if dV.ndim != 1:
        raise ValueError(f"dV must be 1-D, got shape {dV.shape}")
    P, E = J.shape
    if dV.shape[0] != P:
        raise ValueError(f"dV length {dV.shape[0]} does not match J rows {P}")
    if not (np.isfinite(alpha) and alpha > 0):
        raise ValueError(f"alpha must be a positive finite scalar, got {alpha!r}")
    if not (np.isfinite(rho) and rho > 0):
        raise ValueError(f"rho must be a positive finite scalar, got {rho!r}")

    # ── build gradient operator ───────────────────────────────────────────
    D = build_gradient_op(mesh)  # (F, E)  sparse CSR
    F_rows = D.shape[0]

    # ── pre-factor the σ-update system ───────────────────────────────────
    # H = J^T J + ρ D^T D  — symmetric positive definite
    JtJ = J.T @ J  # (E, E)  dense
    DtD = D.T @ D  # (E, E)  sparse
    H = sp.csr_array(JtJ) + rho * DtD  # (E, E)  sparse
    Jt_dV = J.T @ dV  # (E,)    dense

    lu = spla.splu(H.tocsc())

    # ── initialise ADMM variables ─────────────────────────────────────────
    if warm_start is not None:
        ds = np.asarray(warm_start, dtype=np.float64).copy()
    else:
        ds = np.zeros(E, dtype=np.float64)

    z = np.zeros(F_rows, dtype=np.float64)  # splitting variable
    u = np.zeros(F_rows, dtype=np.float64)  # scaled dual variable

    kappa = alpha / rho  # soft-threshold level

    # ── ADMM iterations ───────────────────────────────────────────────────
    for _ in range(max_iter):
        # 1. σ-update: solve H ds = J^T dV + ρ D^T (z - u)
        rhs = Jt_dV + rho * D.T @ (z - u)
        ds = lu.solve(np.asarray(rhs).ravel())

        # 2. z-update: soft-threshold
        Dds = D @ ds
        z_new = _soft_threshold(Dds + u, kappa)

        # 3. u-update: dual ascent
        u = u + Dds - z_new

        # 4. convergence check on primal residual ||D ds - z||
        primal_res = np.linalg.norm(Dds - z_new)
        scale = max(np.linalg.norm(Dds), np.linalg.norm(z_new), 1.0)
        z = z_new
        if primal_res / scale < tol:
            break

    return ds.astype(np.float64)
