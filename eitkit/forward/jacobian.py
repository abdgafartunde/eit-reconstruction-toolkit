"""
eitkit.forward.jacobian
========================
Sensitivity (Jacobian) matrix for 2-D difference EIT via the adjoint method.

Theory
------
The Jacobian entry $J_{ie}$ is the partial derivative of voltage measurement
$i$ with respect to the conductivity of element $e$:

.. math::

    J_{ie} = \\frac{\\partial V_i}{\\partial \\sigma_e}
           = -\\int_{\\Omega_e} \\nabla u_k \\cdot \\nabla \\phi_i \\, d\\Omega

where

* $u_k$ is the **forward** potential for drive step $k$ (source/sink at the
  driven electrodes, conductivity $\\sigma$),
* $\\phi_i$ is the **adjoint** potential for measurement $i$: the solution with
  a unit virtual current injected at the plus-electrode and extracted at the
  minus-electrode of that measurement pair.

For a P1 triangle the element integral reduces to

.. math::

    J_{ie} = -\\frac{A_e}{4 A_e^2}
             \\bigl[(\\mathbf{b}_e \\cdot \\mathbf{u}_k^e)
                    (\\mathbf{b}_e \\cdot \\boldsymbol{\\phi}_i^e)
                  + (\\mathbf{c}_e \\cdot \\mathbf{u}_k^e)
                    (\\mathbf{c}_e \\cdot \\boldsymbol{\\phi}_i^e)\\bigr]
           = -\\frac{1}{4 A_e} \\bigl[(\\mathbf{b}_e \\cdot \\mathbf{u}_k^e)
                                       (\\mathbf{b}_e \\cdot \\boldsymbol{\\phi}_i^e)
                                     + (\\mathbf{c}_e \\cdot \\mathbf{u}_k^e)
                                       (\\mathbf{c}_e \\cdot \\boldsymbol{\\phi}_i^e)\\bigr]

where $\\mathbf{b}_e, \\mathbf{c}_e \\in \\mathbb{R}^3$ are the gradient
coefficients of element $e$, and $\\mathbf{u}_k^e = (u_k[n_0], u_k[n_1],
u_k[n_2])$ are the nodal potentials at the three corners of element $e$.

Adjoint reuse
-------------
For the adjacent-drive / adjacent-measurement protocol each unique
$(e^+, e^-)$ measurement pair appears in exactly $L-3$ drive steps.
So we solve **L adjoint problems** (not P) and look them up by pair index.
This halves the total solve count relative to a naive implementation.

Public API
----------
compute_jacobian(mesh, elec_config, sigma, drive_pairs, meas_pairs,
                 current) → ndarray, shape (P, M)
    Compute the full sensitivity matrix at conductivity *sigma*.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from numpy.typing import NDArray

from eitkit.forward.fem_assembler import assemble_K
from eitkit.forward.gap_model import build_load_vector
from eitkit.forward.solver import apply_dirichlet_bc, pick_ground_node
from eitkit.mesh.electrode_placement import ElectrodeConfig
from eitkit.mesh.mesh import Mesh

__all__ = ["compute_jacobian"]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _gradient_coefficients(
    mesh: Mesh,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return per-element gradient coefficients *b* and *c*.

    For element *e* with local nodes 0,1,2 (indices mod 3):

        b[e, i] = y[i+1] - y[i+2]
        c[e, i] = x[i+2] - x[i+1]

    Parameters
    ----------
    mesh : Mesh

    Returns
    -------
    b, c : ndarray, each shape ``(M, 3)``
    """
    x = mesh.nodes[mesh.elements, 0]  # (M, 3)
    y = mesh.nodes[mesh.elements, 1]  # (M, 3)

    b = np.empty_like(x)
    c = np.empty_like(x)
    for i in range(3):
        j = (i + 1) % 3
        k = (i + 2) % 3
        b[:, i] = y[:, j] - y[:, k]
        c[:, i] = x[:, k] - x[:, j]
    return b, c


def _element_grad_projections(
    mesh: Mesh,
    u: NDArray[np.float64],
    b: NDArray[np.float64],
    c: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Project nodal potential *u* onto element gradient directions.

    Returns
    -------
    gu_b : ndarray, shape ``(M,)``
        ``sum_j b[e,j] * u[node_j]`` for each element.
    gu_c : ndarray, shape ``(M,)``
        ``sum_j c[e,j] * u[node_j]`` for each element.
    """
    u_e = u[mesh.elements]  # (M, 3)  — nodal values per element
    gu_b = (b * u_e).sum(axis=1)  # (M,)
    gu_c = (c * u_e).sum(axis=1)  # (M,)
    return gu_b, gu_c


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_jacobian(
    mesh: Mesh,
    elec_config: ElectrodeConfig,
    sigma: NDArray[np.floating],
    drive_pairs: NDArray[np.int32],
    meas_pairs: NDArray[np.int32],
    current: float = 1.0,
) -> NDArray[np.float64]:
    """Compute the EIT Jacobian (sensitivity matrix) at conductivity *sigma*.

    Parameters
    ----------
    mesh : Mesh
        Triangular mesh.
    elec_config : ElectrodeConfig
        Electrode layout (gap model).
    sigma : ndarray, shape ``(M,)``
        Per-element conductivity at which the Jacobian is evaluated.
        For a linearised reconstruction around a background, pass the
        background conductivity $\\sigma_0$.
    drive_pairs : ndarray, shape ``(L, 2)``
        Stimulation schedule (from :func:`~eitkit.protocol.adjacent_pattern`).
    meas_pairs : ndarray, shape ``(P, 3)``
        Measurement schedule (from :func:`~eitkit.protocol.measurement_pairs`).
        Columns: ``[drive_step, plus_electrode, minus_electrode]``.
    current : float
        Injected current in Amperes (default 1.0 A).

    Returns
    -------
    J : ndarray, shape ``(P, M)``, dtype ``float64``
        Jacobian matrix.  ``J[i, e] = ∂V_i / ∂σ_e``.

    Notes
    -----
    Total linear solves: ``L`` forward + ``L_unique`` adjoint, where
    ``L_unique`` is the number of distinct ``(plus_el, minus_el)`` pairs
    in *meas_pairs* (at most ``L`` for the adjacent protocol).
    """
    sigma = np.asarray(sigma, dtype=np.float64)

    # ── Step 5a: element gradient coefficients ────────────────────────────
    b, c = _gradient_coefficients(mesh)
    prefactor = -1.0 / (4.0 * mesh.areas)  # (M,)  scalar per element

    # ── Step 5b: assemble K, apply BC once, factorize once ────────────────
    g_node = pick_ground_node(mesh)
    K_bc = apply_dirichlet_bc(assemble_K(mesh, sigma), g_node).tocsc()
    lu = sp.linalg.splu(K_bc)
    L = len(drive_pairs)

    # Cache forward gradient projections: shape (L, M) each
    GUb = np.empty((L, mesh.n_elements), dtype=np.float64)
    GUc = np.empty((L, mesh.n_elements), dtype=np.float64)

    for k, dp in enumerate(drive_pairs):
        f = build_load_vector(mesh.n_nodes, elec_config, dp, current)
        f[g_node] = 0.0
        u_k = lu.solve(f)
        GUb[k], GUc[k] = _element_grad_projections(mesh, u_k, b, c)

    # ── Step 5c: solve unique adjoint problems (one factorization reused) ─
    unique_pairs = np.unique(meas_pairs[:, 1:], axis=0)  # (Q, 2)

    adj_gb: dict[tuple[int, int], NDArray[np.float64]] = {}
    adj_gc: dict[tuple[int, int], NDArray[np.float64]] = {}

    for plus_el, minus_el in unique_pairs:
        f_adj = build_load_vector(
            mesh.n_nodes,
            elec_config,
            drive_pair=(int(plus_el), int(minus_el)),
            current=1.0,
        )
        f_adj[g_node] = 0.0
        phi = lu.solve(f_adj)
        key = (int(plus_el), int(minus_el))
        adj_gb[key], adj_gc[key] = _element_grad_projections(mesh, phi, b, c)

    # ── Step 5d: assemble J ───────────────────────────────────────────────
    P = len(meas_pairs)
    J = np.empty((P, mesh.n_elements), dtype=np.float64)

    for i, (k, plus_el, minus_el) in enumerate(meas_pairs):
        key = (int(plus_el), int(minus_el))
        J[i, :] = prefactor * (GUb[int(k)] * adj_gb[key] + GUc[int(k)] * adj_gc[key])

    return J
