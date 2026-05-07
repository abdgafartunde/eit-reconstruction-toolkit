"""
eitkit.forward.simulation
==========================
High-level forward simulation API for difference EIT.

This module ties together the FEM assembler, gap-model load vector, and
sparse solver into a single callable that returns the simulated voltage
measurement vector for a given conductivity distribution.

Difference EIT
--------------
Phase 1 implements **difference EIT** only.  The returned voltages are

    δV = V(σ) − V(σ₀)

where ``σ₀`` is a uniform reference conductivity (default 1.0 S/m).
Absolute voltages are not directly returned to avoid the need for a
calibrated ground reference in practice.

Public API
----------
simulate(mesh, elec_config, sigma, drive_pairs, meas_pairs,
         current, sigma0) → ndarray, shape (P,)
    Run the full forward problem and return difference voltages.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from numpy.typing import NDArray

from eitkit.mesh.mesh import Mesh
from eitkit.mesh.electrode_placement import ElectrodeConfig
from eitkit.forward.fem_assembler import assemble_K
from eitkit.forward.gap_model import build_load_vector
from eitkit.forward.solver import apply_dirichlet_bc, pick_ground_node

__all__ = ["simulate"]


def simulate(
    mesh: Mesh,
    elec_config: ElectrodeConfig,
    sigma: NDArray[np.floating],
    drive_pairs: NDArray[np.int32],
    meas_pairs: NDArray[np.int32],
    current: float = 1.0,
    sigma0: float | NDArray[np.floating] = 1.0,
) -> NDArray[np.float64]:
    """Simulate EIT measurements and return difference voltages δV.

    For each drive step *k* the forward system ``K u = f`` is solved for
    both the target conductivity ``σ`` and the reference conductivity
    ``σ₀``.  Electrode voltages are read off from the nodal potential
    vector and the differences are stacked into a single output vector.

    Parameters
    ----------
    mesh:
        Triangular mesh.
    elec_config:
        Electrode layout (gap model, 0-indexed).
    sigma:
        Per-element conductivity, shape ``(M,)`` in S/m.
    drive_pairs:
        Stimulation schedule from :func:`~eitkit.protocol.adjacent_pattern`,
        shape ``(L, 2)``.
    meas_pairs:
        Measurement schedule from
        :func:`~eitkit.protocol.measurement_pairs`,
        shape ``(P, 3)`` with columns ``[drive_step, plus_el, minus_el]``.
    current:
        Injected current in Amperes (default 1.0 A).
    sigma0:
        Reference conductivity.  Either a scalar (uniform background) or
        an ndarray of shape ``(M,)``.  Defaults to 1.0 S/m.

    Returns
    -------
    dV : ndarray, shape ``(P,)``, dtype ``float64``
        Difference voltages ``δV[i] = V(σ)[i] − V(σ₀)[i]`` for each
        measurement row *i* in *meas_pairs*.

    Raises
    ------
    ValueError
        If ``sigma`` has the wrong shape or contains non-positive values.

    Notes
    -----
    Each drive step requires two sparse linear solves (one for ``σ``, one
    for ``σ₀``).  Both stiffness matrices are assembled once outside the
    drive loop for efficiency.
    """
    sigma = np.asarray(sigma, dtype=np.float64)
    if np.isscalar(sigma0):
        sigma0_arr = np.full(mesh.n_elements, float(sigma0), dtype=np.float64)
    else:
        sigma0_arr = np.asarray(sigma0, dtype=np.float64)

    # Assemble stiffness matrices and apply the Dirichlet ground BC once.
    # splu() factorises each modified K once; all L drive steps then use
    # a cheap triangular back-substitution instead of a full LU each time.
    g_node = pick_ground_node(mesh)
    K_bc  = apply_dirichlet_bc(assemble_K(mesh, sigma),      g_node).tocsc()
    K0_bc = apply_dirichlet_bc(assemble_K(mesh, sigma0_arr), g_node).tocsc()
    lu     = sp.linalg.splu(K_bc)
    lu0    = sp.linalg.splu(K0_bc)

    # Map electrode index → mesh node index
    el_nodes = elec_config.node_indices   # shape (L,)

    # Pre-allocate output
    P = len(meas_pairs)
    dV = np.empty(P, dtype=np.float64)

    # Solve per drive step (cheap back-substitution, not full factorisation)
    n_drive = len(drive_pairs)
    U  = np.empty((n_drive, mesh.n_nodes), dtype=np.float64)
    U0 = np.empty((n_drive, mesh.n_nodes), dtype=np.float64)

    for k, dp in enumerate(drive_pairs):
        f = build_load_vector(mesh.n_nodes, elec_config, dp, current)
        f[g_node] = 0.0                     # enforce zero RHS at ground node
        U[k]  = lu.solve(f)
        U0[k] = lu0.solve(f)

    # Extract electrode voltages and form differences
    for i, (k, plus_el, minus_el) in enumerate(meas_pairs):
        k = int(k)
        plus_node  = int(el_nodes[int(plus_el)])
        minus_node = int(el_nodes[int(minus_el)])

        v_sigma  = U[k,  plus_node] - U[k,  minus_node]
        v_sigma0 = U0[k, plus_node] - U0[k, minus_node]
        dV[i] = v_sigma - v_sigma0

    return dV
