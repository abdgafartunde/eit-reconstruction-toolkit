"""
eitkit.forward.solver
======================
Linear system solver for the EIT forward problem.

Given the global stiffness matrix ``K`` and the Neumann load vector ``f``
(built by :mod:`eitkit.forward.gap_model`), this module applies the
Dirichlet ground condition and calls ``scipy.sparse.linalg.spsolve`` to
obtain the nodal potential vector ``u``.

Dirichlet ground condition
--------------------------
``K`` is singular (has one zero eigenvalue corresponding to an additive
constant).  To obtain a unique solution we pin one interior node to zero
potential.  The standard choice is the **node closest to the mesh
centroid**, i.e. the node with the smallest distance to ``(0, 0)`` that
is **not** on the boundary.  If all nodes are on the boundary (tiny
meshes), the first node is used instead.

The modification is performed on a copy of ``K`` so the original matrix
can be reused across drive steps.

Public API
----------
solve_forward(K, f, ground_node) → ndarray, shape (N,)
    Solve ``Ku = f`` with node ``ground_node`` pinned to zero.
pick_ground_node(mesh) → int
    Select a suitable interior node to use as the ground reference.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from numpy.typing import NDArray

from eitkit.mesh.mesh import Mesh

__all__ = ["solve_forward", "pick_ground_node", "apply_dirichlet_bc"]


def pick_ground_node(mesh: Mesh) -> int:
    """Return the index of the interior node nearest the mesh centre.

    Parameters
    ----------
    mesh:
        Triangular mesh.

    Returns
    -------
    ground : int
        Node index to use as the Dirichlet ground reference.
    """
    interior = mesh.interior_nodes()
    if interior.size == 0:
        return 0  # fallback for degenerate meshes in tests

    r2 = mesh.nodes[interior, 0] ** 2 + mesh.nodes[interior, 1] ** 2
    return int(interior[np.argmin(r2)])


def apply_dirichlet_bc(
    K: sp.csr_array,
    ground_node: int,
) -> sp.csr_array:
    """Return a copy of *K* with the ground-node row and column zeroed.

    This enforces ``u[ground_node] = 0`` (zero-potential Dirichlet BC)
    by the symmetric elimination approach: row *g* and column *g* are
    set to zero and ``K[g, g]`` is set to 1.  The resulting matrix can
    be factorised **once** and reused for all drive steps that share the
    same stiffness matrix.

    Parameters
    ----------
    K:
        Global stiffness matrix, shape ``(N, N)``.
    ground_node:
        Index of the node to pin to zero.

    Returns
    -------
    K_bc : scipy.sparse.csr_array, shape ``(N, N)``
        Modified stiffness matrix with Dirichlet BC applied.
    """
    K_mod = K.tolil()
    g = ground_node
    K_mod[g, :] = 0.0
    K_mod[:, g] = 0.0
    K_mod[g, g] = 1.0
    return K_mod.tocsr()


def solve_forward(
    K: sp.csr_array,
    f: NDArray[np.float64],
    ground_node: int,
) -> NDArray[np.float64]:
    """Solve the EIT forward system ``Ku = f`` with a grounded node.

    Parameters
    ----------
    K:
        Global stiffness matrix, shape ``(N, N)``.  Must be the
        ``csr_array`` returned by :func:`~eitkit.forward.fem_assembler.assemble_K`.
        A working copy is made internally; the original is not modified.
    f:
        Load vector, shape ``(N,)``.
    ground_node:
        Index of the node to pin to zero potential (Dirichlet BC).
        Use :func:`pick_ground_node` to select a suitable node.

    Returns
    -------
    u : ndarray, shape ``(N,)``, dtype ``float64``
        Nodal electric potential.  ``u[ground_node] == 0``.

    Raises
    ------
    ValueError
        If ``ground_node`` is out of range.
    scipy.sparse.linalg.MatrixRankWarning
        Propagated if the modified system is still singular (should not
        happen with a valid mesh and correct ground node).

    Notes
    -----
    The Dirichlet BC is enforced by zeroing row/column *g* of ``K`` and
    setting ``K[g, g] = 1``, ``f[g] = 0``.  This keeps the system
    symmetric.
    """
    N = K.shape[0]
    if not (0 <= ground_node < N):
        raise ValueError(
            f"ground_node {ground_node} out of range [0, {N})."
        )

    # Apply Dirichlet BC and solve.
    # Callers that perform many solves with the same K should call
    # apply_dirichlet_bc() once and pass the result here directly
    # (with ground_node=-1 to skip the redundant modification).
    if ground_node >= 0:
        K_bc = apply_dirichlet_bc(K, ground_node)
    else:
        K_bc = K  # caller already applied BC

    f_mod = f.copy()
    f_mod[ground_node] = 0.0

    u = spla.spsolve(K_bc, f_mod)
    return u.astype(np.float64)
