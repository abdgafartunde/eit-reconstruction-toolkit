"""
eitkit.forward.fem_assembler
=============================
FEM stiffness-matrix assembly for 2-D EIT.

For a piecewise-constant conductivity distribution $\\sigma$ (one value
per triangular element) the global stiffness matrix is assembled by
summing element contributions::

    K = Σ_e  K^e(σ_e)

where the element stiffness matrix for a linear (P1) triangle with
nodes $(x_0,y_0),(x_1,y_1),(x_2,y_2)$ and conductivity $\\sigma_e$ is

.. math::

    K^e = \\frac{\\sigma_e}{4 A_e}  B^T B

with the gradient matrix

.. math::

    B = \\begin{bmatrix}
            b_0 & b_1 & b_2 \\\\
            c_0 & c_1 & c_2
        \\end{bmatrix},
    \\quad
    b_i = y_{i+1} - y_{i+2},
    \\quad
    c_i = x_{i+2} - x_{i+1}

(indices mod 3) and $A_e$ is the (positive) element area.

The assembly is fully vectorised over all elements using NumPy advanced
indexing; no Python loops over elements.  The result is returned as a
``scipy.sparse.csr_array`` for efficient downstream solves.

Public API
----------
assemble_K(mesh, sigma) → scipy.sparse.csr_array, shape (N, N)
    Assemble the global FEM stiffness matrix.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from numpy.typing import NDArray

from eitkit.mesh.mesh import Mesh

__all__ = ["assemble_K"]


def assemble_K(
    mesh: Mesh,
    sigma: NDArray[np.floating],
) -> sp.csr_array:
    """Assemble the global FEM stiffness matrix for conductivity *sigma*.

    Parameters
    ----------
    mesh:
        Triangular mesh.  Must have CCW-wound elements (guaranteed by
        :func:`~eitkit.mesh.make_circle_mesh`).
    sigma:
        Per-element conductivity, shape ``(M,)`` in S/m.  All values must
        be strictly positive.

    Returns
    -------
    K : scipy.sparse.csr_array, shape ``(N, N)``
        Symmetric positive semi-definite stiffness matrix.  One zero
        eigenvalue corresponds to the free constant (ground) mode; the
        caller must pin one node before solving.

    Raises
    ------
    ValueError
        If ``sigma`` has the wrong length or contains non-positive values.

    Notes
    -----
    The assembly is fully vectorised — no Python loop over elements.
    Duplicate ``(i, j)`` entries arising from shared edges are summed
    automatically by ``scipy.sparse.coo_array``.
    """
    sigma = np.asarray(sigma, dtype=np.float64)
    if sigma.shape != (mesh.n_elements,):
        raise ValueError(
            f"sigma must have shape ({mesh.n_elements},), got {sigma.shape}."
        )
    if np.any(sigma <= 0):
        raise ValueError("All conductivity values must be strictly positive.")

    nodes = mesh.nodes          # (N, 2)
    elems = mesh.elements       # (M, 3)  int32
    areas = mesh.areas          # (M,)    float64

    M = mesh.n_elements
    N = mesh.n_nodes

    # Node coordinates for each element corner, shape (M, 3)
    x = nodes[elems, 0]         # (M, 3)
    y = nodes[elems, 1]         # (M, 3)

    # Gradient coefficients (vectorised, indices mod 3)
    # b_i = y_{i+1 mod 3} - y_{i+2 mod 3}
    # c_i = x_{i+2 mod 3} - x_{i+1 mod 3}
    b = np.empty((M, 3), dtype=np.float64)
    c = np.empty((M, 3), dtype=np.float64)
    for i in range(3):
        j = (i + 1) % 3
        k = (i + 2) % 3
        b[:, i] = y[:, j] - y[:, k]
        c[:, i] = x[:, k] - x[:, j]

    # Element stiffness: K^e_ij = σ_e / (4 A_e) * (b_i b_j + c_i c_j)
    # Vectorise over all 9 (local_i, local_j) combinations
    prefactor = sigma / (4.0 * areas)   # (M,)

    # Build COO data arrays: 9 entries per element = 9*M total
    rows_list = []
    cols_list = []
    vals_list = []

    for li in range(3):
        for lj in range(3):
            val = prefactor * (b[:, li] * b[:, lj] + c[:, li] * c[:, lj])
            rows_list.append(elems[:, li])
            cols_list.append(elems[:, lj])
            vals_list.append(val)

    rows = np.concatenate(rows_list).astype(np.int32)
    cols = np.concatenate(cols_list).astype(np.int32)
    vals = np.concatenate(vals_list)

    K = sp.coo_array((vals, (rows, cols)), shape=(N, N))
    return K.tocsr()
