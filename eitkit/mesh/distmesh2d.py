"""
eitkit.mesh.distmesh2d
======================
Quality 2-D triangular mesh generation for the unit disc using a
node-relaxation algorithm inspired by the DistMesh method of Persson & Strang
(2004).  All code is written from scratch; no external mesh library is used.

Algorithm summary
-----------------
1. Seed an initial point cloud inside the unit disc on a regular hex grid.
2. Add the $L$ electrode nodes exactly on the boundary.
3. Triangulate the points with scipy.spatial.Delaunay.
4. Remove triangles whose circumcentre lies outside the domain.
5. Iterate: compute internal "spring" forces along edges, move nodes,
   re-project boundary nodes onto the circle, re-triangulate.
6. Enforce CCW winding and re-compute areas.
7. Return a :class:`~eitkit.mesh.mesh.Mesh`.

References
----------
Persson, P.-O. & Strang, G. (2004).  A simple mesh generator in MATLAB.
*SIAM Review*, 46(2), 329–345.  https://doi.org/10.1137/S0036144503429121

Public API
----------
make_circle_mesh(n_electrodes, h0, radius, max_iter, seed)
    Generate a quality triangular mesh on a disc of given radius.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.spatial import Delaunay  # type: ignore[import]

from eitkit.mesh.mesh import Mesh

__all__ = ["make_circle_mesh"]

# ---------------------------------------------------------------------------
# Public entry-point
# ---------------------------------------------------------------------------


def make_circle_mesh(
    n_electrodes: int = 16,
    h0: float = 0.10,
    radius: float = 1.0,
    max_iter: int = 200,
    seed: int = 42,
) -> Mesh:
    """Generate a quality 2-D triangular mesh on a circular disc.

    Parameters
    ----------
    n_electrodes : int
        Number of electrodes to place on the boundary.  Electrode nodes are
        fixed on the circle and are never moved during relaxation.
    h0 : float
        Target edge length (controls mesh density).  Smaller values produce
        finer meshes.  Typical range: 0.05 (fine) – 0.15 (coarse).
    radius : float
        Radius of the circular domain.  Defaults to 1.0 (unit disc).
    max_iter : int
        Maximum number of relaxation iterations.
    seed : int
        Random seed for reproducibility of the initial node jitter.

    Returns
    -------
    Mesh
        A :class:`~eitkit.mesh.mesh.Mesh` with CCW-oriented triangles,
        sorted boundary nodes, and pre-computed element areas.
    """
    rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------
    # 1. Initial hex-grid point cloud inside the disc
    # ------------------------------------------------------------------
    nodes = _hex_grid(h0, radius)

    # ------------------------------------------------------------------
    # 2. Add electrode nodes exactly on the boundary (fixed during relax)
    # ------------------------------------------------------------------
    elec_angles = np.linspace(0.0, 2.0 * np.pi, n_electrodes, endpoint=False)
    elec_nodes = radius * np.column_stack((np.cos(elec_angles), np.sin(elec_angles)))
    nodes = np.vstack([nodes, elec_nodes])

    # Remove interior points that are too close to electrode nodes
    nodes = _remove_close_points(nodes, h0 * 0.5, n_protect=n_electrodes)

    # n_fixed: electrode nodes are always the last n_electrodes rows
    # (guaranteed by _remove_close_points protecting them)
    n_fixed = n_electrodes

    # Add a tiny jitter to break Delaunay degeneracies (but not to fixed nodes)
    jitter = rng.uniform(-1e-4 * h0, 1e-4 * h0, size=(len(nodes) - n_fixed, 2))
    nodes[: len(nodes) - n_fixed] += jitter

    # ------------------------------------------------------------------
    # 3. Iterative relaxation
    # ------------------------------------------------------------------
    nodes = _relax(nodes, n_fixed, h0, radius, max_iter)

    # ------------------------------------------------------------------
    # 4. Final Delaunay triangulation and domain filtering
    # ------------------------------------------------------------------
    tri = Delaunay(nodes)
    elements = tri.simplices.astype(np.int32)
    elements = _filter_outside(elements, nodes, radius)

    # ------------------------------------------------------------------
    # 5. Enforce CCW winding
    # ------------------------------------------------------------------
    elements = _enforce_ccw(elements, nodes)

    # ------------------------------------------------------------------
    # 6. Identify boundary nodes sorted CCW by angle
    # ------------------------------------------------------------------
    boundary_nodes = _boundary_nodes(nodes, radius)

    return Mesh(
        nodes=nodes,
        elements=elements,
        boundary_nodes=boundary_nodes,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _hex_grid(h0: float, radius: float) -> NDArray[np.float64]:
    """Seed a hex-offset point cloud filling the disc of given radius."""
    row_spacing = h0 * np.sqrt(3.0) / 2.0
    pts: list[list[float]] = []
    y = -radius
    row = 0
    while y <= radius + 1e-10:
        x_offset = (h0 / 2.0) if (row % 2 == 1) else 0.0
        x = -radius + x_offset
        while x <= radius + 1e-10:
            if x**2 + y**2 <= (radius - 1e-3 * h0) ** 2:
                pts.append([x, y])
            x += h0
        y += row_spacing
        row += 1
    return np.array(pts, dtype=np.float64)


def _remove_close_points(
    nodes: NDArray[np.float64],
    min_dist: float,
    n_protect: int = 0,
) -> NDArray[np.float64]:
    """Remove points closer than *min_dist* to any later point in the array.

    The last *n_protect* rows are never removed (they are the fixed electrode
    nodes).  Interior nodes that are too close to an electrode node are
    removed instead of the electrode node.
    """
    keep = np.ones(len(nodes), dtype=bool)
    n_total = len(nodes)
    for i in range(n_total - 1):
        if not keep[i]:
            continue
        dists = np.sqrt(np.sum((nodes[i + 1 :] - nodes[i]) ** 2, axis=1))
        too_close = np.where(dists < min_dist)[0] + (i + 1)
        for j in too_close:
            # Never remove a protected (electrode) node; remove i instead
            if j >= n_total - n_protect:
                keep[i] = False
                break
            else:
                keep[j] = False
    return nodes[keep]


def _relax(
    nodes: NDArray[np.float64],
    n_fixed: int,
    h0: float,
    radius: float,
    max_iter: int,
    delta_t: float = 0.2,
    f_tol: float = 1e-3,
) -> NDArray[np.float64]:
    """Relax interior nodes using a spring-force analogy.

    Each edge acts as a spring with rest length proportional to *h0*.
    Boundary nodes are projected onto the circle after each step.
    Fixed nodes (electrodes) are never moved.

    Parameters
    ----------
    nodes : ndarray, shape (N, 2)
    n_fixed : int
        Number of fixed (electrode) nodes at the *end* of *nodes*.
    h0 : float
        Target edge length.
    radius : float
    max_iter : int
    delta_t : float
        Step size for node displacement.
    f_tol : float
        Convergence tolerance (max displacement relative to *h0*).
    """
    n_total = len(nodes)
    n_free = n_total - n_fixed

    for _ in range(max_iter):
        # Triangulate
        tri = Delaunay(nodes)
        elems = tri.simplices

        # Collect all unique edges
        edges = _edges_from_elements(elems)

        # Spring forces
        force = np.zeros_like(nodes)
        for e0, e1 in edges:
            diff = nodes[e1] - nodes[e0]
            L_actual = np.linalg.norm(diff)
            if L_actual < 1e-14:
                continue
            # Rest length = h0 (uniform desired edge length)
            F = max(h0 - L_actual, 0.0) / L_actual
            fvec = F * diff
            force[e0] -= fvec
            force[e1] += fvec

        # Move free nodes only
        displacement = delta_t * force[:n_free]
        nodes[:n_free] += displacement

        # Re-project free boundary nodes onto the circle
        r = np.sqrt(nodes[:n_free, 0] ** 2 + nodes[:n_free, 1] ** 2)
        outside = r > radius
        if np.any(outside):
            nodes[:n_free][outside] *= radius / r[outside, np.newaxis]

        # Convergence check
        if np.max(np.linalg.norm(displacement, axis=1)) < f_tol * h0:
            break

    return nodes


def _edges_from_elements(
    elements: NDArray[np.int32],
) -> NDArray[np.int32]:
    """Return unique undirected edge pairs from a triangle connectivity array."""
    all_edges = np.vstack(
        [
            elements[:, [0, 1]],
            elements[:, [1, 2]],
            elements[:, [2, 0]],
        ]
    )
    # Normalise direction: smaller index first
    all_edges = np.sort(all_edges, axis=1)
    return np.unique(all_edges, axis=0)


def _filter_outside(
    elements: NDArray[np.int32],
    nodes: NDArray[np.float64],
    radius: float,
    tol: float = 0.01,
) -> NDArray[np.int32]:
    """Remove triangles whose centroid lies outside the domain."""
    centroids = nodes[elements].mean(axis=1)
    r_centroid = np.sqrt(centroids[:, 0] ** 2 + centroids[:, 1] ** 2)
    inside = r_centroid <= radius * (1.0 + tol)
    return elements[inside]


def _enforce_ccw(
    elements: NDArray[np.int32],
    nodes: NDArray[np.float64],
) -> NDArray[np.int32]:
    """Flip any CW-wound triangle to CCW by swapping columns 1 and 2."""
    p0 = nodes[elements[:, 0]]
    p1 = nodes[elements[:, 1]]
    p2 = nodes[elements[:, 2]]
    cross = (p1[:, 0] - p0[:, 0]) * (p2[:, 1] - p0[:, 1]) - (p2[:, 0] - p0[:, 0]) * (
        p1[:, 1] - p0[:, 1]
    )
    cw = cross < 0
    elements[cw] = elements[cw][:, [0, 2, 1]]
    return elements


def _boundary_nodes(
    nodes: NDArray[np.float64],
    radius: float,
    tol: float = 1e-6,
) -> NDArray[np.int32]:
    """Return indices of nodes on the boundary, sorted CCW by angle."""
    r = np.sqrt(nodes[:, 0] ** 2 + nodes[:, 1] ** 2)
    on_boundary = np.where(np.abs(r - radius) <= tol)[0].astype(np.int32)
    angles = np.arctan2(nodes[on_boundary, 1], nodes[on_boundary, 0])
    order = np.argsort(angles)
    return on_boundary[order]
