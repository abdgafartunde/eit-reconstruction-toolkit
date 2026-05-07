"""
eitkit.mesh.electrode_placement
================================
Electrode placement for 2-D EIT meshes.

For the **gap electrode model** used in Phase 1, each electrode is
represented by the single boundary node closest to the electrode's
target angular position.  This gives one node per electrode with no
contact impedance — the simplest possible electrode representation.

The :func:`place_electrodes` function returns an :class:`ElectrodeConfig`
dataclass that is consumed by both the protocol module (to build
stimulation patterns) and the forward solver (to apply boundary
conditions).

Public API
----------
ElectrodeConfig
    Dataclass holding node indices, angles, and arc half-widths.
place_electrodes(mesh, n_electrodes, start_angle)
    Assign boundary nodes to electrodes.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from eitkit.mesh.mesh import Mesh

__all__ = ["ElectrodeConfig", "place_electrodes"]


# ---------------------------------------------------------------------------
# Data container
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ElectrodeConfig:
    """Electrode layout on a 2-D circular mesh (gap model).

    Attributes
    ----------
    node_indices : ndarray, shape (L,)
        Index into ``mesh.nodes`` for each of the $L$ electrodes.
        Electrode $\\ell$ is represented by node ``node_indices[ell]``.
    angles : ndarray, shape (L,)
        Angular position (radians, CCW from positive $x$-axis) of each
        electrode node.
    arc_width : float
        Target arc half-width in radians (= $\\pi / L$).  Used by the
        forward solver to define the Neumann patch width in the gap model.
    n_electrodes : int
        Number of electrodes $L$.
    """

    node_indices: NDArray[np.int32]
    angles: NDArray[np.float64]
    arc_width: float
    n_electrodes: int

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "node_indices", np.asarray(self.node_indices, dtype=np.int32)
        )
        object.__setattr__(self, "angles", np.asarray(self.angles, dtype=np.float64))
        if len(self.node_indices) != self.n_electrodes:
            raise ValueError(
                f"node_indices length {len(self.node_indices)} does not match "
                f"n_electrodes={self.n_electrodes}."
            )

    def __repr__(self) -> str:  # noqa: D105
        return (
            f"ElectrodeConfig(n_electrodes={self.n_electrodes}, "
            f"arc_width={self.arc_width:.4f} rad)"
        )


# ---------------------------------------------------------------------------
# Public function
# ---------------------------------------------------------------------------


def place_electrodes(
    mesh: Mesh,
    n_electrodes: int,
    start_angle: float = 0.0,
) -> ElectrodeConfig:
    """Assign boundary nodes to electrodes using the gap model.

    Electrodes are placed uniformly around the boundary at angles

    .. math::
        \\theta_\\ell = \\text{start\\_angle} + \\frac{2\\pi \\ell}{L},
        \\quad \\ell = 0, 1, \\ldots, L-1.

    For each target angle the *closest* boundary node (by angular distance)
    is selected.  If two electrodes would map to the same node an error is
    raised — the mesh is too coarse for the requested electrode count.

    Parameters
    ----------
    mesh : Mesh
        A :class:`~eitkit.mesh.mesh.Mesh` with a populated ``boundary_nodes``
        array.
    n_electrodes : int
        Number of electrodes $L$.  Must satisfy $L \\leq B$ where $B$ is the
        number of boundary nodes.
    start_angle : float
        Offset (radians) for the first electrode.  Default 0 places the
        first electrode on the positive $x$-axis.

    Returns
    -------
    ElectrodeConfig
    """
    if n_electrodes < 2:
        raise ValueError("n_electrodes must be at least 2.")
    if n_electrodes > mesh.n_boundary_nodes:
        raise ValueError(
            f"Requested {n_electrodes} electrodes but the mesh only has "
            f"{mesh.n_boundary_nodes} boundary nodes.  Refine the mesh "
            f"(smaller h0) or reduce n_electrodes."
        )

    # Angular positions of boundary nodes (in [-π, π])
    bnd_xy = mesh.nodes[mesh.boundary_nodes]
    bnd_angles = np.arctan2(bnd_xy[:, 1], bnd_xy[:, 0])  # shape (B,)

    # Target angles for electrodes
    target_angles = start_angle + 2.0 * np.pi * np.arange(n_electrodes) / n_electrodes
    # Wrap to [-π, π]
    target_angles = (target_angles + np.pi) % (2.0 * np.pi) - np.pi

    # For each target angle find the closest boundary node (angular distance)
    node_indices = np.empty(n_electrodes, dtype=np.int32)
    for ell, theta in enumerate(target_angles):
        ang_diff = _angular_distance(bnd_angles, theta)
        closest_bnd_idx = int(np.argmin(ang_diff))
        node_indices[ell] = mesh.boundary_nodes[closest_bnd_idx]

    # Check uniqueness
    if len(np.unique(node_indices)) != n_electrodes:
        duplicates = n_electrodes - len(np.unique(node_indices))
        raise ValueError(
            f"{duplicates} electrode(s) mapped to the same boundary node.  "
            "Refine the mesh or reduce n_electrodes."
        )

    # Actual angles of selected nodes
    selected_xy = mesh.nodes[node_indices]
    actual_angles = np.arctan2(selected_xy[:, 1], selected_xy[:, 0])

    arc_width = np.pi / n_electrodes  # half the inter-electrode spacing

    return ElectrodeConfig(
        node_indices=node_indices,
        angles=actual_angles,
        arc_width=arc_width,
        n_electrodes=n_electrodes,
    )


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------


def _angular_distance(
    angles: NDArray[np.float64],
    target: float,
) -> NDArray[np.float64]:
    """Minimum angular distance (in [0, π]) from each angle to *target*."""
    diff = np.abs(angles - target)
    return np.minimum(diff, 2.0 * np.pi - diff)
