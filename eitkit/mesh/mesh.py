r"""
eitkit.mesh.mesh
================
Core mesh data container for 2-D triangular meshes.

The ``Mesh`` class is an immutable dataclass that carries every geometric
quantity needed by the FEM assembler and the inverse solvers.  It is the
single data contract that connects the mesh generator, the forward solver,
and the reconstruction algorithms.

Attributes
----------
nodes : ndarray, shape (N, 2)
    $(x, y)$ coordinates of the $N$ mesh nodes.
elements : ndarray, shape (M, 3)
    Integer indices into ``nodes`` for each of the $M$ triangular elements.
    Winding order is counter-clockwise (positive area convention).
boundary_nodes : ndarray, shape (B,)
    Indices of the $B$ nodes that lie on $\partial\Omega$, sorted by
    counter-clockwise angle from the positive $x$-axis.
areas : ndarray, shape (M,)
    Signed area of each element computed from the node coordinates.
    Positive values confirm counter-clockwise winding.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    pass  # keep imports lightweight at runtime


@dataclass(frozen=True)
class Mesh:
    """Immutable container for a 2-D triangular mesh.

    Parameters
    ----------
    nodes : array_like, shape (N, 2)
        Node coordinates.
    elements : array_like, shape (M, 3)
        Triangle connectivity (0-based node indices, CCW winding).
    boundary_nodes : array_like, shape (B,)
        Boundary node indices sorted CCW by angle.
    areas : array_like, shape (M,), optional
        Pre-computed element areas.  If not supplied they are computed
        automatically from *nodes* and *elements*.
    """

    nodes: NDArray[np.float64]
    elements: NDArray[np.int32]
    boundary_nodes: NDArray[np.int32]
    areas: NDArray[np.float64] = field(default=None, repr=False)  # type: ignore[assignment]

    # ------------------------------------------------------------------
    # Post-init: convert to canonical arrays and compute areas if needed
    # ------------------------------------------------------------------
    def __post_init__(self) -> None:
        # Use object.__setattr__ because the dataclass is frozen.
        object.__setattr__(
            self,
            "nodes",
            np.asarray(self.nodes, dtype=np.float64),
        )
        object.__setattr__(
            self,
            "elements",
            np.asarray(self.elements, dtype=np.int32),
        )
        object.__setattr__(
            self,
            "boundary_nodes",
            np.asarray(self.boundary_nodes, dtype=np.int32),
        )

        if self.areas is None:
            object.__setattr__(self, "areas", _compute_areas(self.nodes, self.elements))
        else:
            object.__setattr__(
                self,
                "areas",
                np.asarray(self.areas, dtype=np.float64),
            )

        self._validate()

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------
    def _validate(self) -> None:
        if self.nodes.ndim != 2 or self.nodes.shape[1] != 2:
            raise ValueError(
                f"nodes must have shape (N, 2), got {self.nodes.shape}."
            )
        if self.elements.ndim != 2 or self.elements.shape[1] != 3:
            raise ValueError(
                f"elements must have shape (M, 3), got {self.elements.shape}."
            )
        if self.boundary_nodes.ndim != 1:
            raise ValueError("boundary_nodes must be a 1-D array.")
        neg = np.sum(self.areas < 0)
        if neg > 0:
            raise ValueError(
                f"{neg} element(s) have negative area (CW winding). "
                "Ensure elements are ordered counter-clockwise."
            )

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------
    @property
    def n_nodes(self) -> int:
        """Number of mesh nodes."""
        return int(self.nodes.shape[0])

    @property
    def n_elements(self) -> int:
        """Number of triangular elements."""
        return int(self.elements.shape[0])

    @property
    def n_boundary_nodes(self) -> int:
        """Number of boundary nodes."""
        return int(self.boundary_nodes.shape[0])

    @property
    def h_mean(self) -> float:
        """Mean element edge length (proxy for mesh resolution)."""
        return float(np.sqrt(np.mean(self.areas) * 2.0))

    def interior_nodes(self) -> NDArray[np.int32]:
        """Return indices of all interior (non-boundary) nodes."""
        all_nodes = np.arange(self.n_nodes, dtype=np.int32)
        mask = np.ones(self.n_nodes, dtype=bool)
        mask[self.boundary_nodes] = False
        return all_nodes[mask]

    def centroid(self) -> NDArray[np.float64]:
        """Return the (x, y) centroid of each element, shape (M, 2)."""
        return self.nodes[self.elements].mean(axis=1)

    def __repr__(self) -> str:
        return (
            f"Mesh(n_nodes={self.n_nodes}, n_elements={self.n_elements}, "
            f"n_boundary_nodes={self.n_boundary_nodes}, "
            f"h_mean={self.h_mean:.4f})"
        )


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------

def _compute_areas(
    nodes: NDArray[np.float64],
    elements: NDArray[np.int32],
) -> NDArray[np.float64]:
    """Compute signed element areas using the cross-product formula.

    For a triangle with vertices $p_0, p_1, p_2$:

    .. math::
        A = \\frac{1}{2} \\bigl[(x_1 - x_0)(y_2 - y_0)
                                - (x_2 - x_0)(y_1 - y_0)\\bigr]

    Positive values correspond to counter-clockwise winding.
    """
    p0 = nodes[elements[:, 0]]
    p1 = nodes[elements[:, 1]]
    p2 = nodes[elements[:, 2]]
    return 0.5 * ((p1[:, 0] - p0[:, 0]) * (p2[:, 1] - p0[:, 1])
                  - (p2[:, 0] - p0[:, 0]) * (p1[:, 1] - p0[:, 1]))
