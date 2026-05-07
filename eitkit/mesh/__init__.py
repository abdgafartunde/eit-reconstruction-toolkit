"""
eitkit.mesh — Mesh generation and electrode placement (2D).

Public API
----------
Mesh                   Triangular mesh data container.
make_circle_mesh       Generate a quality 2D triangular mesh on a unit disc.
place_electrodes       Distribute L electrodes uniformly on the boundary.
"""

from eitkit.mesh.mesh import Mesh
from eitkit.mesh.distmesh2d import make_circle_mesh
from eitkit.mesh.electrode_placement import ElectrodeConfig, place_electrodes

__all__ = ["Mesh", "make_circle_mesh", "ElectrodeConfig", "place_electrodes"]
