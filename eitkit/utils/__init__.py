"""
eitkit.utils — Utility functions for EIT.

Public API
----------
make_phantom        Create a synthetic conductivity phantom on a mesh.
plot_mesh           Visualise a 2D triangular mesh.
plot_conductivity   Colour-map a conductivity distribution on a mesh.
"""

from eitkit.utils.phantoms import make_phantom
from eitkit.utils.visualisation import plot_mesh, plot_conductivity, plot_voltages

__all__ = ["make_phantom", "plot_mesh", "plot_conductivity", "plot_voltages"]
