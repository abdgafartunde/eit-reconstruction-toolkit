"""
eitkit — Electrical Impedance Tomography toolkit.

Modules
-------
mesh        Mesh generation and electrode placement (2D).
forward     FEM-based forward solver and Jacobian.
protocol    Stimulation patterns, measurement pairs, and noise models.
inverse     Reconstruction algorithms (classical, data-driven, hybrid).
utils       Phantoms, visualisation, and metrics.
"""

from importlib.metadata import version, PackageNotFoundError

try:
    __version__: str = version("eitkit")
except PackageNotFoundError:  # running from source without install
    __version__ = "0.1.0.dev0"

__all__ = ["__version__"]
