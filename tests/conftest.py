"""
Shared pytest fixtures for eitkit tests.

A small 16-electrode circular mesh is built once per session and reused
across all test modules to keep the suite fast.
"""

import numpy as np
import pytest

from eitkit.mesh import make_circle_mesh, place_electrodes


@pytest.fixture(scope="session")
def circle_mesh():
    """Coarse 16-electrode circular mesh for fast unit tests."""
    return make_circle_mesh(n_electrodes=16, h0=0.12)


@pytest.fixture(scope="session")
def electrodes(circle_mesh):
    """Electrode node indices on the circle mesh boundary."""
    return place_electrodes(circle_mesh, n_electrodes=16)
