"""
Mesh-resolution and convergence checks for the forward model.

On the unit disc with uniform conductivity sigma = 1, a point current +I at
boundary point y_s and -I at y_t produces the potential

    u(x) = -(I / pi) * ( ln|x - y_s| - ln|x - y_t| ) + const,

because d/dn ln|x - y| = 1/2 on the unit circle for every boundary point
x != y, so the boundary flux vanishes away from the electrodes.  The tests
compare FEM electrode voltages with this closed form on refined meshes.
Measured electrodes are never driven, so the voltages are finite and the
FEM error should decrease under refinement.
"""

from __future__ import annotations

from collections import Counter
from functools import cache

import numpy as np
import pytest

from eitkit.forward import assemble_K
from eitkit.forward.gap_model import build_load_vector
from eitkit.forward.solver import pick_ground_node, solve_forward
from eitkit.mesh import make_circle_mesh, place_electrodes
from eitkit.protocol import adjacent_pattern, measurement_pairs

N_ELECTRODES = 16


@cache
def _mesh(h0):
    """Meshes are deterministic for a given h0 and seed, so build each once."""
    return make_circle_mesh(n_electrodes=N_ELECTRODES, h0=h0, seed=42)


def _boundary_edges(mesh):
    """Edges that belong to exactly one triangle."""
    counts = Counter(
        tuple(sorted(edge))
        for tri in mesh.elements
        for edge in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0]))
    )
    return [edge for edge, n in counts.items() if n == 1]


def _exact_potential(x, y_source, y_sink, current=1.0):
    return -(current / np.pi) * (
        np.log(np.linalg.norm(x - y_source, axis=-1))
        - np.log(np.linalg.norm(x - y_sink, axis=-1))
    )


def _electrode_voltage_error(h0):
    """Mean relative error of FEM electrode voltages against the exact solution."""
    mesh = _mesh(h0)
    ec = place_electrodes(mesh, N_ELECTRODES)
    K = assemble_K(mesh, np.ones(mesh.n_elements))
    ground = pick_ground_node(mesh)
    positions = mesh.nodes[ec.node_indices]
    drives = adjacent_pattern(N_ELECTRODES)
    meas = measurement_pairs(N_ELECTRODES)

    errors = []
    for k, (src, snk) in enumerate(drives):
        u = solve_forward(K, build_load_vector(mesh.n_nodes, ec, (src, snk)), ground)
        rows = meas[meas[:, 0] == k]
        plus = ec.node_indices[rows[:, 1]]
        minus = ec.node_indices[rows[:, 2]]
        exact = _exact_potential(mesh.nodes[plus], positions[src], positions[snk])
        exact -= _exact_potential(mesh.nodes[minus], positions[src], positions[snk])
        fem = u[plus] - u[minus]
        errors.append(np.linalg.norm(fem - exact) / np.linalg.norm(exact))
    return float(np.mean(errors))


class TestBoundaryResolution:
    @pytest.mark.parametrize("h0", [0.2, 0.1, 0.05])
    def test_boundary_edges_no_longer_than_h0(self, h0):
        mesh = _mesh(h0)
        edges = _boundary_edges(mesh)
        lengths = [np.linalg.norm(mesh.nodes[a] - mesh.nodes[b]) for a, b in edges]
        assert max(lengths) <= h0 * (1 + 1e-12)
        assert len(edges) >= 2 * np.pi / h0

    @pytest.mark.parametrize("h0", [0.2, 0.1, 0.05])
    def test_flagged_boundary_matches_true_boundary(self, h0):
        mesh = _mesh(h0)
        true_boundary = np.unique(np.array(_boundary_edges(mesh)))
        np.testing.assert_array_equal(np.sort(mesh.boundary_nodes), true_boundary)

    def test_area_converges_to_disc_area(self):
        areas = [_mesh(h0).areas.sum() for h0 in (0.2, 0.1, 0.05)]
        deficits = np.pi - np.array(areas)
        assert np.all(deficits > 0)
        assert np.all(np.diff(deficits) < 0)
        assert deficits[-1] / np.pi < 1e-3

    def test_electrodes_at_exact_angles(self):
        mesh = _mesh(0.07)
        ec = place_electrodes(mesh, N_ELECTRODES)
        xy = mesh.nodes[ec.node_indices]
        target = 2 * np.pi * np.arange(N_ELECTRODES) / N_ELECTRODES
        np.testing.assert_allclose(np.cos(target), xy[:, 0], atol=1e-12)
        np.testing.assert_allclose(np.sin(target), xy[:, 1], atol=1e-12)


class TestForwardConvergence:
    def test_electrode_voltages_converge_to_exact_solution(self):
        errors = [_electrode_voltage_error(h0) for h0 in (0.14, 0.1, 0.07)]
        assert errors[0] > errors[1] > errors[2]
        assert errors[-1] < 0.01
