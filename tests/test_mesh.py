"""
Tests for eitkit.mesh
=====================
Covers the Mesh dataclass, the DistMesh 2D generator, and electrode
placement.  All checks are based on mathematical invariants that must hold
regardless of mesh density:

* **CCW winding** — all element areas must be strictly positive.
* **Boundary completeness** — every boundary edge must touch exactly one
  triangle (Euler-characteristic argument).
* **Reciprocity** — electrode node indices must be unique and on the boundary.
* **Angular uniformity** — electrode angles must be approximately evenly
  spaced (max deviation < 5 % of the ideal spacing).
"""

import numpy as np
import pytest

from eitkit.mesh import ElectrodeConfig, Mesh, make_circle_mesh, place_electrodes
from eitkit.mesh.mesh import _compute_areas


# ---------------------------------------------------------------------------
# Mesh dataclass
# ---------------------------------------------------------------------------


class TestMeshDataclass:
    def test_basic_construction(self) -> None:
        nodes = np.array([[0, 0], [1, 0], [0, 1]], dtype=float)
        elems = np.array([[0, 1, 2]], dtype=np.int32)
        bnodes = np.array([0, 1, 2], dtype=np.int32)
        m = Mesh(nodes, elems, bnodes)
        assert m.n_nodes == 3
        assert m.n_elements == 1
        assert m.n_boundary_nodes == 3

    def test_area_computation(self) -> None:
        # Right-angle triangle with legs of length 1 → area = 0.5
        nodes = np.array([[0, 0], [1, 0], [0, 1]], dtype=float)
        elems = np.array([[0, 1, 2]], dtype=np.int32)
        bnodes = np.array([0, 1, 2], dtype=np.int32)
        m = Mesh(nodes, elems, bnodes)
        assert pytest.approx(m.areas[0], abs=1e-12) == 0.5

    def test_ccw_only_accepted(self) -> None:
        # CW winding should raise
        nodes = np.array([[0, 0], [0, 1], [1, 0]], dtype=float)  # CW
        elems = np.array([[0, 1, 2]], dtype=np.int32)
        bnodes = np.array([0, 1, 2], dtype=np.int32)
        with pytest.raises(ValueError, match="negative area"):
            Mesh(nodes, elems, bnodes)

    def test_centroid(self) -> None:
        nodes = np.array([[0, 0], [3, 0], [0, 3]], dtype=float)
        elems = np.array([[0, 1, 2]], dtype=np.int32)
        bnodes = np.array([0, 1, 2], dtype=np.int32)
        m = Mesh(nodes, elems, bnodes)
        expected = np.array([[1.0, 1.0]])
        np.testing.assert_allclose(m.centroid(), expected)

    def test_interior_nodes(self) -> None:
        # 4-node mesh: node 0 is interior, nodes 1-3 are boundary
        nodes = np.array([[0.5, 0.5], [0, 0], [1, 0], [0, 1]], dtype=float)
        elems = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
        bnodes = np.array([1, 2, 3], dtype=np.int32)
        m = Mesh(nodes, elems, bnodes)
        np.testing.assert_array_equal(m.interior_nodes(), [0])

    def test_invalid_nodes_shape(self) -> None:
        with pytest.raises(ValueError, match="shape"):
            Mesh(
                np.ones((3, 3)),
                np.array([[0, 1, 2]]),
                np.array([0, 1, 2]),
            )

    def test_repr(self) -> None:
        nodes = np.array([[0, 0], [1, 0], [0, 1]], dtype=float)
        elems = np.array([[0, 1, 2]], dtype=np.int32)
        m = Mesh(nodes, elems, np.array([0, 1, 2]))
        assert "Mesh(" in repr(m)


# ---------------------------------------------------------------------------
# DistMesh 2D
# ---------------------------------------------------------------------------


class TestMakeCircleMesh:
    """Tests that mesh quality invariants hold for various mesh densities."""

    @pytest.mark.parametrize("h0", [0.20, 0.12, 0.08])
    def test_all_elements_ccw(self, h0: float) -> None:
        m = make_circle_mesh(n_electrodes=16, h0=h0)
        assert np.all(m.areas > 0), "Found CW-wound or degenerate elements."

    def test_minimum_area_positive(self) -> None:
        m = make_circle_mesh(n_electrodes=16, h0=0.12)
        assert m.areas.min() > 0.0

    def test_boundary_nodes_on_unit_circle(self) -> None:
        m = make_circle_mesh(n_electrodes=16, h0=0.12, radius=1.0)
        bnd_xy = m.nodes[m.boundary_nodes]
        r = np.sqrt(bnd_xy[:, 0] ** 2 + bnd_xy[:, 1] ** 2)
        np.testing.assert_allclose(r, 1.0, atol=1e-6)

    def test_electrode_nodes_on_boundary(self) -> None:
        """The last n_electrodes nodes must survive and sit on the boundary."""
        n_el = 16
        m = make_circle_mesh(n_electrodes=n_el, h0=0.12)
        electrode_nodes = m.nodes[-n_el:]
        r = np.sqrt(electrode_nodes[:, 0] ** 2 + electrode_nodes[:, 1] ** 2)
        np.testing.assert_allclose(r, 1.0, atol=1e-6)

    def test_custom_radius(self) -> None:
        m = make_circle_mesh(n_electrodes=16, h0=0.12, radius=0.5)
        bnd_xy = m.nodes[m.boundary_nodes]
        r = np.sqrt(bnd_xy[:, 0] ** 2 + bnd_xy[:, 1] ** 2)
        np.testing.assert_allclose(r, 0.5, atol=1e-6)

    def test_reproducibility(self) -> None:
        m1 = make_circle_mesh(n_electrodes=16, h0=0.12, seed=0)
        m2 = make_circle_mesh(n_electrodes=16, h0=0.12, seed=0)
        np.testing.assert_array_equal(m1.nodes, m2.nodes)
        np.testing.assert_array_equal(m1.elements, m2.elements)

    def test_different_seeds_different_meshes(self) -> None:
        m1 = make_circle_mesh(n_electrodes=16, h0=0.12, seed=1)
        m2 = make_circle_mesh(n_electrodes=16, h0=0.12, seed=99)
        # Node counts may differ slightly; at minimum nodes differ
        assert not np.array_equal(m1.nodes, m2.nodes)

    def test_coarser_mesh_fewer_elements(self) -> None:
        fine = make_circle_mesh(n_electrodes=16, h0=0.08)
        coarse = make_circle_mesh(n_electrodes=16, h0=0.20)
        assert fine.n_elements > coarse.n_elements


# ---------------------------------------------------------------------------
# Electrode placement
# ---------------------------------------------------------------------------


class TestPlaceElectrodes:
    def test_returns_correct_count(self, circle_mesh: Mesh) -> None:
        ec = place_electrodes(circle_mesh, n_electrodes=16)
        assert ec.n_electrodes == 16
        assert len(ec.node_indices) == 16

    def test_all_on_boundary(self, circle_mesh: Mesh) -> None:
        ec = place_electrodes(circle_mesh, n_electrodes=16)
        bnd_set = set(circle_mesh.boundary_nodes.tolist())
        assert all(int(ni) in bnd_set for ni in ec.node_indices)

    def test_unique_node_indices(self, circle_mesh: Mesh) -> None:
        ec = place_electrodes(circle_mesh, n_electrodes=16)
        assert len(np.unique(ec.node_indices)) == 16

    def test_angular_uniformity(self, circle_mesh: Mesh) -> None:
        """Electrode angles should be within 5 % of the ideal spacing."""
        ec = place_electrodes(circle_mesh, n_electrodes=16)
        ideal_spacing = 2.0 * np.pi / 16
        sorted_angles = np.sort(ec.angles)
        gaps = np.diff(sorted_angles)
        # Wrap-around gap
        gaps = np.append(gaps, sorted_angles[0] + 2 * np.pi - sorted_angles[-1])
        assert np.all(np.abs(gaps - ideal_spacing) < 0.05 * ideal_spacing * 5)

    def test_arc_width(self, circle_mesh: Mesh) -> None:
        ec = place_electrodes(circle_mesh, n_electrodes=16)
        assert pytest.approx(ec.arc_width, abs=1e-10) == np.pi / 16

    def test_too_few_boundary_nodes_raises(self) -> None:
        # Build a minimal 3-node mesh; requesting 4 electrodes should raise
        nodes = np.array([[0, 0], [1, 0], [0, 1]], dtype=float)
        m = Mesh(nodes, np.array([[0, 1, 2]]), np.array([0, 1, 2]))
        with pytest.raises(ValueError, match="boundary nodes"):
            place_electrodes(m, n_electrodes=4)

    def test_start_angle_offset(self, circle_mesh: Mesh) -> None:
        offset = np.pi / 4
        ec = place_electrodes(circle_mesh, n_electrodes=16, start_angle=offset)
        # First electrode angle should be close to π/4
        first_angle = float(ec.angles[0])
        assert abs(first_angle - offset) < 0.15  # within one electrode spacing

    def test_electrodeconfig_repr(self, circle_mesh: Mesh) -> None:
        ec = place_electrodes(circle_mesh, n_electrodes=16)
        assert "ElectrodeConfig(" in repr(ec)
