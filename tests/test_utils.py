"""
tests/test_utils.py
====================
Unit tests for ``eitkit.utils``:
  - make_phantom  (all 5 shapes + validation)
  - plot_mesh
  - plot_conductivity
  - plot_voltages
"""

from __future__ import annotations

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")  # headless — no GUI window
import matplotlib.pyplot as plt

from eitkit.forward import simulate
from eitkit.protocol import adjacent_pattern, measurement_pairs
from eitkit.utils import make_phantom, plot_conductivity, plot_mesh, plot_voltages

# ---------------------------------------------------------------------------
# make_phantom
# ---------------------------------------------------------------------------


class TestMakePhantom:
    def test_background_only_shape(self, circle_mesh):
        sigma = make_phantom(circle_mesh, [])
        assert sigma.shape == (circle_mesh.n_elements,)

    def test_background_only_dtype(self, circle_mesh):
        sigma = make_phantom(circle_mesh, [])
        assert sigma.dtype == np.float64

    def test_background_only_uniform(self, circle_mesh):
        sigma = make_phantom(circle_mesh, [], sigma_background=2.0)
        np.testing.assert_array_equal(sigma, 2.0)

    def test_invalid_background(self, circle_mesh):
        with pytest.raises(ValueError, match="sigma_background"):
            make_phantom(circle_mesh, [], sigma_background=0.0)

    def test_invalid_background_negative(self, circle_mesh):
        with pytest.raises(ValueError):
            make_phantom(circle_mesh, [], sigma_background=-1.0)

    def test_invalid_inclusion_sigma(self, circle_mesh):
        with pytest.raises((ValueError, KeyError)):
            make_phantom(
                circle_mesh,
                [{"shape": "circle", "cx": 0, "cy": 0, "r": 0.1, "sigma": -1.0}],
            )

    def test_unknown_shape(self, circle_mesh):
        with pytest.raises(ValueError, match="Unknown inclusion shape"):
            make_phantom(
                circle_mesh,
                [{"shape": "hexagon", "cx": 0, "cy": 0, "r": 0.1, "sigma": 2.0}],
            )

    # ── circle ──────────────────────────────────────────────────────────────

    def test_circle_values(self, circle_mesh):
        sigma = make_phantom(
            circle_mesh,
            [{"shape": "circle", "cx": 0.0, "cy": 0.0, "r": 0.5, "sigma": 3.0}],
        )
        uniq = np.unique(sigma)
        assert set(uniq).issubset({1.0, 3.0})

    def test_circle_changes_some_elements(self, circle_mesh):
        sigma = make_phantom(
            circle_mesh,
            [{"shape": "circle", "cx": 0.0, "cy": 0.0, "r": 0.3, "sigma": 5.0}],
        )
        assert np.any(sigma == 5.0)
        assert np.any(sigma == 1.0)  # not all elements inside

    def test_circle_all_inside_small_radius(self, circle_mesh):
        # radius=0 → no element centroid should be exactly at origin
        sigma = make_phantom(
            circle_mesh,
            [{"shape": "circle", "cx": 99.0, "cy": 99.0, "r": 0.001, "sigma": 5.0}],
        )
        assert np.all(sigma == 1.0)  # inclusion outside domain

    # ── ellipse ─────────────────────────────────────────────────────────────

    def test_ellipse_changes_elements(self, circle_mesh):
        sigma = make_phantom(
            circle_mesh,
            [
                {
                    "shape": "ellipse",
                    "cx": 0.0,
                    "cy": 0.0,
                    "a": 0.4,
                    "b": 0.2,
                    "theta": 0.0,
                    "sigma": 2.0,
                }
            ],
        )
        assert np.any(sigma == 2.0)
        assert np.any(sigma == 1.0)

    def test_ellipse_rotation_changes_mask(self, circle_mesh):
        # Rotating an asymmetric ellipse should change which elements are inside
        base = {
            "shape": "ellipse",
            "cx": 0.3,
            "cy": 0.0,
            "a": 0.4,
            "b": 0.1,
            "sigma": 2.0,
        }
        s0 = make_phantom(circle_mesh, [{**base, "theta": 0.0}])
        s1 = make_phantom(circle_mesh, [{**base, "theta": np.pi / 2}])
        assert not np.array_equal(s0, s1)

    # ── rectangle ───────────────────────────────────────────────────────────

    def test_rectangle_changes_elements(self, circle_mesh):
        sigma = make_phantom(
            circle_mesh,
            [
                {
                    "shape": "rectangle",
                    "cx": 0.0,
                    "cy": 0.0,
                    "w": 0.3,
                    "h": 0.2,
                    "theta": 0.0,
                    "sigma": 2.0,
                }
            ],
        )
        assert np.any(sigma == 2.0)
        assert np.any(sigma == 1.0)

    def test_rectangle_rotation_changes_mask(self, circle_mesh):
        base = {
            "shape": "rectangle",
            "cx": 0.2,
            "cy": 0.0,
            "w": 0.3,
            "h": 0.05,
            "sigma": 2.0,
        }
        s0 = make_phantom(circle_mesh, [{**base, "theta": 0.0}])
        s1 = make_phantom(circle_mesh, [{**base, "theta": np.pi / 2}])
        assert not np.array_equal(s0, s1)

    # ── ring ────────────────────────────────────────────────────────────────

    def test_ring_changes_elements(self, circle_mesh):
        sigma = make_phantom(
            circle_mesh,
            [
                {
                    "shape": "ring",
                    "cx": 0.0,
                    "cy": 0.0,
                    "r_inner": 0.2,
                    "r_outer": 0.5,
                    "sigma": 0.5,
                }
            ],
        )
        assert np.any(sigma == 0.5)
        assert np.any(sigma == 1.0)

    def test_ring_hollow_centre(self, circle_mesh):
        # Centroids very near origin should not be inside the ring
        sigma = make_phantom(
            circle_mesh,
            [
                {
                    "shape": "ring",
                    "cx": 0.0,
                    "cy": 0.0,
                    "r_inner": 0.3,
                    "r_outer": 0.6,
                    "sigma": 5.0,
                }
            ],
        )
        centroids = circle_mesh.nodes[circle_mesh.elements].mean(axis=1)
        close = np.sqrt(centroids[:, 0] ** 2 + centroids[:, 1] ** 2) < 0.1
        if close.any():
            assert np.all(sigma[close] == 1.0)

    # ── triangle ────────────────────────────────────────────────────────────

    def test_triangle_changes_elements(self, circle_mesh):
        sigma = make_phantom(
            circle_mesh,
            [
                {
                    "shape": "triangle",
                    "cx": 0.0,
                    "cy": 0.0,
                    "side": 0.5,
                    "theta": 0.0,
                    "sigma": 4.0,
                }
            ],
        )
        assert np.any(sigma == 4.0)
        assert np.any(sigma == 1.0)

    # ── overlapping inclusions ───────────────────────────────────────────────

    def test_later_inclusion_overwrites(self, circle_mesh):
        # Place two concentric circles: outer first, inner second
        sigma = make_phantom(
            circle_mesh,
            [
                {"shape": "circle", "cx": 0.0, "cy": 0.0, "r": 0.4, "sigma": 2.0},
                {"shape": "circle", "cx": 0.0, "cy": 0.0, "r": 0.15, "sigma": 5.0},
            ],
        )
        centroids = circle_mesh.nodes[circle_mesh.elements].mean(axis=1)
        dist2 = centroids[:, 0] ** 2 + centroids[:, 1] ** 2
        inner = dist2 < 0.15**2
        if inner.any():
            assert np.all(sigma[inner] == 5.0)


# ---------------------------------------------------------------------------
# Visualisation helpers — smoke tests (no GUI)
# ---------------------------------------------------------------------------


class TestPlotMesh:
    def test_returns_fig_ax(self, circle_mesh):
        fig, ax = plot_mesh(circle_mesh)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_with_electrodes(self, circle_mesh, electrodes):
        fig, ax = plot_mesh(circle_mesh, elec_config=electrodes)
        plt.close(fig)

    def test_into_existing_ax(self, circle_mesh):
        fig0, ax0 = plt.subplots()
        fig1, ax1 = plot_mesh(circle_mesh, ax=ax0)
        assert ax1 is ax0
        assert fig1 is fig0
        plt.close(fig0)


class TestPlotConductivity:
    def test_returns_fig_ax(self, circle_mesh):
        sigma = np.ones(circle_mesh.n_elements)
        fig, ax = plot_conductivity(circle_mesh, sigma)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_wrong_sigma_shape_raises(self, circle_mesh):
        bad = np.ones(5)
        with pytest.raises(ValueError, match="sigma must have shape"):
            plot_conductivity(circle_mesh, bad)

    def test_custom_cmap(self, circle_mesh):
        sigma = np.ones(circle_mesh.n_elements)
        fig, ax = plot_conductivity(circle_mesh, sigma, cmap="viridis")
        plt.close(fig)

    def test_into_existing_ax(self, circle_mesh):
        fig0, ax0 = plt.subplots()
        sigma = np.ones(circle_mesh.n_elements)
        fig1, ax1 = plot_conductivity(circle_mesh, sigma, ax=ax0)
        assert ax1 is ax0
        plt.close(fig0)


class TestPlotVoltages:
    @pytest.fixture(scope="class")
    def dv(self, circle_mesh, electrodes):
        drive = adjacent_pattern(16)
        meas = measurement_pairs(16)
        sigma = make_phantom(
            circle_mesh,
            [{"shape": "circle", "cx": 0.0, "cy": 0.5, "r": 0.2, "sigma": 3.0}],
        )
        return simulate(circle_mesh, electrodes, sigma, drive, meas, sigma0=1.0)

    def test_returns_fig_ax(self, circle_mesh, electrodes, dv):
        meas = measurement_pairs(16)
        fig, ax = plot_voltages(meas, dv)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_with_highlight(self, circle_mesh, electrodes, dv):
        meas = measurement_pairs(16)
        fig, ax = plot_voltages(meas, dv, highlight_drive=3)
        plt.close(fig)

    def test_into_existing_ax(self, circle_mesh, electrodes, dv):
        meas = measurement_pairs(16)
        fig0, ax0 = plt.subplots()
        fig1, ax1 = plot_voltages(meas, dv, ax=ax0)
        assert ax1 is ax0
        plt.close(fig0)
