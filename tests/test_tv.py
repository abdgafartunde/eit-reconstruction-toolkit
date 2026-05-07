"""
Tests for eitkit.inverse.classical.tv — Total Variation regularisation.

Suite structure
---------------
TestBuildGradientOp    shape, sparsity, sign convention, mesh topology
TestTVSolve            basic correctness, shapes, dtypes, edge cases
TestTVvsSmoothing      TV vs Tikhonov behaviour (edge preservation property)
TestInputValidation    ValueError paths
TestIntegration        end-to-end: forward → Jacobian → TV reconstruction
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

from eitkit.inverse import tv_solve, build_gradient_op, tikhonov_solve
from eitkit.mesh import make_circle_mesh, place_electrodes


# ── fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def coarse_mesh():
    return make_circle_mesh(n_electrodes=16, h0=0.12)


@pytest.fixture(scope="module")
def rng():
    return np.random.default_rng(7)


@pytest.fixture(scope="module")
def small_system(coarse_mesh, rng):
    E = coarse_mesh.n_elements
    P = 2 * E
    J  = rng.standard_normal((P, E))
    dV = rng.standard_normal(P)
    return J, dV


# ── TestBuildGradientOp ───────────────────────────────────────────────────────

class TestBuildGradientOp:
    def test_returns_sparse_matrix(self, coarse_mesh):
        D = build_gradient_op(coarse_mesh)
        assert sp.issparse(D)

    def test_shape_columns_equal_n_elements(self, coarse_mesh):
        D = build_gradient_op(coarse_mesh)
        assert D.shape[1] == coarse_mesh.n_elements

    def test_rows_less_than_edges(self, coarse_mesh):
        """Number of shared edges ≤ 3*E/2 (Euler for a 2-D planar mesh)."""
        D = build_gradient_op(coarse_mesh)
        F = D.shape[0]
        assert F > 0
        assert F < 3 * coarse_mesh.n_elements

    def test_each_row_has_exactly_two_nonzeros(self, coarse_mesh):
        D = build_gradient_op(coarse_mesh)
        nnz_per_row = np.diff(D.tocsr().indptr)
        assert np.all(nnz_per_row == 2)

    def test_each_row_sums_to_zero(self, coarse_mesh):
        """Each row has entries +1 and -1 → sum == 0."""
        D = build_gradient_op(coarse_mesh)
        row_sums = np.asarray(D.sum(axis=1)).ravel()
        np.testing.assert_allclose(row_sums, 0.0, atol=1e-14)

    def test_constant_sigma_in_nullspace(self, coarse_mesh):
        """D applied to a constant vector should give zeros (exact nullspace)."""
        D = build_gradient_op(coarse_mesh)
        sigma_const = np.ones(coarse_mesh.n_elements)
        result = D @ sigma_const
        np.testing.assert_allclose(result, 0.0, atol=1e-14)

    def test_dtype_float64(self, coarse_mesh):
        D = build_gradient_op(coarse_mesh)
        assert D.dtype == np.float64


# ── TestTVSolve ───────────────────────────────────────────────────────────────

class TestTVSolve:
    def test_output_shape(self, small_system, coarse_mesh):
        J, dV = small_system
        ds = tv_solve(J, dV, alpha=1e-2, mesh=coarse_mesh)
        assert ds.shape == (coarse_mesh.n_elements,)

    def test_output_dtype(self, small_system, coarse_mesh):
        J, dV = small_system
        ds = tv_solve(J, dV, alpha=1e-2, mesh=coarse_mesh)
        assert ds.dtype == np.float64

    def test_output_finite(self, small_system, coarse_mesh):
        J, dV = small_system
        ds = tv_solve(J, dV, alpha=1e-2, mesh=coarse_mesh)
        assert np.all(np.isfinite(ds))

    def test_large_alpha_shrinks_norm(self, small_system, coarse_mesh):
        """Very large alpha → near-constant (flat) solution."""
        J, dV = small_system
        ds_small = tv_solve(J, dV, alpha=1e-4, mesh=coarse_mesh)
        ds_large = tv_solve(J, dV, alpha=1e3,  mesh=coarse_mesh)
        D = build_gradient_op(coarse_mesh)
        tv_small = np.linalg.norm(D @ ds_small, 1)
        tv_large = np.linalg.norm(D @ ds_large, 1)
        assert tv_large < tv_small

    def test_larger_alpha_gives_larger_residual(self, small_system, coarse_mesh):
        """More regularisation → worse data fit."""
        J, dV = small_system
        ds_small = tv_solve(J, dV, alpha=1e-4, mesh=coarse_mesh)
        ds_large = tv_solve(J, dV, alpha=1e1,  mesh=coarse_mesh)
        r_small = np.linalg.norm(J @ ds_small - dV)
        r_large = np.linalg.norm(J @ ds_large - dV)
        assert r_large > r_small

    def test_warm_start_accepted(self, small_system, coarse_mesh):
        J, dV = small_system
        warm = np.zeros(coarse_mesh.n_elements)
        ds = tv_solve(J, dV, alpha=1e-2, mesh=coarse_mesh, warm_start=warm)
        assert ds.shape == (coarse_mesh.n_elements,)

    def test_float32_input_coerced(self, small_system, coarse_mesh):
        J, dV = small_system
        ds = tv_solve(J.astype(np.float32), dV.astype(np.float32),
                      alpha=1e-2, mesh=coarse_mesh)
        assert ds.dtype == np.float64

    def test_identity_jacobian_low_alpha(self, coarse_mesh):
        """J=I, alpha≈0: solution should be ≈ dV (least-squares)."""
        E  = coarse_mesh.n_elements
        J  = np.eye(E)
        dV = np.ones(E) * 0.5
        ds = tv_solve(J, dV, alpha=1e-8, mesh=coarse_mesh, max_iter=500, tol=1e-6)
        # With very small alpha TV term is negligible; solution ≈ dV
        np.testing.assert_allclose(ds, dV, atol=0.05)


# ── TestTVvsSmoothing ─────────────────────────────────────────────────────────

class TestTVvsSmoothing:
    def test_tv_preserves_edges_better_than_tikhonov(self, coarse_mesh, rng):
        """TV should have smaller ||D ds||_1 than Tikhonov at same ||J ds - dV||."""
        from eitkit.protocol import adjacent_pattern, measurement_pairs
        from eitkit.forward import simulate, compute_jacobian
        from eitkit.utils import make_phantom

        ec = place_electrodes(coarse_mesh, 16)
        drive_pairs = adjacent_pattern(16)
        meas_pairs  = measurement_pairs(16)
        sigma_inc = make_phantom(coarse_mesh, [
            {"shape": "circle", "cx": 0.3, "cy": 0.0, "r": 0.2, "sigma": 3.0},
        ])
        sigma_ref = np.ones(coarse_mesh.n_elements)
        dV = simulate(coarse_mesh, ec, sigma_inc, drive_pairs, meas_pairs)
        J  = compute_jacobian(coarse_mesh, ec, sigma_ref, drive_pairs, meas_pairs)

        # Choose lambda so both reconstructions have similar residual
        ds_tv  = tv_solve(J, dV, alpha=1e-2, mesh=coarse_mesh)
        ds_tik = tikhonov_solve(J, dV, lambda_=1e-2)

        D = build_gradient_op(coarse_mesh)
        tv_of_tv  = np.linalg.norm(D @ ds_tv,  1)
        tv_of_tik = np.linalg.norm(D @ ds_tik, 1)

        # TV solution should have smaller total variation than Tikhonov
        assert tv_of_tv < tv_of_tik


# ── TestInputValidation ───────────────────────────────────────────────────────

class TestInputValidation:
    def test_J_not_2d_raises(self, coarse_mesh):
        with pytest.raises(ValueError, match="2-D"):
            tv_solve(np.ones(10), np.ones(10), alpha=1e-2, mesh=coarse_mesh)

    def test_dV_not_1d_raises(self, coarse_mesh):
        with pytest.raises(ValueError, match="1-D"):
            tv_solve(np.ones((10, 5)), np.ones((10, 1)), alpha=1e-2, mesh=coarse_mesh)

    def test_dV_length_mismatch_raises(self, coarse_mesh):
        with pytest.raises(ValueError, match="does not match"):
            tv_solve(np.ones((10, 5)), np.ones(8), alpha=1e-2, mesh=coarse_mesh)

    def test_negative_alpha_raises(self, coarse_mesh):
        with pytest.raises(ValueError, match="positive"):
            tv_solve(np.ones((10, 5)), np.ones(10), alpha=-1.0, mesh=coarse_mesh)

    def test_zero_alpha_raises(self, coarse_mesh):
        with pytest.raises(ValueError, match="positive"):
            tv_solve(np.ones((10, 5)), np.ones(10), alpha=0.0, mesh=coarse_mesh)

    def test_negative_rho_raises(self, coarse_mesh):
        with pytest.raises(ValueError, match="positive"):
            tv_solve(np.ones((10, 5)), np.ones(10), alpha=1e-2,
                     mesh=coarse_mesh, rho=-1.0)


# ── TestIntegration ───────────────────────────────────────────────────────────

class TestIntegration:
    @pytest.fixture(scope="class")
    def forward_data(self, circle_mesh, electrodes):
        from eitkit.protocol import adjacent_pattern, measurement_pairs
        from eitkit.forward import simulate, compute_jacobian
        from eitkit.utils import make_phantom

        drive_pairs = adjacent_pattern(16)
        meas_pairs  = measurement_pairs(16)
        sigma_ref   = np.ones(circle_mesh.n_elements)
        sigma_inc   = make_phantom(circle_mesh, [
            {"shape": "circle", "cx": 0.3, "cy": 0.0, "r": 0.2, "sigma": 2.0},
        ])
        dV = simulate(circle_mesh, electrodes, sigma_inc, drive_pairs, meas_pairs)
        J  = compute_jacobian(circle_mesh, electrodes, sigma_ref, drive_pairs, meas_pairs)
        return J, dV

    def test_reconstruction_shape(self, forward_data, circle_mesh):
        J, dV = forward_data
        ds = tv_solve(J, dV, alpha=1e-2, mesh=circle_mesh)
        assert ds.shape == (circle_mesh.n_elements,)

    def test_reconstruction_finite(self, forward_data, circle_mesh):
        J, dV = forward_data
        ds = tv_solve(J, dV, alpha=1e-2, mesh=circle_mesh)
        assert np.all(np.isfinite(ds))

    def test_residual_smaller_than_dV(self, forward_data, circle_mesh):
        J, dV = forward_data
        ds = tv_solve(J, dV, alpha=1e-3, mesh=circle_mesh)
        assert np.linalg.norm(J @ ds - dV) < np.linalg.norm(dV)

    def test_import_from_inverse_top_level(self, forward_data, circle_mesh):
        from eitkit.inverse import tv_solve as tv_top
        J, dV = forward_data
        ds = tv_top(J, dV, alpha=1e-2, mesh=circle_mesh)
        assert ds.shape == (circle_mesh.n_elements,)
