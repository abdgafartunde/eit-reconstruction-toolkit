"""
Tests for eitkit.inverse.classical — Tikhonov regularisation.

Suite structure
---------------
TestTikhonovSolve      basic correctness, shapes, dtypes, edge cases
TestTikhonovSolverLSQR  solver='lsqr' parity with direct
TestChooseLambda        L-curve heuristic returns valid results
TestInputValidation     ValueError paths
TestIntegration         end-to-end: forward → Jacobian → Tikhonov reconstruction
"""

from __future__ import annotations

import numpy as np
import pytest

from eitkit.inverse import choose_lambda, tikhonov_solve
from eitkit.inverse.classical import tikhonov_solve as tikhonov_solve_classical

# ── fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def rng():
    return np.random.default_rng(42)


@pytest.fixture(scope="module")
def small_system(rng):
    """Small (P=40, E=20) over-determined system with known ground truth."""
    E, P = 20, 40
    J = rng.standard_normal((P, E))
    ds_true = rng.standard_normal(E)
    dV = J @ ds_true + 1e-4 * rng.standard_normal(P)  # low noise
    return J, dV, ds_true


@pytest.fixture(scope="module")
def square_system(rng):
    """Square (P=E=30) system."""
    n = 30
    J = rng.standard_normal((n, n))
    dV = rng.standard_normal(n)
    return J, dV


# ── TestTikhonovSolve ─────────────────────────────────────────────────────────


class TestTikhonovSolve:
    def test_output_shape(self, small_system):
        J, dV, _ = small_system
        ds = tikhonov_solve(J, dV, lambda_=1e-3)
        assert ds.shape == (J.shape[1],)

    def test_output_dtype(self, small_system):
        J, dV, _ = small_system
        ds = tikhonov_solve(J, dV, lambda_=1e-3)
        assert ds.dtype == np.float64

    def test_zero_lambda_is_lstsq(self, small_system):
        """lambda_=0 should equal the ordinary least-squares solution."""
        J, dV, _ = small_system
        ds_tik = tikhonov_solve(J, dV, lambda_=0.0)
        ds_lstsq, _, _, _ = np.linalg.lstsq(J, dV, rcond=None)
        np.testing.assert_allclose(ds_tik, ds_lstsq, rtol=1e-6)

    def test_large_lambda_shrinks_to_zero(self, small_system):
        """Very large lambda drives delta_sigma toward zero."""
        J, dV, _ = small_system
        ds = tikhonov_solve(J, dV, lambda_=1e12)
        assert np.linalg.norm(ds) < 1e-3

    def test_recovers_ground_truth_low_noise(self, small_system):
        """With low noise and small lambda, should recover signal reasonably."""
        J, dV, ds_true = small_system
        ds = tikhonov_solve(J, dV, lambda_=1e-6)
        rel_err = np.linalg.norm(ds - ds_true) / np.linalg.norm(ds_true)
        assert rel_err < 0.5  # loose but meaningful bound

    def test_identity_jacobian(self):
        """J=I: solution should be dV / (1 + lambda)."""
        E = 10
        J = np.eye(E)
        dV = np.ones(E)
        lam = 0.5
        ds = tikhonov_solve(J, dV, lambda_=lam)
        np.testing.assert_allclose(ds, dV / (1.0 + lam), rtol=1e-12)

    def test_float32_input_coerced(self, small_system):
        """float32 inputs should be silently coerced to float64."""
        J, dV, _ = small_system
        ds = tikhonov_solve(J.astype(np.float32), dV.astype(np.float32), lambda_=1e-3)
        assert ds.dtype == np.float64

    def test_import_path_classical(self, small_system):
        """tikhonov_solve importable from eitkit.inverse.classical directly."""
        J, dV, _ = small_system
        ds1 = tikhonov_solve(J, dV, lambda_=1e-3)
        ds2 = tikhonov_solve_classical(J, dV, lambda_=1e-3)
        np.testing.assert_array_equal(ds1, ds2)

    def test_square_system(self, square_system):
        """Works on square (P == E) systems."""
        J, dV = square_system
        ds = tikhonov_solve(J, dV, lambda_=1.0)
        assert ds.shape == (J.shape[1],)
        assert np.all(np.isfinite(ds))

    def test_underdetermined_system(self, rng):
        """Works on underdetermined (P < E) systems."""
        J = rng.standard_normal((10, 50))
        dV = rng.standard_normal(10)
        ds = tikhonov_solve(J, dV, lambda_=1e-2)
        assert ds.shape == (50,)
        assert np.all(np.isfinite(ds))


# ── TestTikhonovSolverLSQR ────────────────────────────────────────────────────


class TestTikhonovSolverLSQR:
    def test_lsqr_matches_direct(self, small_system):
        """lsqr and direct should agree to ~ 1e-6 relative tolerance."""
        J, dV, _ = small_system
        lam = 1e-3
        ds_direct = tikhonov_solve(J, dV, lambda_=lam, solver="direct")
        ds_lsqr = tikhonov_solve(J, dV, lambda_=lam, solver="lsqr")
        np.testing.assert_allclose(ds_lsqr, ds_direct, rtol=1e-5, atol=1e-10)

    def test_lsqr_output_shape(self, small_system):
        J, dV, _ = small_system
        ds = tikhonov_solve(J, dV, lambda_=1e-3, solver="lsqr")
        assert ds.shape == (J.shape[1],)

    def test_lsqr_output_dtype(self, small_system):
        J, dV, _ = small_system
        ds = tikhonov_solve(J, dV, lambda_=1e-3, solver="lsqr")
        assert ds.dtype == np.float64


# ── TestChooseLambda ──────────────────────────────────────────────────────────


class TestChooseLambda:
    def test_returns_tuple_of_three(self, small_system):
        J, dV, _ = small_system
        result = choose_lambda(J, dV)
        assert len(result) == 3

    def test_lambda_opt_positive(self, small_system):
        J, dV, _ = small_system
        lam_opt, _, _ = choose_lambda(J, dV)
        assert lam_opt > 0.0

    def test_lambda_opt_in_candidate_range(self, small_system):
        J, dV, _ = small_system
        lambdas = np.logspace(-4, 2, 30)
        lam_opt, _, _ = choose_lambda(J, dV, lambdas=lambdas)
        assert lambdas[0] <= lam_opt <= lambdas[-1]

    def test_curve_lengths_match_input(self, small_system):
        J, dV, _ = small_system
        n = 20
        lambdas = np.logspace(-4, 2, n)
        _, residuals, sol_norms = choose_lambda(J, dV, lambdas=lambdas)
        assert residuals.shape == (n,)
        assert sol_norms.shape == (n,)

    def test_residuals_decrease_with_lambda(self, small_system):
        """Larger lambda → larger residual (less data fit)."""
        J, dV, _ = small_system
        lambdas = np.logspace(-6, 4, 40)
        _, residuals, _ = choose_lambda(J, dV, lambdas=lambdas)
        # residuals should be broadly increasing with lambda
        # check that max is near the large-lambda end
        assert np.argmax(residuals) > len(residuals) // 2

    def test_solution_norms_decrease_with_lambda(self, small_system):
        """Larger lambda → smaller solution norm (more regularised)."""
        J, dV, _ = small_system
        lambdas = np.logspace(-6, 4, 40)
        _, _, sol_norms = choose_lambda(J, dV, lambdas=lambdas)
        # solution norms should broadly decrease with lambda
        assert np.argmin(sol_norms) > len(sol_norms) // 2

    def test_n_points_controls_default_grid(self, small_system):
        J, dV, _ = small_system
        n = 15
        _, r, s = choose_lambda(J, dV, n_points=n)
        assert len(r) == n
        assert len(s) == n


# ── TestInputValidation ───────────────────────────────────────────────────────


class TestInputValidation:
    def test_J_not_2d_raises(self):
        with pytest.raises(ValueError, match="2-D"):
            tikhonov_solve(np.ones(10), np.ones(10), lambda_=1e-3)

    def test_dV_not_1d_raises(self):
        with pytest.raises(ValueError, match="1-D"):
            tikhonov_solve(np.ones((10, 5)), np.ones((10, 1)), lambda_=1e-3)

    def test_dV_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="does not match"):
            tikhonov_solve(np.ones((10, 5)), np.ones(8), lambda_=1e-3)

    def test_negative_lambda_raises(self):
        with pytest.raises(ValueError, match="non-negative"):
            tikhonov_solve(np.ones((10, 5)), np.ones(10), lambda_=-1.0)

    def test_nan_lambda_raises(self):
        with pytest.raises(ValueError, match="non-negative"):
            tikhonov_solve(np.ones((10, 5)), np.ones(10), lambda_=float("nan"))

    def test_inf_lambda_raises(self):
        with pytest.raises(ValueError, match="non-negative"):
            tikhonov_solve(np.ones((10, 5)), np.ones(10), lambda_=float("inf"))

    def test_unknown_solver_raises(self):
        with pytest.raises(ValueError, match="Unknown solver"):
            tikhonov_solve(np.ones((10, 5)), np.ones(10), lambda_=1.0, solver="magic")


# ── TestIntegration ───────────────────────────────────────────────────────────


class TestIntegration:
    """End-to-end tests using the full eitkit forward stack."""

    @pytest.fixture(scope="class")
    def forward_data(self, circle_mesh, electrodes):
        """Build J and dV from the shared session-scoped mesh/electrodes."""
        from eitkit.forward import compute_jacobian, simulate
        from eitkit.protocol import adjacent_pattern, measurement_pairs
        from eitkit.utils import make_phantom

        drive_pairs = adjacent_pattern(16)
        meas_pairs = measurement_pairs(16)
        sigma_ref = np.ones(circle_mesh.n_elements)
        sigma_inc = make_phantom(
            circle_mesh,
            [
                {"shape": "circle", "cx": 0.3, "cy": 0.0, "r": 0.2, "sigma": 2.0},
            ],
        )
        dV = simulate(circle_mesh, electrodes, sigma_inc, drive_pairs, meas_pairs)
        J = compute_jacobian(
            circle_mesh, electrodes, sigma_ref, drive_pairs, meas_pairs
        )
        return J, dV, sigma_inc, sigma_ref

    def test_reconstruction_shape(self, forward_data, circle_mesh):
        J, dV, _, _ = forward_data
        ds = tikhonov_solve(J, dV, lambda_=1e-3)
        assert ds.shape == (circle_mesh.n_elements,)

    def test_reconstruction_finite(self, forward_data):
        J, dV, _, _ = forward_data
        ds = tikhonov_solve(J, dV, lambda_=1e-3)
        assert np.all(np.isfinite(ds))

    def test_residual_norm_smaller_than_dV_norm(self, forward_data):
        """The Tikhonov fit should not be worse than dV = 0."""
        J, dV, _, _ = forward_data
        ds = tikhonov_solve(J, dV, lambda_=1e-4)
        residual = np.linalg.norm(J @ ds - dV)
        assert residual < np.linalg.norm(dV)

    def test_larger_lambda_gives_larger_residual(self, forward_data):
        J, dV, _, _ = forward_data
        ds_small = tikhonov_solve(J, dV, lambda_=1e-6)
        ds_large = tikhonov_solve(J, dV, lambda_=1e0)
        r_small = np.linalg.norm(J @ ds_small - dV)
        r_large = np.linalg.norm(J @ ds_large - dV)
        assert r_large > r_small

    def test_l_curve_returns_positive_lambda(self, forward_data):
        J, dV, _, _ = forward_data
        lam_opt, _, _ = choose_lambda(J, dV, n_points=20)
        assert lam_opt > 0.0
