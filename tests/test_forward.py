"""
tests/test_forward.py
=====================
Unit tests for ``eitkit.forward``:
  - assemble_K
  - simulate
  - compute_jacobian
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

from eitkit.forward import assemble_K, compute_jacobian, simulate
from eitkit.protocol import adjacent_pattern, measurement_pairs

# ---------------------------------------------------------------------------
# assemble_K
# ---------------------------------------------------------------------------


class TestAssembleK:
    def test_shape(self, circle_mesh):
        N = circle_mesh.n_nodes
        sigma = np.ones(circle_mesh.n_elements)
        K = assemble_K(circle_mesh, sigma)
        assert K.shape == (N, N)

    def test_is_sparse_csr(self, circle_mesh):
        sigma = np.ones(circle_mesh.n_elements)
        K = assemble_K(circle_mesh, sigma)
        assert sp.issparse(K)
        assert K.format == "csr"

    def test_symmetric(self, circle_mesh):
        sigma = np.ones(circle_mesh.n_elements)
        K = assemble_K(circle_mesh, sigma)
        diff = (K - K.T).toarray()
        np.testing.assert_allclose(diff, 0, atol=1e-12)

    def test_row_sums_zero(self, circle_mesh):
        # FEM stiffness matrix: every row sums to zero (Neumann property)
        sigma = np.ones(circle_mesh.n_elements)
        K = assemble_K(circle_mesh, sigma)
        row_sums = np.array(K.sum(axis=1)).ravel()
        np.testing.assert_allclose(row_sums, 0, atol=1e-10)

    def test_diagonal_positive(self, circle_mesh):
        sigma = np.ones(circle_mesh.n_elements)
        K = assemble_K(circle_mesh, sigma)
        assert np.all(K.diagonal() > 0)

    def test_scales_linearly_with_uniform_sigma(self, circle_mesh):
        sigma1 = np.ones(circle_mesh.n_elements)
        sigma2 = 3.0 * sigma1
        K1 = assemble_K(circle_mesh, sigma1)
        K2 = assemble_K(circle_mesh, sigma2)
        np.testing.assert_allclose(K2.toarray(), 3.0 * K1.toarray(), rtol=1e-12)

    def test_heterogeneous_sigma(self, circle_mesh):
        rng = np.random.default_rng(0)
        sigma = rng.uniform(0.5, 3.0, circle_mesh.n_elements)
        K = assemble_K(circle_mesh, sigma)
        assert K.shape == (circle_mesh.n_nodes, circle_mesh.n_nodes)
        # Still symmetric and positive diagonal
        diff = (K - K.T).toarray()
        np.testing.assert_allclose(diff, 0, atol=1e-12)
        assert np.all(K.diagonal() > 0)


# ---------------------------------------------------------------------------
# simulate
# ---------------------------------------------------------------------------


class TestSimulate:
    @pytest.fixture(scope="class")
    def protocol(self):
        return adjacent_pattern(16), measurement_pairs(16)

    def test_output_shape(self, circle_mesh, electrodes, protocol):
        drive, meas = protocol
        sigma = np.ones(circle_mesh.n_elements)
        dV = simulate(circle_mesh, electrodes, sigma, drive, meas, sigma0=1.0)
        assert dV.shape == (meas.shape[0],)

    def test_output_dtype(self, circle_mesh, electrodes, protocol):
        drive, meas = protocol
        sigma = np.ones(circle_mesh.n_elements)
        dV = simulate(circle_mesh, electrodes, sigma, drive, meas, sigma0=1.0)
        assert dV.dtype == np.float64

    def test_uniform_sigma_zero_dV(self, circle_mesh, electrodes, protocol):
        # Difference EIT: σ == σ₀ → δV = 0 exactly
        drive, meas = protocol
        sigma = np.ones(circle_mesh.n_elements) * 2.5
        dV = simulate(circle_mesh, electrodes, sigma, drive, meas, sigma0=2.5)
        np.testing.assert_allclose(dV, 0.0, atol=1e-12)

    def test_anomaly_nonzero_dV(self, circle_mesh, electrodes, protocol):
        drive, meas = protocol
        sigma = np.ones(circle_mesh.n_elements)
        centroids = circle_mesh.nodes[circle_mesh.elements].mean(axis=1)
        inside = (centroids[:, 0] ** 2 + (centroids[:, 1] - 0.5) ** 2) < 0.1**2
        sigma[inside] = 3.0
        dV = simulate(circle_mesh, electrodes, sigma, drive, meas, sigma0=1.0)
        assert np.any(np.abs(dV) > 1e-8)

    def test_conductive_anomaly_sign(self, circle_mesh, electrodes, protocol):
        # A conductive inclusion near electrode 3 should perturb dV near
        # that drive step more than drive steps far away
        drive, meas = protocol
        sigma = np.ones(circle_mesh.n_elements)
        # Place inclusion near electrode 3 (at ~67.5° on unit circle)
        angle = 2 * np.pi * 3 / 16
        cx, cy = 0.6 * np.cos(angle), 0.6 * np.sin(angle)
        centroids = circle_mesh.nodes[circle_mesh.elements].mean(axis=1)
        inside = (centroids[:, 0] - cx) ** 2 + (centroids[:, 1] - cy) ** 2 < 0.15**2
        sigma[inside] = 4.0
        dV = simulate(circle_mesh, electrodes, sigma, drive, meas, sigma0=1.0)
        # Nearby drive steps (2, 3, 4) have larger |dV| than opposite (10, 11)
        near_steps = [2, 3, 4]
        far_steps = [10, 11]
        n_per = 16 - 3
        near_rms = np.sqrt(
            np.mean([dV[k * n_per : (k + 1) * n_per] ** 2 for k in near_steps])
        )
        far_rms = np.sqrt(
            np.mean([dV[k * n_per : (k + 1) * n_per] ** 2 for k in far_steps])
        )
        assert near_rms > far_rms

    def test_linearity_in_sigma_perturbation(self, circle_mesh, electrodes, protocol):
        # δV should scale (approximately) linearly with anomaly amplitude
        # for small perturbations (Born approximation regime)
        drive, meas = protocol
        centroids = circle_mesh.nodes[circle_mesh.elements].mean(axis=1)
        inside = centroids[:, 0] ** 2 + centroids[:, 1] ** 2 < 0.2**2

        dVs = []
        for amp in [0.01, 0.02]:
            sigma = np.ones(circle_mesh.n_elements)
            sigma[inside] += amp
            dVs.append(
                simulate(circle_mesh, electrodes, sigma, drive, meas, sigma0=1.0)
            )
        # dV(2×amp) ≈ 2 × dV(amp) with tolerance for nonlinearity
        ratio = dVs[1] / (dVs[0] + 1e-20)
        np.testing.assert_allclose(ratio[np.abs(dVs[0]) > 1e-10], 2.0, rtol=0.05)


# ---------------------------------------------------------------------------
# compute_jacobian
# ---------------------------------------------------------------------------


class TestComputeJacobian:
    @pytest.fixture(scope="class")
    def jacobian_data(self, circle_mesh, electrodes):
        drive = adjacent_pattern(16)
        meas = measurement_pairs(16)
        sigma = np.ones(circle_mesh.n_elements)
        J = compute_jacobian(circle_mesh, electrodes, sigma, drive, meas)
        return J, circle_mesh, electrodes, sigma, drive, meas

    def test_shape(self, jacobian_data):
        J, mesh, *_ = jacobian_data
        P = 208
        M = mesh.n_elements
        assert J.shape == (P, M)

    def test_dtype(self, jacobian_data):
        J, *_ = jacobian_data
        assert J.dtype == np.float64

    def test_not_all_zero(self, jacobian_data):
        J, *_ = jacobian_data
        assert np.any(np.abs(J) > 1e-10)

    def test_finite(self, jacobian_data):
        J, *_ = jacobian_data
        assert np.all(np.isfinite(J))

    def test_fd_accuracy(self, circle_mesh, electrodes):
        """Finite-difference check: J·δσ ≈ δV for a small perturbation."""
        drive = adjacent_pattern(16)
        meas = measurement_pairs(16)
        sigma = np.ones(circle_mesh.n_elements)
        J = compute_jacobian(circle_mesh, electrodes, sigma, drive, meas)

        # Perturb a single element
        pert = np.zeros(circle_mesh.n_elements)
        pert[0] = 1e-4
        dV_fd = (
            simulate(circle_mesh, electrodes, sigma + pert, drive, meas, sigma0=1.0)
            - simulate(circle_mesh, electrodes, sigma - pert, drive, meas, sigma0=1.0)
        ) / 2.0
        dV_jac = J @ pert

        rel_err = np.linalg.norm(dV_fd - dV_jac) / (np.linalg.norm(dV_fd) + 1e-20)
        assert rel_err < 1e-4, f"FD relative error {rel_err:.2e} too large"

    def test_jacobian_row_sparsity(self, jacobian_data):
        # Elements far from the driven / measured pair should have near-zero
        # sensitivity; the Jacobian must not be a dense matrix of identical values
        J, *_ = jacobian_data
        col_stds = J.std(axis=0)
        # At least half the columns should have non-trivial spatial variation
        assert np.sum(col_stds > 1e-12) > J.shape[1] // 2
