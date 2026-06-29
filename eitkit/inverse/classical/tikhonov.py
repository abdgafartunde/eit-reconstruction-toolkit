r"""
Tikhonov regularisation for EIT difference imaging.

Theory
------
Given the linearised EIT observation model

.. math::

    \delta V \approx J \, \delta\sigma

the standard (zeroth-order) Tikhonov solution minimises

.. math::

    \|\delta V - J\,\delta\sigma\|_2^2
    + \lambda \|\delta\sigma\|_2^2

whose closed-form solution is

.. math::

    \delta\hat\sigma = (J^T J + \lambda I)^{-1} J^T \delta V

For well-posed, moderate-size EIT problems (P ~ 200, E ~ 1500) this is
solved via the **normal equations** using :func:`numpy.linalg.solve`.
For very large systems an iterative path (LSQR) is preferred; that is
exposed through the ``solver`` keyword.

Public API
----------
tikhonov_solve   Reconstruct δσ from J, δV, and a regularisation parameter λ.
choose_lambda    L-curve corner heuristic for automatic λ selection.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

__all__ = ["tikhonov_solve", "choose_lambda"]


# ── helpers ──────────────────────────────────────────────────────────────────


def _validate_inputs(
    J: NDArray[np.float64],
    dV: NDArray[np.float64],
    lambda_: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Coerce inputs and validate shapes / values."""
    J = np.asarray(J, dtype=np.float64)
    dV = np.asarray(dV, dtype=np.float64)

    if J.ndim != 2:
        raise ValueError(f"J must be 2-D, got shape {J.shape}")
    if dV.ndim != 1:
        raise ValueError(f"dV must be 1-D, got shape {dV.shape}")
    P, E = J.shape
    if dV.shape[0] != P:
        raise ValueError(f"dV length {dV.shape[0]} does not match J rows {P}")
    if not np.isfinite(lambda_) or lambda_ < 0.0:
        raise ValueError(
            f"lambda_ must be a non-negative finite scalar, got {lambda_!r}"
        )
    return J, dV


# ── public API ────────────────────────────────────────────────────────────────


def tikhonov_solve(
    J: NDArray[np.float64],
    dV: NDArray[np.float64],
    lambda_: float,
    *,
    solver: str = "direct",
) -> NDArray[np.float64]:
    r"""Tikhonov-regularised EIT reconstruction.

    Solves the normal equations

    .. math::

        (J^T J + \lambda I)\,\delta\hat\sigma = J^T \delta V

    Parameters
    ----------
    J:
        Jacobian / sensitivity matrix, shape ``(P, E)``.
        ``P`` = number of measurements, ``E`` = number of mesh elements.
    dV:
        Difference voltage vector, shape ``(P,)``.
        Typically ``dV = V(sigma) - V(sigma_ref)``.
    lambda_:
        Tikhonov regularisation parameter :math:`\lambda \geq 0`.
        Larger values yield smoother but less accurate reconstructions.
        ``lambda_ = 0`` gives the ordinary least-squares solution
        (use with caution — the problem is almost always ill-conditioned).
    solver:
        ``"direct"`` (default) — ``numpy.linalg.solve`` on the ``(E, E)``
        normal-equations system.  Fast for ``E`` up to ~10 000.

        ``"lsqr"`` — ``scipy.sparse.linalg.lsqr`` on the augmented system
        :math:`[J; \sqrt\lambda I]\,\delta\sigma = [\delta V; 0]`.
        Preferred when ``E`` is large or ``J`` is sparse.

    Returns
    -------
    delta_sigma : ndarray, shape ``(E,)``, dtype ``float64``
        Reconstructed conductivity perturbation.

    Raises
    ------
    ValueError
        If ``J`` is not 2-D, ``dV`` length mismatches ``J`` rows, or
        ``lambda_`` is negative / non-finite.
    numpy.linalg.LinAlgError
        Propagated from the linear solver if the system is singular
        (should not happen when ``lambda_ > 0``).

    Examples
    --------
    >>> import numpy as np
    >>> from eitkit.inverse.classical import tikhonov_solve
    >>> rng = np.random.default_rng(0)
    >>> J  = rng.standard_normal((208, 100))
    >>> dV = rng.standard_normal(208)
    >>> ds = tikhonov_solve(J, dV, lambda_=1e-3)
    >>> ds.shape
    (100,)
    """
    J, dV = _validate_inputs(J, dV, lambda_)
    P, E = J.shape

    if solver == "direct":
        # Normal equations: (J^T J + λ I) δσ = J^T δV
        A = J.T @ J
        A.flat[:: E + 1] += lambda_  # add λ to diagonal (in-place)
        rhs = J.T @ dV
        return np.linalg.solve(A, rhs)

    if solver == "lsqr":
        import scipy.sparse.linalg as spla  # deferred import — optional dep path

        # Augmented system: [J; sqrt(λ)·I] δσ = [δV; 0]
        sqrt_lam = float(np.sqrt(lambda_))
        A_aug = np.vstack([J, sqrt_lam * np.eye(E)])
        b_aug = np.concatenate([dV, np.zeros(E)])
        result = spla.lsqr(A_aug, b_aug, atol=1e-10, btol=1e-10, iter_lim=10 * E)
        return result[0].astype(np.float64)

    raise ValueError(f"Unknown solver {solver!r}. Choose 'direct' or 'lsqr'.")


def choose_lambda(
    J: NDArray[np.float64],
    dV: NDArray[np.float64],
    lambdas: NDArray[np.float64] | None = None,
    *,
    n_points: int = 50,
    solver: str = "direct",
) -> tuple[float, NDArray[np.float64], NDArray[np.float64]]:
    r"""Select a regularisation parameter via the L-curve heuristic.

    The L-curve is the parametric curve

    .. math::

        \bigl(\log\|J\delta\hat\sigma - \delta V\|,\;
               \log\|\delta\hat\sigma\|\bigr)

    plotted against :math:`\lambda`.  The "corner" of the L — maximum
    curvature — balances data fit against solution norm and provides a
    robust automatic choice of :math:`\lambda`.

    Parameters
    ----------
    J:
        Jacobian matrix, shape ``(P, E)``.
    dV:
        Difference voltage vector, shape ``(P,)``.
    lambdas:
        1-D array of candidate :math:`\lambda` values to evaluate.
        If ``None``, defaults to 50 log-spaced values from
        ``1e-6 * ||J||_F^2 / P`` to ``1e2 * ||J||_F^2 / P``.
    n_points:
        Number of log-spaced candidates when ``lambdas`` is ``None``.
    solver:
        Passed through to :func:`tikhonov_solve`.  ``"direct"`` (default)
        uses the normal equations; ``"lsqr"`` uses an iterative solver
        on the augmented system.

    Returns
    -------
    lambda_opt : float
        The :math:`\lambda` at the L-curve corner (maximum curvature).
    residuals : ndarray, shape ``(n_points,)``
        ``log10`` of the residual norm for each candidate.
    solution_norms : ndarray, shape ``(n_points,)``
        ``log10`` of the solution norm for each candidate.

    Notes
    -----
    Curvature is estimated from the discrete second derivative of the
    L-curve in log-log space using the standard formula [Hansen1992]_.

    .. [Hansen1992] Hansen, P.C. (1992). Analysis of discrete ill-posed problems
       by means of the L-curve. *SIAM Review*, 34(4), 561–580.
    """
    J = np.asarray(J, dtype=np.float64)
    dV = np.asarray(dV, dtype=np.float64)

    if lambdas is None:
        scale = np.linalg.norm(J, "fro") ** 2 / J.shape[0]
        lambdas = np.logspace(
            np.log10(1e-6 * scale),
            np.log10(1e2 * scale),
            n_points,
        )
    lambdas = np.asarray(lambdas, dtype=np.float64)

    residuals = np.empty(len(lambdas))
    solution_norms = np.empty(len(lambdas))

    for i, lam in enumerate(lambdas):
        ds = tikhonov_solve(J, dV, float(lam), solver=solver)
        residuals[i] = np.log10(np.linalg.norm(J @ ds - dV) + 1e-300)
        solution_norms[i] = np.log10(np.linalg.norm(ds) + 1e-300)

    # Maximum-curvature corner via discrete second derivative
    # curvature proxy κ_i ≈ x''y' - x'y'' (unnormalised, sign-consistent)
    x = residuals
    y = solution_norms
    # first differences
    dx = np.diff(x)
    dy = np.diff(y)
    # second differences (align to interior points)
    d2x = np.diff(dx)
    d2y = np.diff(dy)
    # curvature numerator (interior points only: indices 1..-1)
    kappa = np.abs(dx[:-1] * d2y - dy[:-1] * d2x)
    corner_idx = int(np.argmax(kappa)) + 1  # +1: offset back to full array
    lambda_opt = float(lambdas[corner_idx])

    return lambda_opt, residuals, solution_norms
