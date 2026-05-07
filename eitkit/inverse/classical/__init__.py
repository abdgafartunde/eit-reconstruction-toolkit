"""eitkit.inverse.classical — Classical regularisation-based solvers (Phase 2)."""

from eitkit.inverse.classical.tikhonov import tikhonov_solve, choose_lambda
from eitkit.inverse.classical.tv import build_gradient_op, tv_solve

__all__ = ["tikhonov_solve", "choose_lambda", "build_gradient_op", "tv_solve"]
