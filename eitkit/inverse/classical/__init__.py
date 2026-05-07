"""eitkit.inverse.classical — Classical regularisation-based solvers (Phase 2)."""

from eitkit.inverse.classical.tikhonov import choose_lambda, tikhonov_solve
from eitkit.inverse.classical.tv import build_gradient_op, tv_solve

__all__ = ["tikhonov_solve", "choose_lambda", "build_gradient_op", "tv_solve"]
