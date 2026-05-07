"""
eitkit.inverse — Reconstruction algorithms.

Sub-packages
------------
classical       Tikhonov, Gauss-Newton, TV, back-projection, GREIT, D-bar.
data_driven     CNN, U-Net (Phase 3).
physics_informed  PINN, deep image prior (Phase 4).
hybrid          Algorithm unrolling, learned regulariser (Phase 4).
"""

from eitkit.inverse.classical import tikhonov_solve, choose_lambda, build_gradient_op, tv_solve

__all__ = ["tikhonov_solve", "choose_lambda", "build_gradient_op", "tv_solve"]
