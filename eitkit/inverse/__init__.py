"""
eitkit.inverse — Reconstruction algorithms.

Sub-packages
------------
classical       Linearised difference reconstruction: Tikhonov (L2) with
                L-curve parameter choice, and total variation (L1) via ADMM.
"""

from eitkit.inverse.classical import (
    build_gradient_op,
    choose_lambda,
    tikhonov_solve,
    tv_solve,
)

__all__ = ["tikhonov_solve", "choose_lambda", "build_gradient_op", "tv_solve"]
