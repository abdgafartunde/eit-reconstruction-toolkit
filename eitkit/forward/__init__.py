"""
eitkit.forward — FEM-based forward solver for EIT.

Public API
----------
simulate        Run the forward problem for all stimulation patterns.
assemble_K      Assemble the global FEM stiffness matrix.
compute_jacobian  Compute the Jacobian (sensitivity matrix) via the adjoint method.
"""

from eitkit.forward.simulation import simulate
from eitkit.forward.fem_assembler import assemble_K
from eitkit.forward.jacobian import compute_jacobian

__all__ = ["simulate", "assemble_K", "compute_jacobian"]
