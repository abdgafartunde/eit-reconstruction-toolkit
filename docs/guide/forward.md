# Forward Solver

`eitkit.forward` implements a P1 finite-element forward model with a gap
electrode boundary condition.

## FEM assembly

Element stiffness matrices use piecewise-linear (P1) basis functions:

$$K^e = \frac{\sigma_e}{4 A_e} B^T B$$

where $B$ is the $2 \times 3$ gradient matrix of the shape functions and
$A_e$ is the element area.

```python
from eitkit.forward import assemble_K

K = assemble_K(mesh, sigma)   # sparse CSR, shape (N, N)
```

## Gap electrode model

Each electrode injects current over a single boundary node (Neumann BC). The
load vector $f$ has $\pm 1/\text{arc\_width}$ at the source/sink nodes.

```python
from eitkit.forward import apply_neumann_bc, pick_ground_node, solve_forward

f      = apply_neumann_bc(mesh, ec, drive_pair)
ground = pick_ground_node(mesh, ec, drive_pair)
u      = solve_forward(K, f, ground)   # nodal potentials
```

## Difference simulation

`simulate` computes $\delta V = V(\sigma) - V(\sigma_0)$ for all measurement
pairs in one call, using $\sigma_0 = 1$ S/m as reference.

```python
from eitkit.forward import simulate

dV = simulate(mesh, ec, sigma, drive_pairs, meas_pairs)
# shape (P,)  where P = n_drive * n_meas_per_drive
```

## Adjoint Jacobian

`compute_jacobian` builds $J \in \mathbb{R}^{P \times E}$ using $L$ forward
solves and $L$ adjoint solves — much cheaper than finite differences.

$$J_{pe} = -\sigma_e \int_{\Omega_e} \nabla u_p \cdot \nabla v_p \, \mathrm{d}\Omega$$

```python
from eitkit.forward import compute_jacobian

J = compute_jacobian(mesh, ec, sigma_ref, drive_pairs, meas_pairs)
# shape (208, n_elements)
```

!!! warning
    `sigma` is the **third** positional argument, before `drive_pairs` and
    `meas_pairs`. Passing them in the wrong order produces silently incorrect
    results.
