# Inverse Solvers

`eitkit.inverse` provides two classical reconstruction methods for the
linearised difference-EIT problem $J \, \delta\sigma \approx \delta V$.

## Tikhonov (L2) regularisation

Minimises the Tikhonov functional:

$$\hat{\delta\sigma} = \arg\min_{\delta\sigma}
    \| J \delta\sigma - \delta V \|_2^2 + \lambda \| \delta\sigma \|_2^2$$

Solution via normal equations: $(J^T J + \lambda I)\,\delta\sigma = J^T \delta V$.

```python
from eitkit.inverse import tikhonov_solve

dsigma = tikhonov_solve(J, dV, lambda_=1e-4)
dsigma = tikhonov_solve(J, dV, lambda_=1e-4, solver="lsqr")  # iterative
```

### Choosing λ — L-curve

```python
from eitkit.inverse import choose_lambda
import numpy as np

lambdas = np.logspace(-6, 2, 50)
lambda_opt, log_residuals, log_sol_norms = choose_lambda(J, dV, lambdas=lambdas)
```

`choose_lambda` sweeps λ, plots the solution norm vs. residual norm in
log-log space, and selects the corner via maximum discrete curvature.

!!! note
    `choose_lambda` returns **log₁₀** of the norms. Use a plain `ax.plot`
    rather than `ax.loglog` to avoid "no positive values" warnings.

## Total Variation (TV / L1) regularisation

TV regularisation penalises the L1 norm of the element-wise gradient,
promoting piecewise-constant solutions with sharp edges:

$$\hat{\delta\sigma} = \arg\min_{\delta\sigma}
    \| J \delta\sigma - \delta V \|_2^2 + \alpha \| D \delta\sigma \|_1$$

Solved via the **Alternating Direction Method of Multipliers (ADMM)**.

```python
from eitkit.inverse import build_gradient_op, tv_solve

D      = build_gradient_op(mesh)   # sparse (F, E), entries ±1
dsigma = tv_solve(J, dV, alpha, mesh, rho=rho, max_iter=500, tol=1e-6)
```

### ρ scaling — critical for good results

The ADMM σ-update solves $(J^T J + \rho D^T D)\,\delta\sigma = \ldots$
If $\rho$ is too large, $\rho D^T D$ dominates and the solver ignores the
data. Balance both terms with:

$$\rho^* = \frac{\|J\|_F^2}{\text{nnz}(D)}$$

since every entry of $D$ is $\pm 1$, so $\|D\|_F^2 = \text{nnz}(D)$.

```python
rho   = float(np.linalg.norm(J, 'fro')**2) / D.nnz
alpha = 0.15 * rho    # kappa = 0.15: moderate edge-preservation
dsigma = tv_solve(J, dV, alpha, mesh, rho=rho, max_iter=500, tol=1e-6)
```

!!! warning
    The default `rho=1.0` is only appropriate if $J$ and $D$ are pre-scaled.
    Always set `rho` explicitly using the formula above.
