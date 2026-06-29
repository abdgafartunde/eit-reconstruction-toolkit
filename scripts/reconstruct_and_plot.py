"""
scripts/reconstruct_and_plot.py
===============================
End-to-end EIT reconstruction: forward simulation, Jacobian,
Tikhonov (L-curve), and TV (ADMM) — saves figures to ../figures/

Run: python scripts/reconstruct_and_plot.py
"""
from __future__ import annotations

import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt

# Ensure eitkit is importable from the repo root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eitkit.mesh import make_circle_mesh, place_electrodes
from eitkit.protocol import adjacent_pattern, measurement_pairs, add_noise
from eitkit.forward import simulate, compute_jacobian
from eitkit.utils import make_phantom, plot_mesh, plot_conductivity, plot_voltages
from eitkit.inverse import tikhonov_solve, choose_lambda, tv_solve, build_gradient_op

# ---------------------------------------------------------------------------
# Output directory (relative to repo root)
# ---------------------------------------------------------------------------
OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")
os.makedirs(OUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# 1. Mesh & electrodes
# ---------------------------------------------------------------------------
print("Generating mesh …")
mesh = make_circle_mesh(n_electrodes=16, h0=0.08, seed=42)
ec = place_electrodes(mesh, n_electrodes=16)
print(f"  nodes={mesh.n_nodes}, elements={mesh.n_elements}, boundary={mesh.n_boundary_nodes}")

# ---------------------------------------------------------------------------
# 2. Conductivity phantom — moderate contrasts (linear regime)
#
# NOTE: Difference EIT with a single linear step (Born approximation)
# requires *moderate* conductivity contrasts (|δσ|/σ₀ ≲ 0.5).
# The Jacobian is evaluated at the background σ₀; for large contrasts
# the linearisation error ‖δV − J·δσ‖ dominates and reconstructions
# appear severely amplitude-damped.  Nonlinear Gauss-Newton iteration
# (not yet implemented) would be needed for strong contrasts (> ±50 %).
# ---------------------------------------------------------------------------
print("Building phantom …")
sigma_bg = np.ones(mesh.n_elements)
sigma_true = make_phantom(
    mesh,
    [
        # Conductive inclusion: σ = 1.6 S/m  (+60 %)
        {"shape": "circle",   "cx": 0.40, "cy": 0.30, "r": 0.22, "sigma": 1.6},
        # Resistive inclusion:  σ = 0.5 S/m  (−50 %)
        {"shape": "ellipse",  "cx": -0.35, "cy": -0.25, "a": 0.25, "b": 0.14,
         "theta": 0.6, "sigma": 0.5},
    ],
    sigma_background=1.0,
)
delta_sigma_true = sigma_true - sigma_bg

# ---------------------------------------------------------------------------
# 3. Forward simulation
# ---------------------------------------------------------------------------
print("Running forward simulation …")
drive = adjacent_pattern(16)
meas = measurement_pairs(16)

dV_clean = simulate(mesh, ec, sigma_true, drive, meas, sigma0=1.0)
dV_noisy = add_noise(dV_clean, snr_db=45.0, rng=42)

print(f"  dV clean:  range [{dV_clean.min():.4e}, {dV_clean.max():.4e}] V")
print(f"  dV noisy:  range [{dV_noisy.min():.4e}, {dV_noisy.max():.4e}] V")
print(f"  ‖δV_clean‖ = {np.linalg.norm(dV_clean):.4e}")
print(f"  ‖δV_noisy‖ = {np.linalg.norm(dV_noisy):.4e}")

# ---------------------------------------------------------------------------
# 4. Jacobian at background conductivity
# ---------------------------------------------------------------------------
print("Computing Jacobian …")
J = compute_jacobian(mesh, ec, sigma_bg, drive, meas)
print(f"  J shape = {J.shape}")

# Diagnostic: how good is the linear approximation?
dV_linear = J @ delta_sigma_true
lin_err = np.linalg.norm(dV_clean - dV_linear) / np.linalg.norm(dV_clean)
print(f"  linearisation error ‖δV − J·δσ‖/‖δV‖ = {lin_err:.4f}")
if lin_err > 0.5:
    print("  ⚠  Born approximation is poor — expect amplitude damping in reconstructions.")

# ---------------------------------------------------------------------------
# 5. Tikhonov reconstruction — weak regularisation
# ---------------------------------------------------------------------------
print("Tikhonov reconstruction …")
# Use a small fixed λ (L-curve tends to oversmooth for difference EIT)
lam_tik = 1e-7
ds_tik = tikhonov_solve(J, dV_noisy, lambda_=lam_tik)
sigma_tik = sigma_bg + ds_tik
print(f"  λ = {lam_tik:.0e}")
print(f"  ‖δσ_tik‖ = {np.linalg.norm(ds_tik):.4f}")
print(f"  δσ range  = [{ds_tik.min():.3f}, {ds_tik.max():.3f}] S/m")
print(f"  residual  = {np.linalg.norm(J @ ds_tik - dV_noisy):.4e}")

# Also show L-curve for reference
lam_opt, residuals, sol_norms = choose_lambda(J, dV_noisy, n_points=40)
print(f"  L-curve λ_opt = {lam_opt:.2e} (shown for reference; using fixed λ={lam_tik:.0e})")

# ---------------------------------------------------------------------------
# 6. TV (ADMM) reconstruction — weak regularisation
# ---------------------------------------------------------------------------
print("TV reconstruction (ADMM) …")
D = build_gradient_op(mesh)
ds_tv = tv_solve(J, dV_noisy, alpha=5e-8, mesh=mesh, rho="auto",
                  max_iter=300, tol=1e-4)
sigma_tv = sigma_bg + ds_tv
print(f"  ‖δσ_tv‖ = {np.linalg.norm(ds_tv):.4f}")
print(f"  δσ range  = [{ds_tv.min():.3f}, {ds_tv.max():.3f}] S/m")
print(f"  TV(δσ_tv) = {np.linalg.norm(D @ ds_tv, 1):.4f}")
print(f"  residual   = {np.linalg.norm(J @ ds_tv - dV_noisy):.4e}")

# ---------------------------------------------------------------------------
# 7. Figure 1 — Mesh + phantom
# ---------------------------------------------------------------------------
print("Saving figures …")
fig, axes = plt.subplots(1, 2, figsize=(11, 5))
fig.suptitle("Mesh & Conductivity Phantom", fontsize=13, fontweight="bold")

plot_mesh(mesh, elec_config=ec, ax=axes[0], title="Triangular Mesh (16 electrodes)")
plot_conductivity(mesh, sigma_true, ax=axes[1], title="True Conductivity σ (S/m)",
                  cmap="hot_r", show_mesh=False, elec_config=ec)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "01_mesh_and_phantom.png"), dpi=150)
plt.close(fig)

# ---------------------------------------------------------------------------
# 8. Figure 2 — Voltage data
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Simulated Voltage Data", fontsize=13, fontweight="bold")

plot_voltages(meas, dV_clean, ax=axes[0], title="Clean δV",
              highlight_drive=4)
plot_voltages(meas, dV_noisy, ax=axes[1], title=f"Noisy δV (SNR = 45 dB)",
              highlight_drive=4)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "02_voltage_data.png"), dpi=150)
plt.close(fig)

# ---------------------------------------------------------------------------
# 9. Figure 3 — L-curve
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(5.5, 4.5))
fig.suptitle("L-Curve — Tikhonov Regularisation", fontsize=12, fontweight="bold")
ax.plot(residuals, sol_norms, "o-", markersize=3, linewidth=0.8, color="steelblue")
ax.scatter(residuals[np.argmax(np.abs(np.diff(np.diff(residuals)))) + 1],
           sol_norms[np.argmax(np.abs(np.diff(np.diff(residuals)))) + 1],
           s=80, color="crimson", zorder=5, label=f"λ = {lam_opt:.2e}")
ax.set_xlabel("log₁₀ ‖J δσ − δV‖")
ax.set_ylabel("log₁₀ ‖δσ‖")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "03_lcurve.png"), dpi=150)
plt.close(fig)

# ---------------------------------------------------------------------------
# 10. Figure 4 — Reconstructions side by side
# ---------------------------------------------------------------------------
vmin = min(sigma_true.min(), sigma_tik.min(), sigma_tv.min())
vmax = max(sigma_true.max(), sigma_tik.max(), sigma_tv.max())

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("EIT Reconstructions — 16 Electrodes, Adjacent Drive", fontsize=13, fontweight="bold")

plot_conductivity(mesh, sigma_true, ax=axes[0],
                  title="True Conductivity σ", cmap="hot_r",
                  vmin=vmin, vmax=vmax, show_mesh=False, elec_config=ec)
plot_conductivity(mesh, sigma_tik, ax=axes[1],
                  title=f"Tikhonov (λ = {lam_tik:.0e})", cmap="hot_r",
                  vmin=vmin, vmax=vmax, show_mesh=False, elec_config=ec)
plot_conductivity(mesh, sigma_tv, ax=axes[2],
                  title="TV / ADMM (α = 5×10⁻⁸)", cmap="hot_r",
                  vmin=vmin, vmax=vmax, show_mesh=False, elec_config=ec)

fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "04_reconstructions.png"), dpi=150)
plt.close(fig)

# ---------------------------------------------------------------------------
# 11. Figure 5 — Conductivity perturbation δσ
# ---------------------------------------------------------------------------
# Symmetric colour scale for perturbations
pert_vmax = max(np.abs(delta_sigma_true).max(),
                np.abs(ds_tik).max(), np.abs(ds_tv).max())

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("Conductivity Perturbation δσ = σ − σ₀", fontsize=13, fontweight="bold")

plot_conductivity(mesh, delta_sigma_true, ax=axes[0],
                  title="True δσ", cmap="RdBu_r",
                  vmin=-pert_vmax, vmax=pert_vmax,
                  show_mesh=False, elec_config=ec)
plot_conductivity(mesh, ds_tik, ax=axes[1],
                  title="Tikhonov δσ", cmap="RdBu_r",
                  vmin=-pert_vmax, vmax=pert_vmax,
                  show_mesh=False, elec_config=ec)
plot_conductivity(mesh, ds_tv, ax=axes[2],
                  title="TV δσ", cmap="RdBu_r",
                  vmin=-pert_vmax, vmax=pert_vmax,
                  show_mesh=False, elec_config=ec)

fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "05_perturbations.png"), dpi=150)
plt.close(fig)

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("  Reconstruction Summary")
print("=" * 60)
print(f"  Mesh:  {mesh.n_nodes} nodes, {mesh.n_elements} elements")
print(f"  Data:  {len(dV_clean)} measurements, {len(drive)} drive steps")
print(f"  Linearisation error: {lin_err:.4f}")
print(f"  True δσ range:     [{delta_sigma_true.min():.2f}, {delta_sigma_true.max():.2f}] S/m")
print(f"  Tikhonov δσ range: [{ds_tik.min():.3f}, {ds_tik.max():.3f}] S/m  (λ={lam_tik:.0e})")
print(f"  TV δσ range:       [{ds_tv.min():.3f}, {ds_tv.max():.3f}] S/m  (α=5e-8, ρ=auto)")
print(f"  ‖δσ_true − δσ_tik‖ / ‖δσ_true‖ = {np.linalg.norm(delta_sigma_true - ds_tik) / np.linalg.norm(delta_sigma_true):.4f}")
print(f"  ‖δσ_true − δσ_tv‖  / ‖δσ_true‖ = {np.linalg.norm(delta_sigma_true - ds_tv) / np.linalg.norm(delta_sigma_true):.4f}")
print(f"  Figures saved to: {OUT_DIR}")
print("=" * 60)
