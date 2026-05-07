"""
scripts/verify_jacobian.py
===========================
Visual sanity-check for the Jacobian (sensitivity matrix).

Run with:

    python scripts/verify_jacobian.py

Panels
------
A  Sensitivity map for a single measurement — which elements matter most
B  Full J matrix as a heat-map (measurement × element)
C  Singular-value spectrum of J — rank / ill-conditioning
D  Finite-difference validation across 20 random (meas, elem) pairs
"""

from __future__ import annotations

import sys
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri

sys.path.insert(0, ".")

from eitkit.mesh import make_circle_mesh, place_electrodes
from eitkit.protocol import adjacent_pattern, measurement_pairs
from eitkit.forward import compute_jacobian, simulate

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

mesh  = make_circle_mesh(n_electrodes=16, h0=0.12, seed=42)
ec    = place_electrodes(mesh, n_electrodes=16)
drive = adjacent_pattern(16)
meas  = measurement_pairs(16)
sigma = np.ones(mesh.n_elements)

# ---------------------------------------------------------------------------
# Compute J and print stats
# ---------------------------------------------------------------------------

t0 = time.perf_counter()
J  = compute_jacobian(mesh, ec, sigma, drive, meas)
t1 = time.perf_counter()

print("=" * 58)
print("  Jacobian verification")
print("=" * 58)
print(f"  J shape          : {J.shape}  (P × M)")
print(f"  J range          : [{J.min():.5f}, {J.max():.5f}]")
print(f"  J finite?        : {np.all(np.isfinite(J))}")
print(f"  Compute time     : {t1 - t0:.2f} s")
print()

# Finite-difference spot-checks
rng     = np.random.default_rng(0)
eps     = 1e-5
n_check = 20
e_idx   = rng.integers(0, mesh.n_elements, n_check)
i_idx   = rng.integers(0, len(meas),       n_check)

max_rel = 0.0
for e_test, i_test in zip(e_idx, i_idx):
    sp = sigma.copy(); sp[e_test] += eps
    sm = sigma.copy(); sm[e_test] -= eps
    dVp = simulate(mesh, ec, sp, drive, meas, sigma0=sigma)
    dVm = simulate(mesh, ec, sm, drive, meas, sigma0=sigma)
    J_fd  = (dVp[i_test] - dVm[i_test]) / (2 * eps)
    J_adj = J[i_test, e_test]
    rel   = abs(J_adj - J_fd) / (abs(J_fd) + 1e-15)
    max_rel = max(max_rel, rel)

print(f"  FD spot-check ({n_check} pairs)  max rel error: {max_rel:.2e}")
print()

# Singular values
sv = np.linalg.svd(J, compute_uv=False)
print(f"  Singular values  : max={sv[0]:.4f}  min_nonzero≈{sv[sv>1e-10].min():.2e}")
print(f"  Condition number : {sv[0]/sv[sv>1e-10].min():.2e}")
print(f"  Numerical rank   : {(sv > sv[0]*1e-6).sum()}")
print()

# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle("eitkit · Jacobian Verification", fontsize=14, fontweight="bold")

triang = tri.Triangulation(mesh.nodes[:, 0], mesh.nodes[:, 1], mesh.elements)

# ── Panel A: sensitivity map for measurement 0 ────────────────────────────
ax = axes[0, 0]
i_show = 0   # first measurement (drive 0, pair 2→3)
k_show, p_el, m_el = meas[i_show]
ax.set_title(f"A  Sensitivity  J[{i_show},:] "
             f"(drive={k_show}, +el={p_el}, −el={m_el})")
tpc = ax.tripcolor(triang, J[i_show, :], cmap="RdBu_r", shading="flat",
                   vmax=np.abs(J[i_show,:]).max(),
                   vmin=-np.abs(J[i_show,:]).max())
fig.colorbar(tpc, ax=ax, label="∂V/∂σ")
el_xy = mesh.nodes[ec.node_indices]
ax.scatter(el_xy[:, 0], el_xy[:, 1], s=30, color="k", zorder=5)
ax.scatter(*mesh.nodes[int(ec.node_indices[int(p_el)])], s=100,
           color="red",  zorder=6, label=f"+el {p_el}")
ax.scatter(*mesh.nodes[int(ec.node_indices[int(m_el)])], s=100,
           color="blue", zorder=6, label=f"−el {m_el}")
ax.set_aspect("equal"); ax.legend(fontsize=8, loc="lower right")
ax.set_xlabel("x"); ax.set_ylabel("y")

# ── Panel B: full J heat-map ───────────────────────────────────────────────
ax = axes[0, 1]
ax.set_title("B  Full Jacobian J  (208 × 488)")
vmax = np.abs(J).max()
im   = ax.imshow(J, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                 interpolation="none")
fig.colorbar(im, ax=ax, label="∂V/∂σ")
ax.set_xlabel("Element index e")
ax.set_ylabel("Measurement index i")
# Drive-step boundaries
for k in range(1, 16):
    ax.axhline(k * 13 - 0.5, color="white", linewidth=0.4)

# ── Panel C: singular-value spectrum ──────────────────────────────────────
ax = axes[1, 0]
ax.set_title("C  Singular-Value Spectrum of J")
ax.semilogy(sv, "o-", markersize=3, color="steelblue")
ax.axhline(sv[0] * 1e-6, color="crimson", linewidth=1, linestyle="--",
           label="rank threshold (1e-6 × σ_max)")
ax.set_xlabel("Index")
ax.set_ylabel("Singular value (log scale)")
ax.legend(fontsize=9)
ax.grid(True, which="both", alpha=0.3)

# ── Panel D: adjoint vs finite-difference ─────────────────────────────────
ax = axes[1, 1]
ax.set_title(f"D  Adjoint vs Finite-Difference  ({n_check} spot-checks)")
J_adj_vals = []
J_fd_vals  = []
rng2 = np.random.default_rng(0)
e_idx2 = rng2.integers(0, mesh.n_elements, n_check)
i_idx2 = rng2.integers(0, len(meas),       n_check)
for e_test, i_test in zip(e_idx2, i_idx2):
    sp = sigma.copy(); sp[e_test] += eps
    sm = sigma.copy(); sm[e_test] -= eps
    dVp = simulate(mesh, ec, sp, drive, meas, sigma0=sigma)
    dVm = simulate(mesh, ec, sm, drive, meas, sigma0=sigma)
    J_fd_vals.append((dVp[i_test] - dVm[i_test]) / (2 * eps))
    J_adj_vals.append(J[i_test, e_test])

J_adj_arr = np.array(J_adj_vals)
J_fd_arr  = np.array(J_fd_vals)
lims = [min(J_adj_arr.min(), J_fd_arr.min()),
        max(J_adj_arr.max(), J_fd_arr.max())]
ax.scatter(J_fd_arr, J_adj_arr, s=40, color="steelblue", zorder=5)
ax.plot(lims, lims, "r--", linewidth=1, label="ideal (y=x)")
ax.set_xlabel("J  (finite difference)")
ax.set_ylabel("J  (adjoint method)")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

plt.tight_layout()
out_path = "scripts/jacobian_verification.png"
plt.savefig(out_path, dpi=130, bbox_inches="tight")
print(f"Figure saved → {out_path}")
plt.show()
