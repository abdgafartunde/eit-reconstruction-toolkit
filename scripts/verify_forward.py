"""
scripts/verify_forward.py
==========================
Visual sanity-check for the ``eitkit.forward`` module.

Run with:

    python scripts/verify_forward.py

Produces a 2×2 figure and prints a stats table to stdout.
"""

from __future__ import annotations

import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri

sys.path.insert(0, ".")

from eitkit.mesh import make_circle_mesh, place_electrodes
from eitkit.protocol import adjacent_pattern, measurement_pairs
from eitkit.forward import assemble_K, simulate
from eitkit.forward.gap_model import build_load_vector
from eitkit.forward.solver import solve_forward, pick_ground_node

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

mesh  = make_circle_mesh(n_electrodes=16, h0=0.10, seed=42)
ec    = place_electrodes(mesh, n_electrodes=16)
drive = adjacent_pattern(16)
meas  = measurement_pairs(16)

sigma_bg   = np.ones(mesh.n_elements)
centroids  = mesh.nodes[mesh.elements].mean(axis=1)

# Anomaly: conductive disc (σ=3) near top of domain
cx, cy, r_inc = 0.0, 0.55, 0.20
inclusion = ((centroids[:, 0] - cx) ** 2 + (centroids[:, 1] - cy) ** 2) < r_inc ** 2
sigma_anom = np.where(inclusion, 3.0, 1.0)

# ---------------------------------------------------------------------------
# Console stats
# ---------------------------------------------------------------------------

K    = assemble_K(mesh, sigma_bg)
g    = pick_ground_node(mesh)
f0   = build_load_vector(mesh.n_nodes, ec, drive_pair=[0, 1])
u0   = solve_forward(K, f0, g)

dV_bg   = simulate(mesh, ec, sigma_bg,  drive, meas, sigma0=1.0)
dV_anom = simulate(mesh, ec, sigma_anom, drive, meas, sigma0=1.0)

print("=" * 55)
print("  Forward solver verification")
print("=" * 55)
print(f"  Mesh nodes / elements  : {mesh.n_nodes} / {mesh.n_elements}")
print(f"  Stiffness matrix K     : {K.shape}, nnz={K.nnz}")
print(f"  Ground node            : {g}")
print(f"  Potential range (u)    : [{u0.min():.4f}, {u0.max():.4f}] V")
print(f"  dV uniform  max|δV|    : {np.abs(dV_bg).max():.2e}  (must be 0)")
print(f"  dV anomaly  max|δV|    : {np.abs(dV_anom).max():.4f} V")
print(f"  dV anomaly  range      : [{dV_anom.min():.4f}, {dV_anom.max():.4f}] V")
print()

# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle("eitkit · Forward Module Verification", fontsize=14, fontweight="bold")

triang = tri.Triangulation(mesh.nodes[:, 0], mesh.nodes[:, 1], mesh.elements)

# ── Panel A: conductivity phantom ─────────────────────────────────────────
ax = axes[0, 0]
ax.set_title("A  Conductivity Phantom  σ (S/m)")
tpc = ax.tripcolor(triang, sigma_anom, cmap="hot_r", shading="flat",
                   vmin=0.8, vmax=3.2)
fig.colorbar(tpc, ax=ax, label="σ (S/m)")
ax.triplot(triang, color="grey", linewidth=0.2, alpha=0.4)
el_xy = mesh.nodes[ec.node_indices]
ax.scatter(el_xy[:, 0], el_xy[:, 1], s=40, color="cyan", zorder=5)
ax.set_aspect("equal")
ax.set_xlabel("x"); ax.set_ylabel("y")

# ── Panel B: nodal potential map for drive step 0 ─────────────────────────
ax = axes[0, 1]
ax.set_title("B  Nodal Potential  u  (drive step 0, σ=anomaly)")
K_a  = assemble_K(mesh, sigma_anom)
u_a  = solve_forward(K_a, f0, g)
tpc2 = ax.tripcolor(triang, u_a, cmap="RdBu_r", shading="gouraud")
fig.colorbar(tpc2, ax=ax, label="Potential (V)")
ax.triplot(triang, color="grey", linewidth=0.2, alpha=0.3)
src_xy = mesh.nodes[int(ec.node_indices[0])]
snk_xy = mesh.nodes[int(ec.node_indices[1])]
ax.scatter(*src_xy, s=120, color="red",  zorder=6, label="+I (source)")
ax.scatter(*snk_xy, s=120, color="blue", zorder=6, label="−I (sink)")
ax.legend(fontsize=8, loc="lower right")
ax.set_aspect("equal")
ax.set_xlabel("x"); ax.set_ylabel("y")

# ── Panel C: difference voltage spectrum ──────────────────────────────────
ax = axes[1, 0]
ax.set_title("C  Difference Voltages δV  (all 208 measurements)")
ax.plot(dV_anom, color="steelblue", linewidth=0.8, label="anomaly")
ax.axhline(0, color="grey", linewidth=0.6, linestyle="--")
ax.fill_between(range(len(dV_anom)), dV_anom, 0,
                where=(dV_anom > 0), color="steelblue", alpha=0.25)
ax.fill_between(range(len(dV_anom)), dV_anom, 0,
                where=(dV_anom < 0), color="crimson",  alpha=0.25)
ax.set_xlabel("Measurement index")
ax.set_ylabel("δV  (V)")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
# Shade drive steps
for k in range(16):
    if k % 2 == 0:
        ax.axvspan(k * 13, (k + 1) * 13, color="lightyellow", zorder=0)

# ── Panel D: δV per drive step (heat-map) ─────────────────────────────────
ax = axes[1, 1]
ax.set_title("D  δV Heat-map  (drive step × meas index)")
mat = dV_anom.reshape(16, 13)
im  = ax.imshow(mat, aspect="auto", cmap="RdBu_r",
                vmax=np.abs(mat).max(), vmin=-np.abs(mat).max())
fig.colorbar(im, ax=ax, label="δV (V)")
ax.set_xlabel("Measurement index within drive step")
ax.set_ylabel("Drive step k")
ax.set_xticks(range(13))
ax.set_yticks(range(16))

plt.tight_layout()
out_path = "scripts/forward_verification.png"
plt.savefig(out_path, dpi=130, bbox_inches="tight")
print(f"Figure saved → {out_path}")
plt.show()
