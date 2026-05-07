"""
scripts/verify_mesh.py
======================
Quick visual sanity-check for the ``eitkit.mesh`` module.

Run with:

    python scripts/verify_mesh.py

Produces a 2×2 figure and prints a stats table to stdout.
No EIT physics yet — this purely exercises mesh generation and electrode
placement.
"""

from __future__ import annotations

import sys
import textwrap

import matplotlib.pyplot as plt
import matplotlib.tri as tri
import numpy as np

# Make sure the package is importable when run from the project root
sys.path.insert(0, ".")

from eitkit.mesh import make_circle_mesh, place_electrodes  # noqa: E402

# ---------------------------------------------------------------------------
# Generate three meshes at different densities
# ---------------------------------------------------------------------------

meshes = {
    "coarse  (h0=0.20)": make_circle_mesh(n_electrodes=16, h0=0.20, seed=42),
    "medium  (h0=0.12)": make_circle_mesh(n_electrodes=16, h0=0.12, seed=42),
    "fine    (h0=0.08)": make_circle_mesh(n_electrodes=16, h0=0.08, seed=42),
}

# Working mesh for detailed panels
mesh = meshes["medium  (h0=0.12)"]
ec = place_electrodes(mesh, n_electrodes=16)

# ---------------------------------------------------------------------------
# Console stats
# ---------------------------------------------------------------------------

header = f"{'Mesh':<20}  {'nodes':>7}  {'elems':>7}  {'bnd nodes':>10}  " \
         f"{'min area':>12}  {'max area':>12}  {'all CCW':>8}"
separator = "-" * len(header)

print(separator)
print(header)
print(separator)

for label, m in meshes.items():
    all_ccw = "YES" if np.all(m.areas > 0) else "NO !!!"
    print(
        f"{label:<20}  {m.n_nodes:>7d}  {m.n_elements:>7d}  "
        f"{m.n_boundary_nodes:>10d}  {m.areas.min():>12.6f}  "
        f"{m.areas.max():>12.6f}  {all_ccw:>8s}"
    )

print(separator)

print(textwrap.dedent(f"""
Electrode placement  (medium mesh, 16 electrodes, gap model)
  node indices : {ec.node_indices.tolist()}
  arc width    : {ec.arc_width:.6f} rad  (ideal = {np.pi/16:.6f} rad)
  angle range  : [{ec.angles.min():.4f}, {ec.angles.max():.4f}] rad
"""))

# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle("eitkit · Mesh Module Verification", fontsize=14, fontweight="bold")

# ── Panel A: triangulation with electrodes ─────────────────────────────────
ax = axes[0, 0]
ax.set_title("A  Triangulation + Electrodes (medium mesh)")
triang = tri.Triangulation(mesh.nodes[:, 0], mesh.nodes[:, 1], mesh.elements)
ax.triplot(triang, color="steelblue", linewidth=0.4, alpha=0.7)
# Boundary nodes
bnd_xy = mesh.nodes[mesh.boundary_nodes]
ax.scatter(bnd_xy[:, 0], bnd_xy[:, 1], s=6, color="grey", zorder=3,
           label="boundary nodes")
# Electrode nodes
el_xy = mesh.nodes[ec.node_indices]
ax.scatter(el_xy[:, 0], el_xy[:, 1], s=60, color="crimson", zorder=5,
           label="electrodes")
# Label electrodes 1-indexed
for i, (x, y) in enumerate(el_xy):
    ax.text(x * 1.08, y * 1.08, str(i + 1), fontsize=6.5, ha="center",
            va="center", color="crimson")
ax.set_aspect("equal")
ax.legend(fontsize=8, loc="lower right")
ax.set_xlabel("x"); ax.set_ylabel("y")

# ── Panel B: element area histogram ───────────────────────────────────────
ax = axes[0, 1]
ax.set_title("B  Element Area Distribution")
for label, m in meshes.items():
    ax.hist(m.areas, bins=40, alpha=0.55, label=label.strip())
ax.set_xlabel("Element area (m²)")
ax.set_ylabel("Count")
ax.legend(fontsize=8)
ax.axvline(0, color="k", linewidth=0.8, linestyle="--")

# ── Panel C: mesh density vs h0 ───────────────────────────────────────────
ax = axes[1, 0]
ax.set_title("C  Mesh Density vs Target Edge Length h₀")
h0_vals = [0.20, 0.17, 0.14, 0.12, 0.10, 0.08]
n_elems = []
n_nodes = []
for h in h0_vals:
    m = make_circle_mesh(n_electrodes=16, h0=h, seed=42)
    n_elems.append(m.n_elements)
    n_nodes.append(m.n_nodes)
ax.plot(h0_vals, n_elems, "o-", color="steelblue", label="elements")
ax.plot(h0_vals, n_nodes, "s--", color="darkorange", label="nodes")
ax.invert_xaxis()
ax.set_xlabel("h₀ (target edge length)")
ax.set_ylabel("Count")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# ── Panel D: electrode angle uniformity ───────────────────────────────────
ax = axes[1, 1]
ax.set_title("D  Electrode Angular Uniformity (medium mesh)")
# Normalise actual angles to [0, 2π) so they compare fairly with ideal
actual_norm = ec.angles % (2 * np.pi)
ideal_angles = 2 * np.pi * np.arange(16) / 16
# Sort ideal to pair with the sorted actual so the comparison is meaningful
actual_sorted = np.sort(actual_norm)
ideal_sorted = np.sort(ideal_angles)
# Angular distance (handles any residual wrap-around near 0/2π)
raw_diff = np.abs(actual_sorted - ideal_sorted)
deviation_deg = np.degrees(np.minimum(raw_diff, 2 * np.pi - raw_diff))
colors = ["crimson" if d > 1.0 else "steelblue" for d in deviation_deg]
bars = ax.bar(np.arange(1, 17), deviation_deg, color=colors)
ax.axhline(1.0, color="crimson", linewidth=1, linestyle="--",
           label="1° threshold")
ax.set_xlabel("Electrode index")
ax.set_ylabel("Deviation from ideal angle (degrees)")
ax.set_xticks(np.arange(1, 17))
ax.legend(fontsize=9)
ax.grid(True, axis="y", alpha=0.3)

plt.tight_layout()
out_path = "scripts/mesh_verification.png"
plt.savefig(out_path, dpi=130, bbox_inches="tight")
print(f"Figure saved → {out_path}")
plt.show()
