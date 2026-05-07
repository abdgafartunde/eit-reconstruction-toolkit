"""
scripts/verify_utils.py
========================
Visual sanity-check for the ``eitkit.utils`` module.

Run with:

    python scripts/verify_utils.py

Panels
------
A  Mesh + electrode layout
B  circle + ellipse phantom  (with exact shape outlines)
C  rectangle + ring phantom  (with exact shape outlines)
D  triangle phantom + δV comparison across all phantoms
"""

from __future__ import annotations

import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.transforms import Affine2D

sys.path.insert(0, ".")

from eitkit.mesh import make_circle_mesh, place_electrodes
from eitkit.protocol import adjacent_pattern, measurement_pairs
from eitkit.forward import simulate
from eitkit.utils import make_phantom, plot_mesh, plot_conductivity, plot_voltages

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

mesh  = make_circle_mesh(n_electrodes=16, h0=0.05, seed=42)
ec    = place_electrodes(mesh, n_electrodes=16)
drive = adjacent_pattern(16)
meas  = measurement_pairs(16)

sigma_circ_ellipse = make_phantom(mesh, [
    {"shape": "circle",  "cx":  0.35, "cy":  0.35, "r": 0.25, "sigma": 3.0},
    {"shape": "ellipse", "cx": -0.35, "cy": -0.35, "a": 0.30, "b": 0.15,
     "theta": np.pi / 5, "sigma": 0.2},
])

sigma_rect_ring = make_phantom(mesh, [
    {"shape": "rectangle", "cx":  0.20, "cy":  0.30, "w": 0.25, "h": 0.15,
     "theta": np.pi / 6, "sigma": 3.0},
    {"shape": "ring", "cx": -0.30, "cy": -0.25,
     "r_inner": 0.10, "r_outer": 0.28, "sigma": 0.2},
])

sigma_triangle = make_phantom(mesh, [
    {"shape": "triangle", "cx": 0.00, "cy": 0.20, "side": 0.50,
     "theta": 0.0, "sigma": 3.0},
])

dV_ce = simulate(mesh, ec, sigma_circ_ellipse, drive, meas, sigma0=1.0)
dV_rr = simulate(mesh, ec, sigma_rect_ring,    drive, meas, sigma0=1.0)
dV_tr = simulate(mesh, ec, sigma_triangle,     drive, meas, sigma0=1.0)

# ---------------------------------------------------------------------------
# Console stats
# ---------------------------------------------------------------------------

print("=" * 60)
print("  Utils module verification  (5 shapes)")
print("=" * 60)
print(f"  Mesh          : {mesh.n_nodes} nodes, {mesh.n_elements} elements")
for label, s in [
    ("circle + ellipse", sigma_circ_ellipse),
    ("rect + ring",      sigma_rect_ring),
    ("triangle",         sigma_triangle),
]:
    uniq = np.unique(s).round(3)
    print(f"  Phantom [{label:16s}]: unique σ = {uniq}")
for label, dv in [("circle+ellipse", dV_ce),
                   ("rect+ring",      dV_rr),
                   ("triangle",       dV_tr)]:
    print(f"  dV [{label:16s}]: [{dv.min():.5f}, {dv.max():.5f}] V")
print()

# ---------------------------------------------------------------------------
# Helper: dashed outline patches
# ---------------------------------------------------------------------------

def _circle_patch(cx, cy, r, **kw):
    return mpatches.Circle((cx, cy), r, fill=False,
                            linestyle="--", linewidth=1.4, **kw)

def _ellipse_patch(cx, cy, a, b, theta_rad, **kw):
    return mpatches.Ellipse((cx, cy), 2 * a, 2 * b,
                             angle=np.degrees(theta_rad),
                             fill=False, linestyle="--", linewidth=1.4, **kw)

def _rect_patch(cx, cy, w, h, theta_rad, **kw):
    # Rectangle anchor = lower-left corner; we place centre, then rotate
    rect = mpatches.FancyBboxPatch(
        (cx - w, cy - h), 2 * w, 2 * h,
        boxstyle="square,pad=0",
        fill=False, linestyle="--", linewidth=1.4, **kw)
    t = (Affine2D().rotate_around(cx, cy, theta_rad)
         + rect.axes.transData if rect.axes else Affine2D())
    # Return a rotated rectangle via Polygon
    corners_local = np.array([
        [-w, -h], [+w, -h], [+w, +h], [-w, +h], [-w, -h]
    ])
    cos_t, sin_t = np.cos(theta_rad), np.sin(theta_rad)
    rot = np.array([[cos_t, -sin_t], [sin_t, cos_t]])
    corners_world = (rot @ corners_local.T).T + np.array([cx, cy])
    return mpatches.Polygon(corners_world, closed=True,
                             fill=False, linestyle="--", linewidth=1.4, **kw)

def _ring_patches(cx, cy, r_inner, r_outer, **kw):
    outer = mpatches.Circle((cx, cy), r_outer, fill=False,
                             linestyle="--", linewidth=1.4, **kw)
    inner = mpatches.Circle((cx, cy), r_inner, fill=False,
                             linestyle="--", linewidth=1.4, **kw)
    return outer, inner

def _triangle_patch(cx, cy, side, theta_rad, **kw):
    R = side / np.sqrt(3.0)
    angles = np.array([np.pi/2, np.pi/2 + 2*np.pi/3,
                       np.pi/2 + 4*np.pi/3]) + theta_rad
    verts = np.stack([cx + R * np.cos(angles),
                      cy + R * np.sin(angles)], axis=1)
    return mpatches.Polygon(verts, closed=True,
                             fill=False, linestyle="--", linewidth=1.4, **kw)

# ---------------------------------------------------------------------------
# Figure  (2 × 2)
# ---------------------------------------------------------------------------

fig, axes = plt.subplots(2, 2, figsize=(13, 10))
fig.suptitle(
    "eitkit · Utils Module Verification  (5 inclusion shapes)\n"
    "Dashed outlines = exact theoretical boundaries; "
    "coloured fill = FEM piecewise-constant approximation",
    fontsize=12, fontweight="bold")

# ── Panel A: mesh ─────────────────────────────────────────────────────────
plot_mesh(mesh, elec_config=ec, ax=axes[0, 0],
          title="A  Mesh + Electrode Layout  (h₀ = 0.05)")

# ── Panel B: circle + ellipse ─────────────────────────────────────────────
_, ax_b = plot_conductivity(mesh, sigma_circ_ellipse, ax=axes[0, 1],
                             elec_config=ec, show_mesh=False,
                             title="B  circle (σ=3) + ellipse (σ=0.2)",
                             cmap="RdYlBu_r", vmin=0.1, vmax=3.2)
ax_b.add_patch(_circle_patch(0.35, 0.35, 0.25, color="darkred"))
ax_b.add_patch(_ellipse_patch(-0.35, -0.35, 0.30, 0.15,
                               np.pi / 5, color="navy"))

# ── Panel C: rectangle + ring ─────────────────────────────────────────────
_, ax_c = plot_conductivity(mesh, sigma_rect_ring, ax=axes[1, 0],
                             elec_config=ec, show_mesh=False,
                             title="C  rectangle (σ=3) + ring (σ=0.2)",
                             cmap="RdYlBu_r", vmin=0.1, vmax=3.2)
ax_c.add_patch(_rect_patch(0.20, 0.30, 0.25, 0.15,
                            np.pi / 6, color="darkred"))
for p in _ring_patches(-0.30, -0.25, 0.10, 0.28, color="navy"):
    ax_c.add_patch(p)

# ── Panel D: δV heatmap — circle+ellipse phantom (shows hot diagonal) ────
plot_voltages(meas, dV_ce, n_electrodes=16, ax=axes[1, 1],
              title="D  δV heatmap — circle+ellipse phantom\n"
                    "(hot diagonal = max sensitivity near drive pair)",
              highlight_drive=4)

plt.tight_layout()
out_path = "scripts/utils_verification.png"
plt.savefig(out_path, dpi=130, bbox_inches="tight")
print(f"Figure saved → {out_path}")
plt.show()


