"""
scripts/verify_protocol.py
===========================
Visual sanity-check for the ``eitkit.protocol`` module.

Run with:

    python scripts/verify_protocol.py

Produces a 2×2 figure and prints a stats table to stdout.
"""

from __future__ import annotations

import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

sys.path.insert(0, ".")

from eitkit.protocol import adjacent_pattern, measurement_pairs, add_noise

L = 16

drive   = adjacent_pattern(L)
meas    = measurement_pairs(L)

# ---------------------------------------------------------------------------
# Console stats
# ---------------------------------------------------------------------------

print("=" * 54)
print(f"  Protocol stats  (L = {L} electrodes)")
print("=" * 54)
print(f"  Drive steps              : {len(drive)}")
print(f"  Measurements / step      : {L - 3}")
print(f"  Total measurements       : {len(meas)}  (= {L}×{L-3})")
print(f"  Drive pairs dtype        : {drive.dtype}")
print(f"  Meas pairs dtype         : {meas.dtype}")

# SNR check
rng = np.random.default_rng(7)
v   = rng.normal(size=len(meas))
for snr_target in [20, 40, 60]:
    noisy   = add_noise(v, snr_db=snr_target, rng=np.random.default_rng(42))
    achieved = 20 * np.log10(np.linalg.norm(v) / np.linalg.norm(noisy - v))
    print(f"  add_noise  target={snr_target:2d} dB  achieved={achieved:.2f} dB")
print()

# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle("eitkit · Protocol Module Verification", fontsize=14, fontweight="bold")

# ── Panel A: drive pattern wheel ──────────────────────────────────────────
ax = axes[0, 0]
ax.set_title("A  Adjacent Drive Pattern (all 16 steps)")
theta = 2 * np.pi * np.arange(L) / L
xs = np.cos(theta)
ys = np.sin(theta)
# Draw electrode positions
ax.scatter(xs, ys, s=80, zorder=5, color="steelblue")
for i, (x, y) in enumerate(zip(xs, ys)):
    ax.text(x * 1.18, y * 1.18, str(i), fontsize=7.5, ha="center", va="center")
# Draw all drive arcs
cmap = plt.cm.hsv
for k, (src, snk) in enumerate(drive):
    color = cmap(k / L)
    ax.annotate(
        "", xy=(xs[snk], ys[snk]), xytext=(xs[src], ys[src]),
        arrowprops=dict(arrowstyle="->", color=color, lw=1.5),
    )
ax.set_xlim(-1.4, 1.4); ax.set_ylim(-1.4, 1.4)
ax.set_aspect("equal"); ax.axis("off")

# ── Panel B: measurement map ───────────────────────────────────────────────
ax = axes[0, 1]
ax.set_title("B  Measurement Map  (drive step × meas index)")
# Build full measurement matrix: rows = drive steps, cols = meas index within step
n_per_step = L - 3
mat = np.full((L, n_per_step), np.nan)
for k in range(L):
    rows_k = meas[meas[:, 0] == k]
    for j, row in enumerate(rows_k):
        mat[k, j] = row[1]   # colour by plus-electrode index

im = ax.imshow(mat, aspect="auto", cmap="tab20", vmin=0, vmax=L - 1)
ax.set_xlabel("Measurement index within drive step")
ax.set_ylabel("Drive step k")
ax.set_xticks(range(n_per_step))
ax.set_yticks(range(L))
cbar = fig.colorbar(im, ax=ax, ticks=range(0, L, 2))
cbar.set_label("Plus-electrode index")

# ── Panel C: noise — time-series view ─────────────────────────────────────
ax = axes[1, 0]
ax.set_title("C  AWGN Noise at Different SNRs")
v_demo = np.random.default_rng(3).normal(size=208)
ax.plot(v_demo, color="black", linewidth=0.8, label="clean", zorder=5)
for snr, color, alpha in [(20, "crimson", 0.7), (40, "darkorange", 0.7),
                           (60, "steelblue", 0.7)]:
    noisy = add_noise(v_demo, snr_db=snr, rng=np.random.default_rng(42))
    ax.plot(noisy, color=color, linewidth=0.5, alpha=alpha, label=f"SNR={snr} dB")
ax.set_xlabel("Measurement index")
ax.set_ylabel("Voltage (a.u.)")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# ── Panel D: noise distribution ───────────────────────────────────────────
ax = axes[1, 1]
ax.set_title("D  Noise Distribution  (SNR = 40 dB, 1000 realisations)")
residuals = []
for seed in range(1000):
    noisy = add_noise(v_demo, snr_db=40.0, rng=np.random.default_rng(seed))
    residuals.extend((noisy - v_demo).tolist())
residuals = np.array(residuals)
ax.hist(residuals, bins=80, density=True, color="steelblue", alpha=0.75,
        label="noise samples")
# Overlay theoretical Gaussian
sigma_est = residuals.std()
x_plot = np.linspace(residuals.min(), residuals.max(), 400)
from scipy.stats import norm
ax.plot(x_plot, norm.pdf(x_plot, 0, sigma_est), "crimson", linewidth=1.5,
        label=f"N(0, σ²)  σ={sigma_est:.4f}")
ax.set_xlabel("Noise value (a.u.)")
ax.set_ylabel("Density")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

plt.tight_layout()
out_path = "scripts/protocol_verification.png"
plt.savefig(out_path, dpi=130, bbox_inches="tight")
print(f"Figure saved → {out_path}")
plt.show()
