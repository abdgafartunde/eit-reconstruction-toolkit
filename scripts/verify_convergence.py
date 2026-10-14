"""
scripts/verify_convergence.py
=============================
Mesh-refinement study for the ``eitkit.forward`` module.

Run with:

    python scripts/verify_convergence.py

It runs from any directory. Use ``--outdir DIR`` to choose where the
figure is saved (default: this scripts directory) and ``--show`` to also
open it in a window.

On the unit disc with sigma = 1, point currents +I at y_s and -I at y_t
give the exact potential

    u(x) = -(I / pi) * ( ln|x - y_s| - ln|x - y_t| ) + const.

For every drive step of the adjacent protocol, the script compares the FEM
electrode voltages (measured electrodes are never driven) with this closed
form, prints a table, and plots the relative error against h0.
"""

from __future__ import annotations

import time

import matplotlib.pyplot as plt
import numpy as np

from _cli import parse_args  # noqa: E402  (puts the repository root on sys.path)

args = parse_args("Mesh-refinement study for the eitkit forward model.")

from eitkit.forward import assemble_K  # noqa: E402
from eitkit.forward.gap_model import build_load_vector  # noqa: E402
from eitkit.forward.solver import pick_ground_node, solve_forward  # noqa: E402
from eitkit.mesh import make_circle_mesh, place_electrodes  # noqa: E402
from eitkit.protocol import adjacent_pattern, measurement_pairs  # noqa: E402

L = 16
H0_VALUES = [0.2, 0.14, 0.1, 0.07, 0.05, 0.035, 0.025]


def exact_potential(x, y_source, y_sink, current=1.0):
    return -(current / np.pi) * (
        np.log(np.linalg.norm(x - y_source, axis=-1))
        - np.log(np.linalg.norm(x - y_sink, axis=-1))
    )


drives = adjacent_pattern(L)
meas = measurement_pairs(L)
rows = []
print(f"{'h0':>6} {'nodes':>6} {'boundary':>8} {'area/pi':>9} {'rel. error':>11} {'rate':>6} {'time':>7}")
for h0 in H0_VALUES:
    t0 = time.perf_counter()
    mesh = make_circle_mesh(n_electrodes=L, h0=h0, seed=42)
    ec = place_electrodes(mesh, L)
    K = assemble_K(mesh, np.ones(mesh.n_elements))
    ground = pick_ground_node(mesh)
    pos = mesh.nodes[ec.node_indices]
    errors = []
    for k, (src, snk) in enumerate(drives):
        u = solve_forward(K, build_load_vector(mesh.n_nodes, ec, (src, snk)), ground)
        sel = meas[meas[:, 0] == k]
        plus, minus = ec.node_indices[sel[:, 1]], ec.node_indices[sel[:, 2]]
        exact = exact_potential(mesh.nodes[plus], pos[src], pos[snk]) - exact_potential(
            mesh.nodes[minus], pos[src], pos[snk]
        )
        errors.append(np.linalg.norm((u[plus] - u[minus]) - exact) / np.linalg.norm(exact))
    err = float(np.mean(errors))
    rate = np.log(rows[-1][4] / err) / np.log(rows[-1][0] / h0) if rows else float("nan")
    rows.append((h0, mesh.n_nodes, mesh.n_boundary_nodes, mesh.areas.sum() / np.pi, err))
    print(f"{h0:6.3f} {mesh.n_nodes:6d} {mesh.n_boundary_nodes:8d} {rows[-1][3]:9.5f} "
          f"{err:11.3e} {rate:6.2f} {time.perf_counter() - t0:6.1f}s")

h = np.array([r[0] for r in rows])
e = np.array([r[4] for r in rows])
fig, ax = plt.subplots(figsize=(6, 4.5))
ax.loglog(h, e, "o-", color="steelblue", label="FEM electrode voltages")
ax.loglog(h, e[0] * (h / h[0]) ** 2, "--", color="0.5", label=r"$O(h_0^2)$ reference")
ax.invert_xaxis()
ax.set_xlabel(r"target edge length $h_0$")
ax.set_ylabel("mean relative error vs. exact solution")
ax.set_title("eitkit · forward-model convergence (unit disc, sigma = 1)")
ax.grid(True, which="both", alpha=0.3)
ax.legend()
fig.tight_layout()

out_path = args.outdir / "convergence_verification.png"
fig.savefig(out_path, dpi=130, bbox_inches="tight")
print(f"Figure saved → {out_path}")
if args.show:
    plt.show()
