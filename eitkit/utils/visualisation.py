"""
eitkit.utils.visualisation
===========================
Reusable Matplotlib helpers for 2-D EIT visualisation.

All functions return ``(fig, ax)`` so the caller can further customise or
save the figure.  Passing an existing ``ax`` avoids creating a new figure,
making it easy to embed panels in multi-subplot layouts.

Public API
----------
plot_mesh(mesh, ax, **kwargs)
    Draw the triangulation and highlight boundary / electrode nodes.
plot_conductivity(mesh, sigma, ax, **kwargs)
    Colour-map a per-element conductivity distribution on the mesh.
plot_voltages(meas_pairs, dV, n_electrodes, ax, **kwargs)
    2-D heatmap of difference voltages (drive steps × measurement pairs).
    The characteristic hot diagonal arises because electrode pairs adjacent
    to the current-injection site carry the strongest signal.
"""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import matplotlib.tri as tri
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy.typing import NDArray

from eitkit.mesh.electrode_placement import ElectrodeConfig
from eitkit.mesh.mesh import Mesh

__all__ = ["plot_mesh", "plot_conductivity", "plot_voltages"]


def plot_mesh(
    mesh: Mesh,
    elec_config: ElectrodeConfig | None = None,
    ax: Axes | None = None,
    title: str = "Mesh",
    linewidth: float = 0.4,
    mesh_color: str = "steelblue",
    boundary_color: str = "dimgrey",
    electrode_color: str = "crimson",
    **kwargs: Any,
) -> tuple[Figure, Axes]:
    """Plot the triangular mesh with optional electrode overlay.

    Parameters
    ----------
    mesh:
        Triangular mesh to visualise.
    elec_config:
        If supplied, electrode nodes are highlighted as large dots and
        labelled 1-indexed.
    ax:
        Existing Axes to draw into.  If ``None`` a new figure is created.
    title:
        Axes title.
    linewidth:
        Triangle edge line width.
    mesh_color:
        Colour of triangle edges.
    boundary_color:
        Colour of boundary node scatter.
    electrode_color:
        Colour of electrode node scatter and labels.

    Returns
    -------
    fig, ax : Figure, Axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 5))
    else:
        fig = ax.get_figure()

    triang = tri.Triangulation(mesh.nodes[:, 0], mesh.nodes[:, 1], mesh.elements)
    ax.triplot(triang, color=mesh_color, linewidth=linewidth, alpha=0.7, **kwargs)

    # Boundary nodes
    bnd_xy = mesh.nodes[mesh.boundary_nodes]
    ax.scatter(
        bnd_xy[:, 0],
        bnd_xy[:, 1],
        s=8,
        color=boundary_color,
        zorder=3,
        label="boundary nodes",
    )

    # Electrode nodes
    if elec_config is not None:
        el_xy = mesh.nodes[elec_config.node_indices]
        ax.scatter(
            el_xy[:, 0],
            el_xy[:, 1],
            s=60,
            color=electrode_color,
            zorder=5,
            label="electrodes",
        )
        # Place labels just outside the boundary — use the actual
        # boundary radius rather than a hard-coded scale factor.
        bnd_radius = float(np.mean(np.sqrt(
            mesh.nodes[mesh.boundary_nodes, 0] ** 2
            + mesh.nodes[mesh.boundary_nodes, 1] ** 2
        )))
        label_offset = bnd_radius * 1.10 if bnd_radius > 0 else 1.10
        for i, (x, y) in enumerate(el_xy):
            r = np.hypot(x, y)
            if r < 1e-12:
                r = 1.0
            scale = label_offset / r
            ax.text(
                x * scale,
                y * scale,
                str(i + 1),
                fontsize=6,
                ha="center",
                va="center",
                color=electrode_color,
            )

    ax.set_aspect("equal")
    ax.set_title(title)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    if elec_config is not None:
        ax.legend(fontsize=7, loc="lower right")
    return fig, ax


def plot_conductivity(
    mesh: Mesh,
    sigma: NDArray[np.floating],
    ax: Axes | None = None,
    title: str = "Conductivity σ (S/m)",
    cmap: str = "hot_r",
    show_mesh: bool = False,
    elec_config: ElectrodeConfig | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    **kwargs: Any,
) -> tuple[Figure, Axes]:
    """Colour-map a per-element conductivity distribution on the mesh.

    Parameters
    ----------
    mesh:
        Triangular mesh.
    sigma:
        Per-element conductivity, shape ``(M,)``.
    ax:
        Existing Axes to draw into.  If ``None`` a new figure is created.
    title:
        Axes title.
    cmap:
        Matplotlib colour-map name.
    show_mesh:
        If ``True``, overlay triangle edges in grey.
    elec_config:
        If supplied, electrode nodes are marked.
    vmin, vmax:
        Colour scale limits.  Defaults to the data range.

    Returns
    -------
    fig, ax : Figure, Axes
    """
    sigma = np.asarray(sigma, dtype=np.float64)
    if sigma.shape != (mesh.n_elements,):
        raise ValueError(
            f"sigma must have shape ({mesh.n_elements},), got {sigma.shape}."
        )

    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 5))
    else:
        fig = ax.get_figure()

    triang = tri.Triangulation(mesh.nodes[:, 0], mesh.nodes[:, 1], mesh.elements)
    _vmin = float(sigma.min()) if vmin is None else vmin
    _vmax = float(sigma.max()) if vmax is None else vmax

    tpc = ax.tripcolor(
        triang, sigma, cmap=cmap, shading="flat", vmin=_vmin, vmax=_vmax, **kwargs
    )
    fig.colorbar(tpc, ax=ax, label="σ (S/m)")

    if show_mesh:
        ax.triplot(triang, color="grey", linewidth=0.2, alpha=0.4)

    if elec_config is not None:
        el_xy = mesh.nodes[elec_config.node_indices]
        ax.scatter(
            el_xy[:, 0], el_xy[:, 1], s=40, color="cyan", zorder=5, label="electrodes"
        )
        ax.legend(fontsize=7, loc="lower right")

    ax.set_aspect("equal")
    ax.set_title(title)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    return fig, ax


def plot_voltages(
    meas_pairs: NDArray[np.int32],
    dV: NDArray[np.floating],
    n_electrodes: int = 16,
    ax: Axes | None = None,
    title: str = "Difference Voltages δV",
    cmap: str = "RdBu_r",
    highlight_drive: int | None = None,
    symmetric: bool = True,
    **kwargs: Any,
) -> tuple[Figure, Axes]:
    """Plot difference voltages as an :math:`L \\times L` matrix.

    Each measurement is placed at ``(drive_step, plus_electrode)`` in an
    :math:`L \\times L` grid.  Unmeasured electrode pairs (driven
    electrodes and their immediate neighbours) are shown in light grey.
    For adjacent-drive EIT the largest voltage magnitudes cluster in
    off-diagonal bands, giving the characteristic diagonal pattern seen
    in the literature.

    Parameters
    ----------
    meas_pairs:
        Measurement schedule, shape ``(P, 3)`` — columns
        ``[drive_step, plus_el, minus_el]``.
    dV:
        Difference voltage vector, shape ``(P,)``.
    n_electrodes:
        Number of electrodes :math:`L`.
    ax:
        Existing Axes to draw into.  If ``None`` a new figure is created.
    title:
        Axes title.
    cmap:
        Matplotlib colour-map.  A diverging map (e.g. ``'RdBu_r'``) works
        best because δV is signed.
    highlight_drive:
        If set, draw a horizontal line at that drive-step row.
    symmetric:
        If ``True`` (default) the colour scale is symmetric about zero so
        that positive and negative values use equal saturation.

    Returns
    -------
    fig, ax : Figure, Axes
    """
    dV = np.asarray(dV, dtype=np.float64)
    L = n_electrodes
    expected = L * (L - 3)
    if len(dV) != expected:
        raise ValueError(
            f"len(dV)={len(dV)} does not match "
            f"L × (L−3) = {expected} for L={L}."
        )

    # Build L×L matrix — NaN where no measurement exists
    mat = np.full((L, L), np.nan, dtype=np.float64)
    for i in range(len(meas_pairs)):
        k = int(meas_pairs[i, 0])        # drive step
        plus_el = int(meas_pairs[i, 1])   # plus-electrode
        mat[k, plus_el] = dV[i]

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5.5))
    else:
        fig = ax.get_figure()

    vmax = float(np.nanmax(np.abs(mat))) if symmetric else None
    vmin = -vmax if symmetric else None

    # Use a masked array so NaN cells get the 'bad' colour
    mat_masked = np.ma.masked_invalid(mat)
    # Light grey for unmeasured cells
    current_cmap = plt.colormaps.get_cmap(cmap).copy()
    current_cmap.set_bad(color="0.85")

    im = ax.imshow(
        mat_masked,
        aspect="equal",
        origin="upper",
        cmap=current_cmap,
        vmin=vmin,
        vmax=vmax,
        interpolation="nearest",
        **kwargs,
    )
    cbar = fig.colorbar(im, ax=ax, label="δV (V)", shrink=0.82)

    # Thin grid to delineate cells
    ax.set_xticks(np.arange(-0.5, L, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, L, 1), minor=True)
    ax.grid(which="minor", color="0.7", linewidth=0.3, alpha=0.6)
    ax.tick_params(which="minor", size=0)

    if highlight_drive is not None:
        ax.axhline(
            highlight_drive - 0.5, color="gold", linewidth=2.0, linestyle="--"
        )
        ax.axhline(
            highlight_drive + 0.5,
            color="gold",
            linewidth=2.0,
            linestyle="--",
            label=f"drive step {highlight_drive}",
        )
        ax.legend(fontsize=8, loc="upper right")

    ax.set_title(title)
    ax.set_xlabel("Measurement electrode  (+ terminal)")
    ax.set_ylabel("Drive step  (source electrode)")
    ax.set_xticks(range(0, L, max(1, L // 8)))
    ax.set_yticks(range(0, L, max(1, L // 8)))
    return fig, ax
