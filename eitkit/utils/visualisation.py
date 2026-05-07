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

from eitkit.mesh.mesh import Mesh
from eitkit.mesh.electrode_placement import ElectrodeConfig

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

    triang = tri.Triangulation(
        mesh.nodes[:, 0], mesh.nodes[:, 1], mesh.elements
    )
    ax.triplot(triang, color=mesh_color, linewidth=linewidth, alpha=0.7,
               **kwargs)

    # Boundary nodes
    bnd_xy = mesh.nodes[mesh.boundary_nodes]
    ax.scatter(bnd_xy[:, 0], bnd_xy[:, 1], s=8, color=boundary_color,
               zorder=3, label="boundary nodes")

    # Electrode nodes
    if elec_config is not None:
        el_xy = mesh.nodes[elec_config.node_indices]
        ax.scatter(el_xy[:, 0], el_xy[:, 1], s=60, color=electrode_color,
                   zorder=5, label="electrodes")
        for i, (x, y) in enumerate(el_xy):
            ax.text(
                x * 1.10, y * 1.10, str(i + 1),
                fontsize=6, ha="center", va="center", color=electrode_color,
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

    triang = tri.Triangulation(
        mesh.nodes[:, 0], mesh.nodes[:, 1], mesh.elements
    )
    _vmin = float(sigma.min()) if vmin is None else vmin
    _vmax = float(sigma.max()) if vmax is None else vmax

    tpc = ax.tripcolor(triang, sigma, cmap=cmap, shading="flat",
                       vmin=_vmin, vmax=_vmax, **kwargs)
    fig.colorbar(tpc, ax=ax, label="σ (S/m)")

    if show_mesh:
        ax.triplot(triang, color="grey", linewidth=0.2, alpha=0.4)

    if elec_config is not None:
        el_xy = mesh.nodes[elec_config.node_indices]
        ax.scatter(el_xy[:, 0], el_xy[:, 1], s=40, color="cyan",
                   zorder=5, label="electrodes")
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
    """Plot difference voltages as a 2-D heatmap.

    The δV vector is reshaped to ``(n_electrodes, n_electrodes − 3)`` so
    that rows correspond to drive steps and columns to measurement pairs
    within each step.  For adjacent-drive EIT the largest magnitudes cluster
    near the **main diagonal** because the electrode pairs closest to the
    current-injection site carry the strongest signal.

    Parameters
    ----------
    meas_pairs:
        Measurement schedule, shape ``(P, 3)`` — columns
        ``[drive_step, plus_el, minus_el]``.
    dV:
        Difference voltage vector, shape ``(P,)``.
    n_electrodes:
        Number of electrodes; determines the heatmap shape
        ``(n_electrodes, n_electrodes − 3)``.
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
    n_per_step = n_electrodes - 3
    expected   = n_electrodes * n_per_step
    if len(dV) != expected:
        raise ValueError(
            f"len(dV)={len(dV)} does not match "
            f"n_electrodes × (n_electrodes−3) = {expected}."
        )

    # Reshape: rows = drive steps, cols = meas within each step
    mat = dV.reshape(n_electrodes, n_per_step)

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
    else:
        fig = ax.get_figure()

    vmax = float(np.abs(mat).max()) if symmetric else None
    vmin = -vmax if symmetric else None

    im = ax.imshow(
        mat,
        aspect="auto",
        origin="upper",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        interpolation="nearest",
        **kwargs,
    )
    fig.colorbar(im, ax=ax, label="δV (V)", shrink=0.85)

    if highlight_drive is not None:
        ax.axhline(highlight_drive - 0.5, color="gold", linewidth=1.8,
                   linestyle="--")
        ax.axhline(highlight_drive + 0.5, color="gold", linewidth=1.8,
                   linestyle="--", label=f"drive step {highlight_drive}")
        ax.legend(fontsize=8, loc="upper right")

    ax.set_title(title)
    ax.set_xlabel("Measurement pair index within drive step")
    ax.set_ylabel("Drive step")
    ax.set_yticks(range(0, n_electrodes, max(1, n_electrodes // 8)))
    return fig, ax
