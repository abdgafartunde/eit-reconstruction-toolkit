"""
eitkit.forward.gap_model
=========================
Neumann boundary conditions for the **gap electrode model**.

In the gap model each electrode is represented by a single boundary node.
Current injection is modelled as a Neumann condition: a point current
source/sink of magnitude ``+I`` / ``-I`` at the driven electrode nodes.
This is equivalent to adding ``±I`` directly to the corresponding entries
of the load vector ``f`` (the right-hand side of ``Ku = f``).

No current flows at any other boundary node (homogeneous Neumann —
the default if we simply leave those entries at zero).

The Dirichlet ground condition (unique solution) is **not** applied here;
it is handled in :mod:`eitkit.forward.solver` just before the linear
solve, so this module stays pure Neumann.

Public API
----------
build_load_vector(n_nodes, elec_config, drive_pair, current) → ndarray (N,)
    Construct the load vector for one drive step.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from eitkit.mesh.electrode_placement import ElectrodeConfig

__all__ = ["build_load_vector"]


def build_load_vector(
    n_nodes: int,
    elec_config: ElectrodeConfig,
    drive_pair: NDArray[np.int32] | tuple[int, int],
    current: float = 1.0,
) -> NDArray[np.float64]:
    """Return the Neumann load vector ``f`` for one stimulation step.

    The source electrode receives ``+current`` and the sink electrode
    receives ``-current``; all other entries are zero.

    Parameters
    ----------
    n_nodes:
        Total number of mesh nodes ``N``.
    elec_config:
        Electrode layout produced by :func:`~eitkit.mesh.place_electrodes`.
    drive_pair:
        Length-2 sequence ``[source_electrode_index, sink_electrode_index]``
        where indices are **electrode** indices (0-based into
        ``elec_config.node_indices``), **not** mesh node indices.
    current:
        Injected current in Amperes.  Defaults to 1.0 A.

    Returns
    -------
    f : ndarray, shape ``(N,)``, dtype ``float64``
        Load vector with ``f[source_node] = +current``,
        ``f[sink_node] = -current``, all other entries zero.

    Raises
    ------
    ValueError
        If ``drive_pair`` contains electrode indices out of range.

    Examples
    --------
    >>> # With 4 electrodes placed at nodes [10, 20, 30, 40]:
    >>> # drive_pair = [0, 1] → f[10] = +1, f[20] = -1
    """
    src_el, snk_el = int(drive_pair[0]), int(drive_pair[1])
    n_el = elec_config.n_electrodes

    if not (0 <= src_el < n_el) or not (0 <= snk_el < n_el):
        raise ValueError(
            f"drive_pair electrodes {(src_el, snk_el)} out of range "
            f"[0, {n_el})."
        )
    if src_el == snk_el:
        raise ValueError("Source and sink electrodes must differ.")

    f = np.zeros(n_nodes, dtype=np.float64)
    src_node = int(elec_config.node_indices[src_el])
    snk_node = int(elec_config.node_indices[snk_el])
    f[src_node] = +current
    f[snk_node] = -current
    return f
