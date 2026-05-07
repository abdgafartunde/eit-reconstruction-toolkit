"""
eitkit.protocol.patterns
========================
Stimulation (current-injection) patterns for 2-D EIT.

Phase 1 supports **adjacent drive** only: electrode ℓ sources current
+I and electrode ℓ+1 (mod L) sinks current −I for each of the L
stimulation steps.

The returned array is the canonical representation consumed by the
forward solver:

    drive_pairs[k] = [source_electrode, sink_electrode]  (0-indexed)

so ``drive_pairs[k, 0]`` carries ``+I`` and ``drive_pairs[k, 1]`` carries
``−I``.

Public API
----------
adjacent_pattern(n_electrodes) → ndarray, shape (L, 2)
    Build the adjacent-drive electrode pairs.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

__all__ = ["adjacent_pattern"]


def adjacent_pattern(n_electrodes: int) -> NDArray[np.int32]:
    """Return adjacent-drive electrode pairs for *n_electrodes* electrodes.

    For ``L = n_electrodes`` electrodes (0-indexed ``0 … L-1``), produces
    ``L`` stimulation steps::

        step k:  source = k,  sink = (k + 1) % L

    Parameters
    ----------
    n_electrodes:
        Total number of electrodes ``L ≥ 2``.

    Returns
    -------
    drive_pairs : ndarray, shape (L, 2), dtype int32
        ``drive_pairs[k] = [source, sink]`` for stimulation step *k*.

    Raises
    ------
    ValueError
        If ``n_electrodes < 2``.

    Examples
    --------
    >>> adjacent_pattern(4)
    array([[0, 1],
           [1, 2],
           [2, 3],
           [3, 0]], dtype=int32)
    """
    if n_electrodes < 2:
        raise ValueError(
            f"n_electrodes must be ≥ 2, got {n_electrodes}."
        )

    L = n_electrodes
    sources = np.arange(L, dtype=np.int32)
    sinks = (sources + 1) % L
    return np.column_stack([sources, sinks]).astype(np.int32)
