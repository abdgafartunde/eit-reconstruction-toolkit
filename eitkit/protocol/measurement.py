"""
eitkit.protocol.measurement
============================
Voltage measurement pairs for 2-D EIT.

For adjacent-drive EIT the standard convention is to measure the voltage
difference between every adjacent electrode pair **except** the two
electrodes that are currently sourcing/sinking current (and their
immediate neighbours, which carry essentially no independent signal).

Concretely, for drive step *k* (source = *k*, sink = *k+1 mod L*),
measurements are taken on pairs::

    (k+2, k+3), (k+3, k+4), …, (k+L-2, k+L-1)   (all mod L)

That gives ``L - 3`` measurements per drive step and
``L × (L - 3)`` measurements in total for a full dataset.

The returned array lists all measurement pairs across all drive steps in
a single flat structure, along with the associated drive-step index, so
the forward solver can fill the voltage array row-by-row.

Public API
----------
measurement_pairs(n_electrodes) → ndarray, shape (L*(L-3), 3), dtype int32
    Columns:  [drive_step, meas_plus, meas_minus]
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

__all__ = ["measurement_pairs"]


def measurement_pairs(n_electrodes: int) -> NDArray[np.int32]:
    """Return the voltage measurement schedule for adjacent-drive EIT.

    For ``L`` electrodes, drive step ``k`` injects current between
    electrodes ``k`` and ``(k+1) % L``.  The ``L - 3`` measurements per
    step use adjacent pairs that are at least two hops away from the
    driven electrodes.

    Parameters
    ----------
    n_electrodes:
        Total number of electrodes ``L ≥ 4``.

    Returns
    -------
    pairs : ndarray, shape ``(L*(L-3), 3)``, dtype ``int32``
        Each row is ``[drive_step, plus_electrode, minus_electrode]``.
        The measured voltage for row *i* is
        ``V[pairs[i,1]] - V[pairs[i,2]]``.

    Raises
    ------
    ValueError
        If ``n_electrodes < 4``.

    Notes
    -----
    The total number of independent measurements per frame is
    ``L × (L - 3)``.  For ``L = 16`` that is **208 measurements**.

    Examples
    --------
    >>> pairs = measurement_pairs(4)
    >>> pairs.shape
    (4, 3)
    >>> pairs
    array([[0, 2, 3],
           [1, 3, 0],
           [2, 0, 1],
           [3, 1, 2]], dtype=int32)
    """
    if n_electrodes < 4:
        raise ValueError(
            f"n_electrodes must be ≥ 4 for adjacent-drive EIT, got {n_electrodes}."
        )

    L = n_electrodes
    rows: list[tuple[int, int, int]] = []

    for k in range(L):
        # skip electrodes k (source) and (k+1)%L (sink) and their
        # immediate neighbours; measure pairs starting at k+2
        for m in range(2, L - 1):
            plus = (k + m) % L
            minus = (k + m + 1) % L
            rows.append((k, plus, minus))

    return np.array(rows, dtype=np.int32)
