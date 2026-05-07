"""
eitkit.protocol.noise
=====================
Noise models for simulated EIT voltage data.

Phase 1 implements a single model: **additive white Gaussian noise**
(AWGN) calibrated to a desired signal-to-noise ratio in decibels.

The SNR is defined on the *full voltage vector* as::

    SNR_dB = 10 · log10(‖v‖² / ‖η‖²)

so the noise standard deviation is::

    σ_noise = ‖v‖ / (√N · 10^(SNR_dB / 20))

where ``N = len(v)``.

Public API
----------
add_noise(voltages, snr_db, rng) → ndarray, same shape as voltages
    Return a noisy copy of *voltages* at the requested SNR.
"""

from __future__ import annotations

import numpy as np
from numpy.random import Generator
from numpy.typing import NDArray

__all__ = ["add_noise"]


def add_noise(
    voltages: NDArray[np.floating],
    snr_db: float = 40.0,
    rng: Generator | int | None = None,
) -> NDArray[np.float64]:
    """Return a noisy copy of *voltages* corrupted by AWGN.

    The noise level is set so that the resulting SNR (measured over the
    entire vector) equals *snr_db* decibels.

    Parameters
    ----------
    voltages:
        1-D (or arbitrarily shaped) array of clean voltage values.
    snr_db:
        Desired signal-to-noise ratio in decibels.  Typical EIT hardware
        achieves 60–80 dB; a value of 40 dB is a conservative lab
        estimate.  Must be finite and positive.
    rng:
        Random-number source.  Accepts:

        * ``None``  — use a fresh :func:`numpy.random.default_rng` (not
          reproducible across runs).
        * ``int``   — seed a new :class:`numpy.random.Generator`.
        * :class:`numpy.random.Generator` — use directly (allows the
          caller to control reproducibility).

    Returns
    -------
    noisy : ndarray, shape identical to *voltages*, dtype ``float64``
        ``voltages + η`` where ``η ~ N(0, σ²·I)``.

    Raises
    ------
    ValueError
        If *snr_db* is not finite or is ≤ 0.

    Examples
    --------
    >>> import numpy as np
    >>> v = np.ones(100)
    >>> noisy = add_noise(v, snr_db=40.0, rng=0)
    >>> achieved_snr = 20 * np.log10(np.linalg.norm(v) / np.linalg.norm(noisy - v))
    >>> abs(achieved_snr - 40.0) < 1.0
    True
    """
    if not np.isfinite(snr_db) or snr_db <= 0:
        raise ValueError(
            f"snr_db must be a finite positive number, got {snr_db!r}."
        )

    v = np.asarray(voltages, dtype=np.float64)

    if isinstance(rng, int):
        generator = np.random.default_rng(rng)
    elif rng is None:
        generator = np.random.default_rng()
    else:
        generator = rng

    signal_power = np.sqrt(np.sum(v ** 2))  # ‖v‖
    n_samples = v.size

    # σ such that ‖η‖ / ‖v‖ = 10^(−SNR/20)  in expectation
    sigma = signal_power / (np.sqrt(n_samples) * 10 ** (snr_db / 20.0))

    noise = generator.normal(loc=0.0, scale=sigma, size=v.shape)
    return v + noise
