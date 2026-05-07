"""
eitkit.protocol — Stimulation patterns, measurement pairs, and noise.

Public API
----------
adjacent_pattern    Build the adjacent-drive stimulation matrix (Phase 1).
measurement_pairs   Generate the corresponding voltage measurement pairs.
add_noise           Corrupt a voltage vector with additive white Gaussian noise.
"""

from eitkit.protocol.measurement import measurement_pairs
from eitkit.protocol.noise import add_noise
from eitkit.protocol.patterns import adjacent_pattern

__all__ = ["adjacent_pattern", "measurement_pairs", "add_noise"]
