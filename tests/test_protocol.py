"""
tests/test_protocol.py
======================
Unit tests for ``eitkit.protocol``:
  - adjacent_pattern
  - measurement_pairs
  - add_noise
"""

from __future__ import annotations

import numpy as np
import pytest

from eitkit.protocol import adjacent_pattern, measurement_pairs, add_noise


# ---------------------------------------------------------------------------
# adjacent_pattern
# ---------------------------------------------------------------------------


class TestAdjacentPattern:
    def test_shape_16(self):
        dp = adjacent_pattern(16)
        assert dp.shape == (16, 2)

    def test_shape_8(self):
        dp = adjacent_pattern(8)
        assert dp.shape == (8, 2)

    def test_dtype(self):
        dp = adjacent_pattern(16)
        assert dp.dtype == np.int32

    def test_first_pair(self):
        dp = adjacent_pattern(16)
        assert list(dp[0]) == [0, 1]

    def test_last_pair_wraps(self):
        # Last pair must wrap: [L-1, 0]
        n = 16
        dp = adjacent_pattern(n)
        assert list(dp[-1]) == [n - 1, 0]

    def test_all_pairs_adjacent(self):
        n = 16
        dp = adjacent_pattern(n)
        for k, (src, snk) in enumerate(dp):
            assert src == k % n
            assert snk == (k + 1) % n

    def test_no_self_injection(self):
        dp = adjacent_pattern(16)
        assert np.all(dp[:, 0] != dp[:, 1])

    def test_minimum_electrodes(self):
        dp = adjacent_pattern(4)
        assert dp.shape == (4, 2)
        # Full wrap: [0,1],[1,2],[2,3],[3,0]
        expected = np.array([[0, 1], [1, 2], [2, 3], [3, 0]], dtype=np.int32)
        np.testing.assert_array_equal(dp, expected)


# ---------------------------------------------------------------------------
# measurement_pairs
# ---------------------------------------------------------------------------


class TestMeasurementPairs:
    def test_shape_16(self):
        mp = measurement_pairs(16)
        # 16 drive steps × 13 measurements = 208 rows, 3 columns
        assert mp.shape == (208, 3)

    def test_shape_8(self):
        mp = measurement_pairs(8)
        # 8 × (8-3) = 40 rows
        assert mp.shape == (40, 3)

    def test_dtype(self):
        mp = measurement_pairs(16)
        assert mp.dtype == np.int32

    def test_column_names_semantics(self):
        # col 0 = drive_step, col 1 = plus_el, col 2 = minus_el
        mp = measurement_pairs(16)
        drive_steps = mp[:, 0]
        assert drive_steps.min() == 0
        assert drive_steps.max() == 15

    def test_each_drive_step_has_13_rows(self):
        mp = measurement_pairs(16)
        for k in range(16):
            count = np.sum(mp[:, 0] == k)
            assert count == 13, f"drive step {k} has {count} rows, expected 13"

    def test_no_measurement_on_driven_electrodes(self):
        n = 16
        mp = measurement_pairs(n)
        dp = adjacent_pattern(n)
        for row in mp:
            k, plus_el, minus_el = int(row[0]), int(row[1]), int(row[2])
            src, snk = int(dp[k, 0]), int(dp[k, 1])
            driven = {src, snk}
            assert plus_el not in driven, (
                f"step {k}: meas electrode {plus_el} is driven"
            )
            assert minus_el not in driven, (
                f"step {k}: meas electrode {minus_el} is driven"
            )

    def test_plus_minus_different(self):
        mp = measurement_pairs(16)
        assert np.all(mp[:, 1] != mp[:, 2])

    def test_electrode_indices_in_range(self):
        n = 16
        mp = measurement_pairs(n)
        assert mp[:, 1].min() >= 0
        assert mp[:, 1].max() < n
        assert mp[:, 2].min() >= 0
        assert mp[:, 2].max() < n

    def test_small_n(self):
        # n=4: 4 steps × 1 measurement each = 4 rows
        mp = measurement_pairs(4)
        assert mp.shape == (4, 3)


# ---------------------------------------------------------------------------
# add_noise
# ---------------------------------------------------------------------------


class TestAddNoise:
    def _clean_signal(self, rng=None):
        if rng is None:
            rng = np.random.default_rng(0)
        return rng.standard_normal(208) * 1e-3  # 208 measurements, ~1 mV rms

    def test_output_shape(self):
        v = self._clean_signal()
        vn = add_noise(v, snr_db=40.0, rng=np.random.default_rng(0))
        assert vn.shape == v.shape

    def test_output_dtype(self):
        v = self._clean_signal()
        vn = add_noise(v, snr_db=40.0, rng=np.random.default_rng(0))
        assert vn.dtype == np.float64

    def test_deterministic_with_rng(self):
        v = self._clean_signal()
        vn1 = add_noise(v, snr_db=40.0, rng=np.random.default_rng(7))
        vn2 = add_noise(v, snr_db=40.0, rng=np.random.default_rng(7))
        np.testing.assert_array_equal(vn1, vn2)

    def test_different_rng_different_noise(self):
        v = self._clean_signal()
        vn1 = add_noise(v, snr_db=40.0, rng=np.random.default_rng(1))
        vn2 = add_noise(v, snr_db=40.0, rng=np.random.default_rng(2))
        assert not np.allclose(vn1, vn2)

    def test_high_snr_close_to_original(self):
        # At 80 dB SNR the added noise should be negligible
        v = self._clean_signal(np.random.default_rng(42))
        vn = add_noise(v, snr_db=80.0, rng=np.random.default_rng(0))
        np.testing.assert_allclose(vn, v, rtol=1e-2)

    def test_lower_snr_adds_more_noise(self):
        v = self._clean_signal(np.random.default_rng(42))
        vn_high = add_noise(v, snr_db=60.0, rng=np.random.default_rng(0))
        vn_low  = add_noise(v, snr_db=20.0, rng=np.random.default_rng(0))
        noise_high = np.std(vn_high - v)
        noise_low  = np.std(vn_low  - v)
        assert noise_low > noise_high

    def test_zero_vector_no_noise(self):
        # If the signal is all-zero, SNR is undefined; implementation should
        # return zeros (sigma=0) rather than raising
        v = np.zeros(10)
        vn = add_noise(v, snr_db=40.0, rng=np.random.default_rng(0))
        np.testing.assert_array_equal(vn, v)
