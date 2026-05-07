# Protocol

`eitkit.protocol` defines the stimulation pattern, measurement pairs, and
noise model.

## Drive pattern

`adjacent_pattern` generates the standard adjacent-electrode stimulation
scheme: electrode $i$ drives current, electrode $(i+1) \bmod L$ is the return.

```python
from eitkit.protocol import adjacent_pattern

drive_pairs = adjacent_pattern(n_electrodes=16)
# shape (16, 2): each row is [source_electrode, sink_electrode]
```

## Measurement pairs

`measurement_pairs` returns all differential voltage measurements that are
**not** on the active drive pair, giving $L(L-3)$ rows.

```python
from eitkit.protocol import measurement_pairs

meas_pairs = measurement_pairs(n_electrodes=16)
# shape (208, 3): columns are [drive_step, +electrode, -electrode]
```

!!! note
    `plot_voltages` expects `len(dV) == n_electrodes * (n_electrodes - 3)`
    exactly. Passing a wrong-length array raises `ValueError`.

## Adding noise

```python
from eitkit.protocol import add_noise

dV_noisy = add_noise(dV_clean, snr_db=40)   # 40 dB ≈ 1% noise
```

The noise is additive white Gaussian, scaled so that
$\text{SNR} = 20 \log_{10}(\|V\| / \|\epsilon\|)$.
