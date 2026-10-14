# Figures

Sample results, committed so the README and the tech note can show them.
Regenerate them from the repository root with

```bash
python scripts/reconstruct_and_plot.py               # 01-05
python scripts/verify_convergence.py --outdir figures
```

**End-to-end reconstruction** (16 electrodes, adjacent drive, h0 = 0.08,
45 dB SNR, two-inclusion phantom; data simulated on the reconstruction mesh),
from `scripts/reconstruct_and_plot.py`:

- `01_mesh_and_phantom.png`: mesh with electrodes, and the true conductivity.
- `02_voltage_data.png`: clean and noisy difference voltages as electrode-pair
  matrices.
- `03_lcurve.png`: L-curve for the Tikhonov parameter.
- `04_reconstructions.png`: true conductivity, Tikhonov (lambda = 1e-7), and
  TV/ADMM (alpha = 5e-8, run to convergence). Relative errors 0.579 and 0.395.
- `05_perturbations.png`: the same results as perturbations
  delta sigma = sigma - sigma_0 on a symmetric colour scale.

**Forward-model verification**, from `scripts/verify_convergence.py`:

- `convergence_verification.png`: relative error of FEM electrode voltages
  against the closed-form point-electrode solution on the uniform unit disc,
  for h0 from 0.2 to 0.025 (roughly second order).
