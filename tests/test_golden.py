"""
Golden-output test for the end-to-end pipeline in the demo notebook.

The configuration matches ``examples/01_reconstruction_demo.ipynb``.  The
expected values are stored in ``tests/golden/demo_notebook.json``; if a
change to eitkit is meant to alter them, regenerate the file with

    python tests/test_golden.py

and explain the change in the commit message.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from eitkit.forward import compute_jacobian, simulate
from eitkit.inverse import tikhonov_solve, tv_solve
from eitkit.mesh import make_circle_mesh, place_electrodes
from eitkit.protocol import add_noise, adjacent_pattern, measurement_pairs
from eitkit.utils import make_phantom

GOLDEN = Path(__file__).parent / "golden" / "demo_notebook.json"

CONFIG = {
    "n_electrodes": 16,
    "h0": 0.08,
    "snr_db": 45.0,
    "seed": 42,
    "lambda_tik": 1e-7,
    "alpha_tv": 5e-8,
    "tv_max_iter": 300,
}
INCLUSIONS = [
    {"shape": "circle", "cx": 0.40, "cy": 0.30, "r": 0.22, "sigma": 1.6},
    {
        "shape": "ellipse",
        "cx": -0.35,
        "cy": -0.25,
        "a": 0.25,
        "b": 0.14,
        "theta": 0.6,
        "sigma": 0.5,
    },
]

# Relative tolerances: direct solves are reproducible to rounding error;
# the ADMM iterations for TV are allowed a little more drift.
RTOL = {
    "norm_dV_clean": 1e-6,
    "norm_dV_noisy": 1e-6,
    "linearisation_error": 1e-6,
    "tikhonov_rel_error": 1e-6,
    "tikhonov_rel_residual": 1e-6,
    "tv_rel_error": 1e-4,
    "tv_rel_residual": 1e-4,
}


def run_pipeline() -> dict:
    c = CONFIG
    mesh = make_circle_mesh(n_electrodes=c["n_electrodes"], h0=c["h0"], seed=c["seed"])
    ec = place_electrodes(mesh, n_electrodes=c["n_electrodes"])
    sigma_ref = np.ones(mesh.n_elements)
    sigma_true = make_phantom(mesh, INCLUSIONS, sigma_background=1.0)
    delta_true = sigma_true - sigma_ref

    drive = adjacent_pattern(c["n_electrodes"])
    meas = measurement_pairs(c["n_electrodes"])
    dV_clean = simulate(mesh, ec, sigma_true, drive, meas, current=1.0, sigma0=1.0)
    dV_noisy = add_noise(dV_clean, snr_db=c["snr_db"], rng=c["seed"])

    J = compute_jacobian(mesh, ec, sigma_ref, drive, meas, current=1.0)
    ds_tik = tikhonov_solve(J, dV_noisy, lambda_=c["lambda_tik"])
    ds_tv = tv_solve(
        J,
        dV_noisy,
        alpha=c["alpha_tv"],
        mesh=mesh,
        rho="auto",
        max_iter=c["tv_max_iter"],
        tol=1e-4,
    )

    def rel(a, b):
        return float(np.linalg.norm(a - b) / np.linalg.norm(b))

    return {
        "n_nodes": int(mesh.n_nodes),
        "n_elements": int(mesh.n_elements),
        "n_boundary_nodes": int(mesh.n_boundary_nodes),
        "norm_dV_clean": float(np.linalg.norm(dV_clean)),
        "norm_dV_noisy": float(np.linalg.norm(dV_noisy)),
        "linearisation_error": rel(J @ delta_true, dV_clean),
        "tikhonov_rel_error": rel(ds_tik, delta_true),
        "tikhonov_rel_residual": rel(J @ ds_tik, dV_noisy),
        "tv_rel_error": rel(ds_tv, delta_true),
        "tv_rel_residual": rel(J @ ds_tv, dV_noisy),
    }


@pytest.fixture(scope="module")
def results():
    return run_pipeline()


@pytest.fixture(scope="module")
def expected():
    return json.loads(GOLDEN.read_text())


@pytest.mark.parametrize("key", ["n_nodes", "n_elements", "n_boundary_nodes"])
def test_mesh_counts(results, expected, key):
    assert results[key] == expected[key]


@pytest.mark.parametrize("key", sorted(RTOL))
def test_values(results, expected, key):
    assert results[key] == pytest.approx(expected[key], rel=RTOL[key])


if __name__ == "__main__":
    GOLDEN.parent.mkdir(exist_ok=True)
    GOLDEN.write_text(json.dumps(run_pipeline(), indent=2) + "\n")
    print(f"Wrote {GOLDEN}")
