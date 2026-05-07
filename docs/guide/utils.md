# Utilities

`eitkit.utils` provides phantom generation and visualisation helpers.

## Phantom shapes

`make_phantom` fills a background conductivity array and overlays inclusions.
Five shapes are supported:

| Shape | Required keys |
|---|---|
| `circle` | `cx`, `cy`, `r`, `sigma` |
| `ellipse` | `cx`, `cy`, `a`, `b`, `theta`, `sigma` |
| `rectangle` | `cx`, `cy`, `w`, `h`, `theta`, `sigma` |
| `ring` | `cx`, `cy`, `r_inner`, `r_outer`, `sigma` |
| `triangle` | `cx`, `cy`, `side`, `theta`, `sigma` |

```python
from eitkit.utils import make_phantom

sigma = make_phantom(mesh, inclusions=[
    {"shape": "circle",    "cx":  0.0, "cy": 0.0, "r": 0.3,  "sigma": 3.0},
    {"shape": "ellipse",   "cx":  0.2, "cy": 0.0, "a": 0.3,  "b": 0.15,
     "theta": 30.0, "sigma": 0.5},
    {"shape": "rectangle", "cx":  0.0, "cy": 0.2, "w": 0.4,  "h": 0.2,
     "theta": 0.0,  "sigma": 4.0},
    {"shape": "ring",      "cx":  0.0, "cy": 0.0, "r_inner": 0.2,
     "r_outer": 0.4, "sigma": 2.0},
    {"shape": "triangle",  "cx":  0.0, "cy": 0.0, "side": 0.4,
     "theta": 0.0,  "sigma": 3.0},
], sigma_background=1.0)
```

## Visualisation

### Mesh

```python
from eitkit.utils import plot_mesh

fig, ax = plot_mesh(mesh, elec_config=ec, title="16-electrode mesh")
```

### Conductivity map

```python
from eitkit.utils import plot_conductivity

fig, ax = plot_conductivity(mesh, sigma, cmap="RdBu_r", show_mesh=False)
# Pass ax= to embed in a subplot; vmin/vmax for explicit colour limits.
```

### Voltage sinogram

```python
from eitkit.utils import plot_voltages

fig, ax = plot_voltages(meas_pairs, dV, n_electrodes=16, highlight_drive=4)
```

`plot_voltages` expects `len(dV) == n_electrodes * (n_electrodes - 3)`.
