# Mesh

`eitkit.mesh` generates the 2-D finite-element mesh and places electrodes on
its boundary.

## Mesh generation

`make_circle_mesh` uses the DistMesh2D algorithm:

1. Seed nodes on a hexagonal grid inside the unit disk.
2. Iterate spring-relaxation forces until convergence.
3. Delaunay-triangulate the final node positions.
4. Remove exterior triangles (those whose centroid lies outside the disk).

```python
from eitkit.mesh import make_circle_mesh

mesh = make_circle_mesh(
    n_electrodes=16,   # controls boundary node density
    h0=0.07,           # target element edge length (smaller → finer mesh)
    seed=42,           # reproducibility
)
print(mesh.nodes.shape)     # (N, 2)
print(mesh.elements.shape)  # (E, 3)  — node indices per triangle
print(mesh.areas.shape)     # (E,)    — element areas
```

!!! tip
    `h0=0.07` produces ~1,400 elements — a good balance between accuracy and
    solve time. Use `h0=0.05` for publication-quality meshes (~3,000 elements).

## Electrode placement

`place_electrodes` selects the boundary node closest to each equally-spaced
angular position (counter-clockwise from the +x axis).

```python
from eitkit.mesh import place_electrodes

ec = place_electrodes(mesh, n_electrodes=16)
print(ec.node_indices)   # shape (16,) — one mesh node per electrode
print(ec.angles)         # shape (16,) — angles in radians
print(ec.arc_width)      # half-width of Neumann patch = π/L
```

## Visualisation

```python
from eitkit.utils import plot_mesh
import matplotlib.pyplot as plt

fig, ax = plot_mesh(mesh, elec_config=ec, title="Unit-disk mesh")
plt.show()
```
