"""
eitkit.utils.phantoms
======================
Synthetic conductivity phantoms for EIT simulation and testing.

A phantom is a piecewise-constant conductivity distribution defined on
the mesh elements.  It is built from a uniform background conductivity
plus a list of *inclusions* — simple geometric shapes with a different
conductivity value.

Supported inclusion shapes
---------------------------
``circle``
    Defined by centre ``(cx, cy)``, radius ``r``, and conductivity
    ``sigma``.  An element is inside if its centroid lies within the
    disc.

``ellipse``
    Defined by centre ``(cx, cy)``, semi-axes ``(a, b)``,
    rotation angle ``theta`` (radians, default 0), and conductivity
    ``sigma``.

``rectangle``
    Axis-aligned or rotated rectangle.  Parameters: ``cx, cy``
    (centre), ``w`` (half-width along local x), ``h`` (half-height
    along local y), ``theta`` (rotation, radians, default 0), and
    ``sigma``.

``ring``
    Annular (hollow-circle) region.  Parameters: ``cx, cy``
    (centre), ``r_inner``, ``r_outer``, and ``sigma``.
    An element is inside if ``r_inner² ≤ dist² ≤ r_outer²``.

``triangle``
    Equilateral triangle.  Parameters: ``cx, cy`` (centroid),
    ``side`` (edge length), ``theta`` (rotation, radians, default 0),
    and ``sigma``.  Uses a point-in-triangle (barycentric) test.

Each inclusion is specified as a plain Python dict; the ``"shape"`` key
selects the type and the remaining keys supply the parameters.

Public API
----------
make_phantom(mesh, inclusions, sigma_background) → ndarray, shape (M,)
    Return a per-element conductivity array.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from eitkit.mesh.mesh import Mesh

__all__ = ["make_phantom"]

# Type alias for a single inclusion specification
InclusionSpec = dict[str, Any]


def make_phantom(
    mesh: Mesh,
    inclusions: list[InclusionSpec],
    sigma_background: float = 1.0,
) -> NDArray[np.float64]:
    """Build a per-element conductivity phantom.

    Parameters
    ----------
    mesh:
        Triangular mesh.
    inclusions:
        List of inclusion dicts.  Each dict **must** contain the key
        ``"shape"`` (``"circle"`` or ``"ellipse"``) and ``"sigma"``
        (conductivity of the inclusion), plus shape-specific parameters:

        Circle::

            {"shape": "circle", "cx": 0.0, "cy": 0.5,
             "r": 0.2, "sigma": 3.0}

        Ellipse::

            {"shape": "ellipse", "cx": 0.0, "cy": 0.5,
             "a": 0.3, "b": 0.15, "theta": 0.0, "sigma": 0.5}

        Inclusions are applied in list order; later ones overwrite
        earlier ones where they overlap.
    sigma_background:
        Uniform conductivity value for elements outside all inclusions.
        Must be strictly positive.  Default is 1.0 S/m.

    Returns
    -------
    sigma : ndarray, shape ``(M,)``, dtype ``float64``
        Per-element conductivity.

    Raises
    ------
    ValueError
        If ``sigma_background ≤ 0`` or an inclusion has an unknown shape
        or missing required keys.

    Examples
    --------
    >>> from eitkit.mesh import make_circle_mesh
    >>> mesh = make_circle_mesh(n_electrodes=16, h0=0.12)
    >>> sigma = make_phantom(mesh, [
    ...     {"shape": "circle", "cx": 0.0, "cy": 0.5,
    ...      "r": 0.2, "sigma": 3.0},
    ... ])
    >>> sigma.shape
    (488,)
    """
    if sigma_background <= 0:
        raise ValueError(
            f"sigma_background must be positive, got {sigma_background}."
        )

    # Element centroids, shape (M, 2)
    centroids = mesh.nodes[mesh.elements].mean(axis=1)
    cx_e = centroids[:, 0]
    cy_e = centroids[:, 1]

    sigma = np.full(mesh.n_elements, sigma_background, dtype=np.float64)

    for inc in inclusions:
        shape = inc.get("shape", "").lower()
        inc_sigma = float(inc["sigma"])
        if inc_sigma <= 0:
            raise ValueError(
                f"Inclusion sigma must be positive, got {inc_sigma}."
            )

        if shape == "circle":
            _cx = float(inc["cx"])
            _cy = float(inc["cy"])
            _r  = float(inc["r"])
            inside = (cx_e - _cx) ** 2 + (cy_e - _cy) ** 2 <= _r ** 2
            sigma[inside] = inc_sigma

        elif shape == "ellipse":
            _cx    = float(inc["cx"])
            _cy    = float(inc["cy"])
            _a     = float(inc["a"])
            _b     = float(inc["b"])
            _theta = float(inc.get("theta", 0.0))
            cos_t  = np.cos(_theta)
            sin_t  = np.sin(_theta)
            dx = cx_e - _cx
            dy = cy_e - _cy
            x_rot =  cos_t * dx + sin_t * dy
            y_rot = -sin_t * dx + cos_t * dy
            inside = (x_rot / _a) ** 2 + (y_rot / _b) ** 2 <= 1.0
            sigma[inside] = inc_sigma

        elif shape == "rectangle":
            inside = _in_rectangle(
                cx_e, cy_e,
                float(inc["cx"]), float(inc["cy"]),
                float(inc["w"]),  float(inc["h"]),
                float(inc.get("theta", 0.0)),
            )
            sigma[inside] = inc_sigma

        elif shape == "ring":
            inside = _in_ring(
                cx_e, cy_e,
                float(inc["cx"]),      float(inc["cy"]),
                float(inc["r_inner"]), float(inc["r_outer"]),
            )
            sigma[inside] = inc_sigma

        elif shape == "triangle":
            inside = _in_triangle(
                cx_e, cy_e,
                float(inc["cx"]),           float(inc["cy"]),
                float(inc["side"]),
                float(inc.get("theta", 0.0)),
            )
            sigma[inside] = inc_sigma

        else:
            raise ValueError(
                f"Unknown inclusion shape {shape!r}. "
                "Supported: 'circle', 'ellipse', 'rectangle', 'ring', "
                "'triangle'."
            )

    return sigma


# ---------------------------------------------------------------------------
# Internal shape helpers
# ---------------------------------------------------------------------------

def _in_rectangle(
    cx_e: np.ndarray,
    cy_e: np.ndarray,
    cx: float, cy: float,
    w: float, h: float,
    theta: float,
) -> np.ndarray:
    """Return boolean mask: centroid inside axis-aligned/rotated rectangle."""
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    dx = cx_e - cx
    dy = cy_e - cy
    x_rot =  cos_t * dx + sin_t * dy
    y_rot = -sin_t * dx + cos_t * dy
    return (np.abs(x_rot) <= w) & (np.abs(y_rot) <= h)


def _in_ring(
    cx_e: np.ndarray,
    cy_e: np.ndarray,
    cx: float, cy: float,
    r_inner: float, r_outer: float,
) -> np.ndarray:
    """Return boolean mask: centroid inside annular ring."""
    dist2 = (cx_e - cx) ** 2 + (cy_e - cy) ** 2
    return (dist2 >= r_inner ** 2) & (dist2 <= r_outer ** 2)


def _in_triangle(
    cx_e: np.ndarray,
    cy_e: np.ndarray,
    cx: float, cy: float,
    side: float,
    theta: float,
) -> np.ndarray:
    """Return boolean mask: centroid inside equilateral triangle (barycentric)."""
    # Equilateral triangle vertices centred at origin, before rotation
    R = side / np.sqrt(3.0)          # circumradius
    angles = np.array([np.pi / 2, np.pi / 2 + 2 * np.pi / 3,
                       np.pi / 2 + 4 * np.pi / 3]) + theta
    vx = cx + R * np.cos(angles)
    vy = cy + R * np.sin(angles)

    # Barycentric coordinate test
    def _sign(ax, ay, bx, by, px, py):
        return (ax - px) * (by - py) - (bx - px) * (ay - py)

    d1 = _sign(vx[0], vy[0], vx[1], vy[1], cx_e, cy_e)
    d2 = _sign(vx[1], vy[1], vx[2], vy[2], cx_e, cy_e)
    d3 = _sign(vx[2], vy[2], vx[0], vy[0], cx_e, cy_e)
    has_neg = (d1 < 0) | (d2 < 0) | (d3 < 0)
    has_pos = (d1 > 0) | (d2 > 0) | (d3 > 0)
    return ~(has_neg & has_pos)
