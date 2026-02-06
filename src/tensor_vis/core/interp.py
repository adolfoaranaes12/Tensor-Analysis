import numpy as np
from typing import Tuple


def bilinear_interpolate(
    data: np.ndarray,
    bounds: Tuple[Tuple[float, float], Tuple[float, float]],
    point: Tuple[float, float],
) -> np.ndarray:
    """
    Bilinear interpolation on a uniform 2D grid.

    Args:
        data: Array shaped (..., nx, ny). Interpolation is applied on the last two axes.
        bounds: ((x_min, x_max), (y_min, y_max)) for the grid domain.
        point: (x, y) query point in physical coordinates.

    Returns:
        Interpolated value with shape data.shape[:-2].
    """
    if data.ndim < 2:
        raise ValueError("data must have at least 2 dimensions for bilinear interpolation")

    (x_min, x_max), (y_min, y_max) = bounds
    x, y = point

    nx, ny = data.shape[-2], data.shape[-1]
    if nx < 2 or ny < 2:
        raise ValueError("data must have at least 2 points along each axis")

    # Map physical coordinates to fractional index space
    tx = (x - x_min) / (x_max - x_min) * (nx - 1)
    ty = (y - y_min) / (y_max - y_min) * (ny - 1)

    # Clamp to valid range
    tx = np.clip(tx, 0.0, nx - 1.0)
    ty = np.clip(ty, 0.0, ny - 1.0)

    i0 = int(np.floor(tx))
    j0 = int(np.floor(ty))
    i1 = min(i0 + 1, nx - 1)
    j1 = min(j0 + 1, ny - 1)

    wx = tx - i0
    wy = ty - j0

    f00 = data[..., i0, j0]
    f10 = data[..., i1, j0]
    f01 = data[..., i0, j1]
    f11 = data[..., i1, j1]

    return (
        f00 * (1 - wx) * (1 - wy)
        + f10 * wx * (1 - wy)
        + f01 * (1 - wx) * wy
        + f11 * wx * wy
    )
