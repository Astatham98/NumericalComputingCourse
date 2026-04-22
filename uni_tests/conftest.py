import numpy as np


DEFAULT_X_SET = (-2.0, 1.0)
DEFAULT_Y_SET = (-1.5, 1.5)
SHIFTED_X_SET = (-0.8, -0.7)
SHIFTED_Y_SET = (0.05, 0.15)


def reference_escape_post_update(c: complex, max_iters: int) -> int:
    z = 0j
    for iteration in range(max_iters):
        z = z * z + c
        if abs(z) > 2:
            return iteration
    return max_iters


def reference_escape_bounded(c: complex, max_iters: int) -> int:
    z = 0j
    for iteration in range(max_iters):
        if z.real * z.real + z.imag * z.imag > 4.0:
            return iteration
        z = z * z + c
    return max_iters


def complex_grid(
    win_size: int,
    x_set: tuple[float, float],
    y_set: tuple[float, float],
    dtype: np.dtype = np.float64,
) -> np.ndarray:
    width = np.linspace(x_set[0], x_set[1], win_size, dtype=dtype)
    height = np.linspace(y_set[0], y_set[1], win_size, dtype=dtype)
    X, Y = np.meshgrid(width, height)
    return X + 1j * Y


def reference_grid_post_update(
    win_size: int,
    max_iters: int,
    x_set: tuple[float, float] = DEFAULT_X_SET,
    y_set: tuple[float, float] = DEFAULT_Y_SET,
    dtype: np.dtype = np.float64,
) -> np.ndarray:
    grid = complex_grid(win_size, x_set, y_set, dtype=dtype)
    result = np.zeros(grid.shape, dtype=np.int32)
    for row, values in enumerate(grid):
        for col, value in enumerate(values):
            result[row, col] = reference_escape_post_update(value, max_iters)
    return result


def reference_grid_bounded(
    win_size: int,
    max_iters: int,
    x_set: tuple[float, float] = DEFAULT_X_SET,
    y_set: tuple[float, float] = DEFAULT_Y_SET,
    dtype: np.dtype = np.float64,
) -> np.ndarray:
    grid = complex_grid(win_size, x_set, y_set, dtype=dtype)
    result = np.zeros(grid.shape, dtype=np.int32)
    for row, values in enumerate(grid):
        for col, value in enumerate(values):
            result[row, col] = reference_escape_bounded(value, max_iters)
    return result
