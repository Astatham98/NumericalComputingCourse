from matplotlib.colors import LogNorm
import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt


def check_machine_differences() -> dict[type, tuple[float, float]]:
    """
    Checks machine differences between the computed machine epsilon and
    the reference machine epsilon for each numpy float type.

    Returns a dictionary with the computed and reference machine
    epsilon for each numpy float type.
    """
    s = {}
    for dtype in [np.float16, np.float32, np.float64]:
        computed = find_machine_epsilon(dtype)
        reference = np.finfo(dtype).eps
        s[dtype] = (computed, reference)
        print(f"{dtype.__name__}:")
        print(f" Computed:{float(computed):.4e}")
        print(f" Reference:{float(reference):.4e}")
        print("-------------------------------")
    return s

def find_machine_epsilon(dtype: type[np.floating] = np.float64) -> float:
    """
    Estimate machine epsilon for the given floating-point dtype.

    Args:
        dtype: NumPy floating-point type to evaluate.

    Returns:
        The estimated machine epsilon value for ``dtype``.
    """
    one = dtype(1.0)
    eps = np.finfo(dtype).eps
    while one + eps != one:
        eps /= 2.0
    return eps

def quadratic_stable(a: float, b: float, c: float) -> tuple[float, float]:
    """
    Solve a quadratic equation using a numerically stable formula.

    Args:
        a: Quadratic coefficient.
        b: Linear coefficient.
        c: Constant term.

    Returns:
        A tuple ``(x1, x2)`` containing the two roots.
    """
    t = type(a)
    disc = t(np.sqrt(b*b - t(4)*a*c))
    if b > 0:
        x1 = (-b - disc) / (t(2)*a)
    else:
        x1 = (-b + disc) / (t(2)*a)
    x2 = c / (a * x1)
    return x1, x2

def quadratic_unstable(a: float, b: float, c: float) -> tuple[float, float]:
    """
    Solve a quadratic equation using the direct quadratic formula.

    Args:
        a: Quadratic coefficient.
        b: Linear coefficient.
        c: Constant term.

    Returns:
        A tuple ``(x1, x2)`` containing the two roots.
    """
    t = type(a)
    disc = t(np.sqrt(b**2 - t(4)*a*c))
    x1 = (-b + disc) / (t(2)*a)
    x2 = (-b - disc) / (t(2)*a)
    return x1, x2

def check_quadratic_stable() -> dict[type, tuple[float, float]]:
    """
    Compare stable and unstable quadratic formulas across float dtypes.

    Returns:
        Dictionary mapping dtype to ``(stable_error, unstable_error)``.
    """
    true_small = 1.0 / 10000.0001
    s = {}
    for dtype in [np.float16, np.float32, np.float64]:
        a = dtype(1.0)
        b = dtype(-10000.0001)
        c = dtype(1.0)
        _, x2_stable = quadratic_stable(a, b, c)
        _, x2_unstable = quadratic_unstable(a, b, c)
        err_unstable = abs(float(x2_unstable) - true_small) / true_small
        err_stable = abs(float(x2_stable) - true_small) / true_small
        s[dtype] = (err_stable, err_unstable)
        print(f"{dtype.__name__}:")
        print(f" Unstable: {float(x2_unstable):.4e}, Relative Error: {err_unstable:.4e}")
        print(f" Stable: {float(x2_stable):.4e}, Relative Error: {err_stable:.4e}")
        print("-------------------------------")
    return s


def mandelbrot_divergence(
    win_size: int = 1024,
    max_iters: int = 100,
    TAU: float = 0.01,
    x_set: tuple[float, float] = (-0.7530, -0.7490),
    y_set: tuple[float, float] = (0.0990, 0.1030),
) -> npt.NDArray[np.int32]:
    """
    Plot the first iteration where float32 and float64 trajectories diverge.

    Args:
        win_size: Grid resolution per axis.
        max_iters: Maximum Mandelbrot iterations.
        TAU: Divergence threshold for complex value difference.
        x_set: Range of real-axis values.
        y_set: Range of imaginary-axis values.

    Returns:
        2D array with first divergence iteration for each pixel.
    """
    x = np.linspace(x_set[0], x_set[1], win_size)
    y = np.linspace(y_set[0], y_set[1], win_size)
    C64 = (x[np.newaxis, :] + 1j * y[:, np.newaxis]).astype(np.complex128)
    C32 = C64.astype(np.complex64)
    z32 = np.zeros_like(C32)
    z64 = np.zeros_like(C64)
    diverge = np.full((win_size, win_size), max_iters, dtype=np.int32)
    active = np.ones((win_size, win_size), dtype=bool)
    for k in range(max_iters):
        if not active.any(): break
        z32[active] = z32[active]**2 + C32[active]
        z64[active] = z64[active]**2 + C64[active]
        diff = (np.abs(z32.real.astype(np.float64) - z64.real) + np.abs(z32.imag.astype(np.float64) - z64.imag))
        newly = active & (diff > TAU)
        diverge[newly] = k
        active[newly] = False
    plt.imshow(diverge, cmap='gist_earth', origin='lower', extent=[-0.7530, -0.7490, 0.0990, 0.1030])
    plt.colorbar(label='First divergence iteration')
    plt.title(f'Mandelbrot Set Divergence (TAU={TAU})')
    plt.xlabel('Real Part')
    plt.ylabel('Imaginary Part')
    plt.show()
    return diverge

def mandelbrot_sensitivity(
    win_size: int = 1024,
    max_iters: int = 100,
    x_set: tuple[float, float] = (-0.7530, -0.7490),
    y_set: tuple[float, float] = (0.0990, 0.1030),
) -> None:
    """
    Visualize local Mandelbrot sensitivity to float32-scale perturbations.

    Args:
        win_size: Grid resolution per axis.
        max_iters: Maximum Mandelbrot iterations.
        x_set: Range of real-axis values.
        y_set: Range of imaginary-axis values.
    """
    x = np.linspace(x_set[0], x_set[1], win_size)
    y = np.linspace(y_set[0], y_set[1], win_size)
    C = (x[np.newaxis, :] + 1j * y[:, np.newaxis]).astype(np.complex128)
    eps32 = float(np.finfo(np.float32).eps)
    delta = np.maximum(eps32 * np.abs(C), 1e-10)
    def escape_count(C: npt.NDArray[np.complex128], max_iters: int) -> npt.NDArray[np.int32]:
        """Return escape-iteration counts for a complex grid."""
        z = np.zeros_like(C); cnt = np.full(C.shape, max_iters, dtype=np.int32)
        esc = np.zeros(C.shape, dtype=bool)
        for k in range(max_iters):
            z[~esc] = z[~esc]**2 + C[~esc]
            newly = ~esc & (np.abs(z) > 2.0)
            cnt[newly] = k; esc[newly] = True
        return cnt
    n_base = escape_count(C, max_iters).astype(float)
    n_perturb = escape_count(C + delta, max_iters).astype(float)
    dn = np.abs(n_base - n_perturb)
    kappa = np.where(n_base > 0, dn / (eps32 * n_base), np.nan)
    cmap_k = plt.cm.hot.copy(); cmap_k.set_bad('0.25')
    vmax = np.nanpercentile(kappa, 99)
    plt.imshow(kappa, cmap=cmap_k, origin='lower',
    extent=[-0.7530, -0.7490, 0.0990, 0.1030],
    norm=LogNorm(vmin=1, vmax=vmax))
    plt.colorbar(label=r'$\kappa$ (c)$ (log scale,$\kappa \geq 1$)')
    plt.title(f'Mandelbrot Sensitivity map')
    plt.show()

if __name__ == "__main__":
    # print(np.__version__)
    # print("Checking machine differences:")
    # check_machine_differences()
    # print("\nChecking quadratic stability:")
    # check_quadratic_stable()
    mandelbrot_sensitivity(max_iters=1000, x_set=(-0.750, -0.747), y_set=(0.099, 0.101))
