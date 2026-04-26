import numpy as np
import matplotlib.pyplot as plt
import time
from functools import wraps
from numba import jit, njit, prange

# from line_profiler import profile
from typing import Any, Callable, Tuple, List, Dict
from multiprocessing import Pool
import psutil
import numpy.typing as npt
from helper_funcs.multiprocessing_helpers import (
    mandelbrot_serial,
    mandelbrot_parallel,
    mandelbrot_parallel_chunks,
    get_pool,
    compute_mandelbrot_chunk,
)
from helper_funcs.distributed_helpers import (
    mandelbrot_dask_worker,
    load_dask_client_local,
    load_dask_client,
)
from helper_funcs.gpu_helpers import gpu_mandelbrot_f64, gpu_mandelbrot

"""
Mandelbrot Set Generator
Author : Alex Statham
Course : Numerical Scientific Computing 2026
"""


def timing(func: Callable[..., Any]) -> Callable[..., Any]:
    """Decorator to measure the execution time of a function."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        print(f"Execution took: {end_time - start_time:.6f} seconds")
        return result

    return wrapper


def benchmark(
    func: Callable[..., Any], *args: Any, n_runs: int = 3
) -> Tuple[float, npt.NDArray[Any], float]:
    """Run a function multiple times and print the average execution time.
    Args:
        func: The function to benchmark.
        *args: The arguments to pass to the function.
        n_runs: The number of times to run the function.
    Returns:
        float: The median execution time in seconds.
        np.ndarray: The result of the function.
        float: The variance of the execution times.
    """
    times = []
    for _ in range(n_runs):
        start_time = time.perf_counter()
        result = func(*args)
        end_time = time.perf_counter()
        times.append(end_time - start_time)
    median = np.median(times)
    variance = np.var(times)
    print(f"Median execution time over {n_runs} runs: {median:.6f} seconds")
    return median, result, variance


def w1_f(
    points: np.ndarray[complex],
    mandelbrot_set: np.ndarray[int],
    max_iters: int = 100,
) -> np.ndarray:
    """Compute week-1 Mandelbrot escape counts for a flattened complex grid."""
    for i, c in enumerate(points):
        z = 0
        for j in range(max_iters):
            z = z**2 + c
            if abs(z) > 2:
                mandelbrot_set[i] = j
                break
        else:
            mandelbrot_set[i] = max_iters
    return mandelbrot_set


def w2_f(
    C: np.ndarray[complex], Z: np.ndarray, M: np.ndarray, max_iters: int
) -> np.ndarray:
    """Compute week-2 vectorized Mandelbrot escape counts for a complex grid."""
    for _ in range(max_iters):
        mask = np.abs(Z) <= 2
        Z[mask] = Z[mask] ** 2 + C[mask]
        M[mask] += 1
    return M


@timing
def w1_main(
    max_iters: int = 100,
    x_set: tuple = (-2.0, 1.0),  # Changed to tuple + floats
    y_set: tuple = (-1.5, 1.5),  # Changed to tuple + floats
    win_size: int = 100,
) -> np.ndarray:
    """Generate and plot the Mandelbrot set.
    Args:
        max_iter (int): Maximum number of iterations.
        x_set (tuple): X-axis range.
        y_set (tuple): Y-axis range.
        win_size (int): Number of points in each axis.
    Returns:
        np.ndarray: Mandelbrot set values in a 2D array.
    """
    width = np.linspace(x_set[0], x_set[1], win_size)
    height = np.linspace(y_set[0], y_set[1], win_size)
    points = []
    for w in width:
        for h in height:
            points.append(w + h * 1j)
    points = np.array(points)

    # Based on the 100x100 points, compute the mandelbrot set
    mandelbrot_set = np.zeros(points.shape, dtype=int)
    # Get n and c
    mandelbrot_set = w1_f(points, mandelbrot_set, max_iters)
    # Reshape to 2D and plot
    mandelbrot_set = np.reshape(mandelbrot_set, (len(width), len(height))).T
    return mandelbrot_set


def w2_main(
    max_iters: int = 100,
    x_set: tuple = (-2.0, 1.0),  # Changed to tuple + floats
    y_set: tuple = (-1.5, 1.5),  # Changed to tuple + floats
    win_size: int = 100,
) -> np.ndarray:
    """Generate and plot the Mandelbrot set.
    Args:
        max_iter (int): Maximum number of iterations.
        x_set (tuple): X-axis range.
        y_set (tuple): Y-axis range.
        win_size (int): Number of points in each axis.
    Returns:
        np.ndarray: Mandelbrot set values in a 2D array.
    """

    width = np.linspace(x_set[0], x_set[1], win_size, dtype=np.float32)
    height = np.linspace(y_set[0], y_set[1], win_size, dtype=np.float32)
    X, Y = np.meshgrid(width, height)
    C = X + 1j * Y

    # Based on the 100x100 points, compute the mandelbrot set
    mandelbrot_set = np.zeros(C.shape, dtype=C.dtype)

    # Initialize Z and M arrays
    Z = np.zeros_like(C, dtype=np.complex64)
    M = np.zeros_like(C, dtype=np.int32)
    # Compute the Mandelbrot set using the function f
    mandelbrot_set = timing(w2_f)(C, Z, M, max_iters)
    return mandelbrot_set


@jit(nopython=True)
def f_jit(c: complex, max_iters: int) -> int:
    """
    Compute Mandelbrot escape iteration count for a single complex point.

    Args:
        c (complex): Complex point to evaluate.
        max_iters (int): Maximum number of iterations.

    Returns:
        int: Iteration where escape occurs, or ``max_iters`` if bounded.
    """
    z = 0
    for j in range(max_iters):
        z = z**2 + c
        # Break if |Z| > 2
        if abs(z) > 2:
            return j
    return max_iters


@timing
def w_1_5_main(
    max_iters: int = 100,
    x_set: tuple = (-2.0, 1.0),  # Changed to tuple + floats
    y_set: tuple = (-1.5, 1.5),  # Changed to tuple + floats
    win_size: int = 100,
) -> np.ndarray:
    """Generate and plot the Mandelbrot set.
    Args:
        max_iter (int): Maximum number of iterations.
        x_set (tuple): X-axis range.
        y_set (tuple): Y-axis range.
        win_size (int): Number of points in each axis.
    Returns:
        np.ndarray: Mandelbrot set values in a 2D array.
    """

    width = np.linspace(x_set[0], x_set[1], win_size)
    height = np.linspace(y_set[0], y_set[1], win_size)
    points = [complex(x, y) for x in width for y in height]

    # Based on the 100x100 points, compute the mandelbrot set
    mandelbrot_set = np.zeros(len(points))
    # Get n and c
    for i, c in enumerate(points):
        mandelbrot_set[i] = f_jit(c, max_iters)
    # Reshape to 2D and plot
    mandelbrot_set = np.reshape(mandelbrot_set, (len(width), len(height))).T
    return mandelbrot_set


def w2_memory_access(N: int = 1000) -> None:
    """
    Compare row-major and column-major memory access timing.

    Args:
        N (int): Side length of the generated square matrix.
    """
    np.random.seed(42)
    A = np.random.rand(N, N)

    @timing
    def row_major_sum(A):
        for i in range(N):
            s = np.sum(A[i, :])

    @timing
    def column_major_sum(A):
        for j in range(N):
            s = np.sum(A[:, j])

    print("Using C order:")
    row_major_sum(A)
    column_major_sum(A)

    print("\nUsing Fortran order:")
    A_F = np.asfortranarray(A)
    row_major_sum(A_F)
    column_major_sum(A_F)


def w2_scaling() -> tuple[list[int], list[float]]:
    """
    Measure and plot scaling behavior of ``w2_main`` across grid sizes.

    Returns:
        tuple[list[int], list[float]]: Tested sizes and measured times.
    """
    sizes = [1024, 2048, 4096, 8192]
    times = []
    for size in sizes:
        time, _, _ = benchmark(w2_main, 100, (-2.0, 1.0), (-1.5, 1.5), size, n_runs=1)
        times.append(time)

    plt.figure(figsize=(10, 6))
    plt.plot(sizes, times, marker="o")
    plt.title("Execution Time vs Window Size for w2_main")
    plt.xlabel("Window Size (win_size)")
    plt.ylabel("Execution Time (seconds)")
    plt.show()
    return sizes, times


@timing
@njit
def w3_f(
    C: np.ndarray[complex], mandelbrot_set: np.ndarray, max_iters: int
) -> np.ndarray:
    """_summary_

    Args:
        C (np.ndarray[complex]): _description_
        mandelbrot_set (np.ndarray): _description_
        max_iters (int): _description_

    Returns:
        np.ndarray: _description_
    """
    for i, col in enumerate(C):
        for j, c in enumerate(col):
            z = 0j
            for k in range(max_iters):
                z = z**2 + c
                # Break if |Z| > 2
                if z.real * z.real + z.imag * z.imag > 4.0:
                    mandelbrot_set[i, j] = k
                    break
            else:
                mandelbrot_set[i, j] = max_iters
    return mandelbrot_set


@timing
@njit(parallel=True)
def w3_parallel(
    C: np.ndarray[complex], mandelbrot_set: np.ndarray, winsize: int, max_iters: int
) -> np.ndarray:
    """
    Compute Mandelbrot escape counts using Numba parallel loops.

    Args:
        C (np.ndarray[complex]): Complex grid values.
        mandelbrot_set (np.ndarray): Output array for escape counts.
        winsize (int): Grid size along each axis.
        max_iters (int): Maximum number of iterations.

    Returns:
        np.ndarray: Filled Mandelbrot escape-count array.
    """
    for i in prange(winsize):
        for j in prange(winsize):
            z = 0j
            n = 0
            while n < max_iters and z.real * z.real + z.imag * z.imag <= 4.0:
                z = z * z + C[i, j]
                n += 1
            mandelbrot_set[i, j] = n
    return mandelbrot_set


def w3_main(
    max_iters: int = 100,
    x_set: Tuple[float, float] = (-2.0, 1.0),  # Changed to tuple + floats
    y_set: Tuple[float, float] = (-1.5, 1.5),  # Changed to tuple + floats
    win_size: int = 100,
    dtype: npt.DTypeLike = np.float64,
) -> np.ndarray:
    """Generate and plot the Mandelbrot set.
    Args:
        max_iter (int): Maximum number of iterations.
        x_set (tuple): X-axis range.
        y_set (tuple): Y-axis range.
        win_size (int): Number of points in each axis.
        dtype (np.dtype): Data type for the Mandelbrot set.
    Returns:
        np.ndarray: Mandelbrot set values in a 2D array.
    """
    width = np.linspace(x_set[0], x_set[1], win_size, dtype=dtype)
    height = np.linspace(y_set[0], y_set[1], win_size, dtype=dtype)
    X, Y = np.meshgrid(width, height)
    C = X + 1j * Y

    # Based on the 100x100 points, compute the mandelbrot set
    mandelbrot_set = np.zeros(C.shape, dtype=np.int32)
    # Compute the Mandelbrot set using the function f
    mandelbrot_set = w3_f(C, mandelbrot_set, max_iters)
    return mandelbrot_set


def w3_parallel_main(
    max_iters: int = 100,
    x_set: tuple = (-2.0, 1.0),  # Changed to tuple + floats
    y_set: tuple = (-1.5, 1.5),  # Changed to tuple + floats
    win_size: int = 100,
    dtype: npt.DTypeLike = np.float64,
) -> np.ndarray:
    """Generate and plot the Mandelbrot set.
    Args:
        max_iter (int): Maximum number of iterations.
        x_set (tuple): X-axis range.
        y_set (tuple): Y-axis range.
        win_size (int): Number of points in each axis.
    Returns:
        np.ndarray: Mandelbrot set values in a 2D array.
    """
    width = np.linspace(x_set[0], x_set[1], win_size, dtype=dtype)
    height = np.linspace(y_set[0], y_set[1], win_size, dtype=dtype)
    X, Y = np.meshgrid(width, height)
    C = X + 1j * Y

    # Based on the 100x100 points, compute the mandelbrot set
    mandelbrot_set = np.zeros(C.shape, dtype=np.int32)
    # Compute the Mandelbrot set using the function f
    mandelbrot_set = w3_parallel(C, mandelbrot_set, win_size, max_iters)
    return mandelbrot_set


@njit
def mandelbrot_calc(c: complex, max_iters: int = 100) -> int:
    """
    Compute Mandelbrot escape count for one complex point.

    Args:
        c (complex): Complex point to evaluate.
        max_iters (int): Maximum number of iterations.

    Returns:
        int: Escape iteration count.
    """
    z = 0j
    for n in range(max_iters):
        if z.real * z.real + z.imag * z.imag > 4.0:
            return n
        z = z * z + c
    return max_iters


@timing
def numba_hybrid(
    max_iters: int = 100,
    x_set: tuple = (-2.0, 1.0),  # Changed to tuple + floats
    y_set: tuple = (-1.5, 1.5),  # Changed to tuple + floats
    win_size: int = 100,
    dtype: npt.DTypeLike = np.float64,
) -> np.ndarray:
    """Generate and plot the Mandelbrot set.
    Args:
        max_iter (int): Maximum number of iterations.
        x_set (tuple): X-axis range.
        y_set (tuple): Y-axis range.
        win_size (int): Number of points in each axis.
    Returns:
        np.ndarray: Mandelbrot set values in a 2D array.
    """

    width = np.linspace(x_set[0], x_set[1], win_size, dtype=dtype)
    height = np.linspace(y_set[0], y_set[1], win_size, dtype=dtype)
    X, Y = np.meshgrid(width, height)
    C = X + 1j * Y

    # Based on the 100x100 points, compute the mandelbrot set
    mandelbrot_set = np.zeros(C.shape, dtype=np.int32)

    for i in range(len(height)):
        for j in range(len(width)):
            c = C[i, j]
            mandelbrot_set[i, j] = mandelbrot_calc(c, max_iters)
    return mandelbrot_set


def w4_testing(
    max_iters: int = 100,
    x_set: Tuple[float, float] = (-2.0, 1.0),  # Changed to tuple + floats
    y_set: Tuple[float, float] = (-1.5, 1.5),  # Changed to tuple + floats
    win_size: int = 100,
    dtype: npt.DTypeLike = np.float64,
) -> np.ndarray:
    """Generate and plot the Mandelbrot set.
    Args:
        max_iter (int): Maximum number of iterations.
        x_set (tuple): X-axis range.
        y_set (tuple): Y-axis range.
        win_size (int): Number of points in each axis.
        dtype (np.dtype): Data type for the Mandelbrot set.
    Returns:
        np.ndarray: Mandelbrot set values in a 2D array.
    """
    print("Benchmarking serial approach")
    median_serial, _, var_serial = benchmark(
        mandelbrot_serial,
        win_size,
        x_set[0],
        x_set[1],
        y_set[0],
        y_set[1],
        max_iters,
        n_runs=4,
    )

    medians = []
    print("Benchmarking parallel approach")
    for n_cores in range(1, psutil.cpu_count() + 1):
        print(f"Running on {n_cores} cores")
        mandelbrot_set, timings = mandelbrot_parallel(
            win_size,
            x_set[0],
            x_set[1],
            y_set[0],
            y_set[1],
            max_iters,
            n_cores,
            NoRuns=3,
        )
        # [print(f"Execution took: {t:.6f} seconds \n") for t in timings]
        print(
            f"Median execution time over 3 runs for {n_cores}: {np.median(timings):.6f} seconds"
        )
        medians.append(np.median(timings))
    return mandelbrot_set, median_serial, var_serial, medians


def w4_main(
    max_iters: int = 100,
    x_set: Tuple[float, float] = (-2.0, 1.0),  # Changed to tuple + floats
    y_set: Tuple[float, float] = (-1.5, 1.5),  # Changed to tuple + floats
    win_size: int = 100,
    dtype: npt.DTypeLike = np.float64,
    NoCores: int = psutil.cpu_count(),
    n_runs: int = 3,
) -> np.ndarray:
    """Generate and plot the Mandelbrot set.
    Args:
        max_iter (int): Maximum number of iterations.
        x_set (tuple): X-axis range.
        y_set (tuple): Y-axis range.
        win_size (int): Number of points in each axis.
        dtype (np.dtype): Data type for the Mandelbrot set.
        NoCores (int): Number of cores to use.
        n_runs (int): Number of times to run the computation.
    Returns:
        np.ndarray: Mandelbrot set values in a 2D array.
    """

    mandelbrot_set, timings = mandelbrot_parallel(
        win_size, x_set[0], x_set[1], y_set[0], y_set[1], max_iters, NoCores, n_runs
    )
    return np.median(timings), mandelbrot_set, np.var(timings)


def w5_chunk_testing(
    max_iters: int = 100,
    x_set: Tuple[float, float] = (-2.0, 1.0),  # Changed to tuple + floats
    y_set: Tuple[float, float] = (-1.5, 1.5),  # Changed to tuple + floats
    win_size: int = 100,
    dtype: npt.DTypeLike = np.float64,
    NoProcesses: int = psutil.cpu_count(),
    chunk_range: range = range(1, 17),
    n_runs: int = 3,
) -> Tuple[Dict[int, float], Dict[int, float], np.ndarray]:
    """Generate and plot the Mandelbrot set.

    Args:
        max_iter (int): Maximum number of iterations.
        x_set (tuple): X-axis range.
        y_set (tuple): Y-axis range.
        win_size (int): Number of points in each axis.
        dtype (np.dtype): Data type for the Mandelbrot set.
        NoProcesses (int): Number of cores to use.
        chunk_range (range): Range of chunk sizes to test.
        n_runs (int): Number of times to run the computation.

    Returns:
        A dict containing the median timings for each chunk size.
        A dict containing the median lifs for each chunk size.
        A 2D array containing the mandelbrot set values.

    """
    tiny = [(0, 8, 8, x_set[0], x_set[1], y_set[0], y_set[1], max_iters)]
    pool = get_pool(8, tiny)

    processers = NoProcesses
    timing_s, _, _ = benchmark(
        mandelbrot_serial,
        win_size,
        x_set[0],
        x_set[1],
        y_set[0],
        y_set[1],
        max_iters,
        n_runs=4,
    )

    outer_timings = {}
    outer_lifs = {}
    for chunk in chunk_range:
        chunk = processers * chunk
        inner_timings = []
        inner_lifs = []
        for n in range(n_runs):
            mandelbrot_set, timing_p = mandelbrot_parallel_chunks(
                win_size,
                x_set[0],
                x_set[1],
                y_set[0],
                y_set[1],
                max_iters,
                processers,
                n_chunks=chunk,
                pool=pool,
            )
            # Calculate the lif for this chunk size
            lif = ((processers * timing_p) / timing_s) - 1
            inner_timings.append(timing_p)
            inner_lifs.append(lif)
        print(
            f"Execution time for {processers} processes and {chunk} chunks: {np.median(inner_timings):.6f} seconds and {np.median(inner_lifs):.6f} LIF"
        )
        outer_timings[chunk] = np.median(inner_timings)
        outer_lifs[chunk] = np.median(inner_lifs)
    return outer_timings, outer_lifs, mandelbrot_set


def w5_main(
    max_iters: int = 100,
    x_set: Tuple[float, float] = (-2.0, 1.0),  # Changed to tuple + floats
    y_set: Tuple[float, float] = (-1.5, 1.5),  # Changed to tuple + floats
    win_size: int = 100,
    dtype: npt.DTypeLike = np.float64,
    NoProcesses: int = psutil.cpu_count(),
    n_runs: int = 3,
    chunk: int = 8,
) -> np.ndarray:
    """Generate and plot the Mandelbrot set.
    Args:
        max_iter (int): Maximum number of iterations.
        x_set (tuple): X-axis range.
        y_set (tuple): Y-axis range.
        win_size (int): Number of points in each axis.
        dtype (npt.DTypeLike): Data type for the Mandelbrot set.

    Returns:
        np.ndarray: Mandelbrot set values in a 2D array.
    """
    tiny = [(0, 8, 8, x_set[0], x_set[1], y_set[0], y_set[1], max_iters)]
    pool = get_pool(8, tiny)

    processers = NoProcesses
    timing_s, _, _ = benchmark(
        mandelbrot_serial,
        win_size,
        x_set[0],
        x_set[1],
        y_set[0],
        y_set[1],
        max_iters,
        n_runs=4,
    )

    chunk = processers * chunk
    timings = []
    for n in range(n_runs):
        mandelbrot_set, timing_p = mandelbrot_parallel_chunks(
            win_size,
            x_set[0],
            x_set[1],
            y_set[0],
            y_set[1],
            max_iters,
            processers,
            n_chunks=chunk,
            pool=pool,
        )
        timings.append(timing_p)

    lif = ((processers * np.median(timings)) / timing_s) - 1
    return np.median(timings), mandelbrot_set, np.var(timings), lif


def w5_testing(process_max: int = 4, n_runs: int = 3) -> None:
    """
    Benchmark the parallelized Mandelbrot set computation using multiple processes.

    Args:
        process_max (int): Maximum number of processes to use * 4.
        n_runs (int): Number of times to run the computation.

    Returns:
        None
    """
    for n in range(1, process_max + 1):
        median, _, var, _ = w5_main(NoProcesses=4 * n, n_runs=n_runs)
        print(
            f"Median execution time over {n_runs} runs for {4 * n}: {median:.6f} seconds"
        )


def w6_testing(
    max_iters: int = 100,
    x_set: Tuple[float, float] = (-2.0, 1.0),  # Changed to tuple + floats
    y_set: Tuple[float, float] = (-1.5, 1.5),  # Changed to tuple + floats
    win_size: int = 100,
    dtype: npt.DTypeLike = np.float64,
    NoProcesses: int = 8,
    n_runs: int = 3,
    testing_chunks: List[int] = [64, 128, 256, 512],
) -> np.ndarray:
    """Generate and plot the Mandelbrot set.
    Args:
        max_iter (int): Maximum number of iterations.
        x_set (tuple): X-axis range.
        y_set (tuple): Y-axis range.
        win_size (int): Number of points in each axis.
        dtype (np.dtype): Data type for the Mandelbrot set.
    Returns:
        np.ndarray: Mandelbrot set values in a 2D array.
    """

    # Warmup
    client, cluster = load_dask_client_local(NoProcesses, 1)
    client.run(
        lambda: compute_mandelbrot_chunk(
            0, 8, 8, x_set[0], x_set[1], y_set[0], y_set[1], 10
        )
    )
    # Get serial time
    t_serial = get_mandelbrot_serial_time(win_size, x_set, y_set, max_iters)

    chunk_times = {}
    for chunk in testing_chunks:
        times = []
        for _ in range(n_runs):
            timing, results = mandelbrot_dask_worker(
                win_size,
                x_set[0],
                x_set[1],
                y_set[0],
                y_set[1],
                max_iters,
                chunks=chunk,
            )
            times.append(timing)
        print(f"median time for dask worker: {np.median(times)}")
        LIF = NoProcesses * np.median(times) / t_serial - 1
        chunk_times[chunk] = (np.median(times), LIF)
    client.close()
    cluster.close()

    return chunk_times, t_serial


def get_mandelbrot_serial_time(
    win_size: int,
    x_set: tuple[float, float],
    y_set: tuple[float, float],
    max_iters: int,
) -> float:
    """
    Measure serial Mandelbrot execution time with a short warmup.

    Args:
        win_size (int): Grid size along each axis.
        x_set (tuple[float, float]): Real-axis range.
        y_set (tuple[float, float]): Imaginary-axis range.
        max_iters (int): Maximum number of iterations.

    Returns:
        float: Serial execution time in seconds.
    """
    mandelbrot_serial(10, x_set[0], x_set[1], y_set[0], y_set[1], max_iters)

    t0 = time.perf_counter()
    mandelbrot_serial(win_size, x_set[0], x_set[1], y_set[0], y_set[1], max_iters)
    t1 = time.perf_counter()
    print(f"Execution for serial took: {t1 - t0:.6f} seconds")

    return t1 - t0


def w6_main(
    max_iters: int = 100,
    x_set: Tuple[float, float] = (-2.0, 1.0),  # Changed to tuple + floats
    y_set: Tuple[float, float] = (-1.5, 1.5),  # Changed to tuple + floats
    win_size: int = 100,
    dtype: npt.DTypeLike = np.float64,
    NoProcesses: int = 8,
    n_runs: int = 3,
    chunks: int = 64,
) -> np.ndarray:
    """Generate and plot the Mandelbrot set.
    Args:
        max_iter (int): Maximum number of iterations.
        x_set (tuple): X-axis range.
        y_set (tuple): Y-axis range.
        win_size (int): Number of points in each axis.
        dtype (np.dtype): Data type for the Mandelbrot set.
    Returns:
        np.ndarray: Mandelbrot set values in a 2D array.
    """

    client, cluster = load_dask_client_local(NoProcesses, 1)
    client.run(
        lambda: compute_mandelbrot_chunk(
            0, 8, 8, x_set[0], x_set[1], y_set[0], y_set[1], 10
        )
    )

    times = []
    for _ in range(n_runs):
        timing, results = mandelbrot_dask_worker(
            win_size, x_set[0], x_set[1], y_set[0], y_set[1], max_iters, chunks=chunks
        )
        times.append(timing)
    print(f"median time for dask worker: {np.median(times)}")

    client.close()
    cluster.close()

    t_serial = get_mandelbrot_serial_time(win_size, x_set, y_set, max_iters)

    LIF = ((NoProcesses * np.median(times)) / t_serial) - 1

    return np.median(times), results, np.var(times), LIF


def w7_testing(
    max_iters: int = 100,
    x_set: Tuple[float, float] = (-2.0, 1.0),  # Changed to tuple + floats
    y_set: Tuple[float, float] = (-1.5, 1.5),  # Changed to tuple + floats
    win_size: int = 100,
    dtype: npt.DTypeLike = np.float64,
    NoProcesses: int = 8,
    n_runs: int = 3,
    testing_chunks: List[int] = [64, 128, 256, 512],
    ip: str = "10.92.0.0",
) -> np.ndarray:
    """Generate and plot the Mandelbrot set.
    Args:
        max_iter (int): Maximum number of iterations.
        x_set (tuple): X-axis range.
        y_set (tuple): Y-axis range.
        win_size (int): Number of points in each axis.
        dtype (np.dtype): Data type for the Mandelbrot set.
    Returns:
        np.ndarray: Mandelbrot set values in a 2D array.
    """

    # Warmup
    client = load_dask_client(ip)
    client.run(
        lambda: compute_mandelbrot_chunk(
            0, 8, 8, x_set[0], x_set[1], y_set[0], y_set[1], 10
        )
    )
    # Get serial time
    # t_serial = get_mandelbrot_serial_time(win_size, x_set, y_set, max_iters)

    chunk_times = {}
    for chunk in testing_chunks:
        times = []
        for _ in range(n_runs):
            timing, _ = mandelbrot_dask_worker(
                win_size,
                x_set[0],
                x_set[1],
                y_set[0],
                y_set[1],
                max_iters,
                chunks=chunk,
            )
            times.append(timing)
        print(f"median time for dask worker: {np.median(times)}")
        # LIF = NoProcesses * np.median(times) / t_serial - 1
        chunk_times[chunk] = np.median(times)
    client.close()
    return chunk_times


def w7_main(
    max_iters: int = 100,
    x_set: Tuple[float, float] = (-2.0, 1.0),  # Changed to tuple + floats
    y_set: Tuple[float, float] = (-1.5, 1.5),  # Changed to tuple + floats
    win_size: int = 100,
    dtype: npt.DTypeLike = np.float64,
    NoProcesses: int = 8,
    n_runs: int = 3,
    chunks: int = 64,
    ip: str = "10.92.0.0",
) -> np.ndarray:
    """Generate and plot the Mandelbrot set.
    Args:
        max_iter (int): Maximum number of iterations.
        x_set (tuple): X-axis range.
        y_set (tuple): Y-axis range.
        win_size (int): Number of points in each axis.
        dtype (np.dtype): Data type for the Mandelbrot set.
        NoProcesses (int): Number of processes to use.
        n_runs (int): Number of runs to perform to obtain the median.
        chunks (int): Number of chunks to split the computation into.
        ip (str): IP address of the Dask scheduler.
    Returns:
        np.ndarray: Mandelbrot set values in a 2D array.
    """

    client = load_dask_client(ip)
    client.run(
        lambda: compute_mandelbrot_chunk(
            0, 8, 8, x_set[0], x_set[1], y_set[0], y_set[1], 10
        )
    )

    times = []
    for _ in range(n_runs):
        timing, results = mandelbrot_dask_worker(
            win_size, x_set[0], x_set[1], y_set[0], y_set[1], max_iters, chunks=chunks
        )
        times.append(timing)
    print(f"median time for dask worker: {np.median(times)}")

    client.close()

    # t_serial = get_mandelbrot_serial_time(win_size, x_set, y_set, max_iters)

    # TODO change NoProcesses
    # LIF = NoProcesses * np.median(times) / t_serial - 1

    return np.median(times), results, np.var(times)


def benchmark_all(n_runs: int = 3, size: int = 1024) -> dict[str, float]:
    """
    Run benchmark suite across week 1-6 Mandelbrot implementations.

    Args:
        n_runs (int): Number of runs for median/variance timing.
        size (int): Grid size along each axis.

    Returns:
        dict: Summary dictionary of median, variance, and LIF metrics.
    """
    print("Week 1: Naive python implementation")
    median_w1, mandelbrot_set_w1, var_w1 = benchmark(
        w1_main, 100, (-2.0, 1.0), (-1.5, 1.5), size, n_runs=n_runs
    )
    print("Week 2: numpy vectorization")
    median_w2, mandelbrot_set_w2, var_w2 = benchmark(
        w2_main, 100, (-2.0, 1.0), (-1.5, 1.5), size, n_runs=n_runs
    )
    print("Week 3: Naive numba")
    median_w1_5, mandelbrot_set_w1_5, var_w1_5 = benchmark(
        w_1_5_main, 100, (-2.0, 1.0), (-1.5, 1.5), size, n_runs=n_runs
    )
    print("Week 3: optimized numba")
    median_w3_f64, _, median_w3_f32, _, var_w3_f64, var_w3_f32 = benchmark_dtype(n_runs=n_runs)

    print("Week 4: parallel computing")
    # W4 main now uses its own timings and we do not want to time the workers in the benchmark
    median_w4, mandelbrot_set_w4, var_w4 = w4_main(
        100, (-2.0, 1.0), (-1.5, 1.5), size, n_runs=n_runs
    )
    print(f"w4 median_w4 {median_w4}, \nw4 variance {var_w4}")
    print("Week 5: parallel computing with pools")
    median_w5, mandelbrot_set_w5, var_w5, w5_LIF = w5_main(
        100, (-2.0, 1.0), (-1.5, 1.5), size, chunk=6, n_runs=n_runs
    )
    print(f"Median time for w5: {median_w5}, \nw5 variance {var_w5}")
    print("Week 6: Dask local")
    median_w6, mandelbrot_set_26, var_w6, w6_LIF = w6_main(
        100, (-2.0, 1.0), (-1.5, 1.5), size, n_runs=n_runs, chunks=48
    )
    print(f"Median time for w6: {median_w6}, \nw6 variance {var_w6}")

    print("Week 10: GPU with OPENcl (f32)")
    median_w10, mandelbrot_set_w10, var_w10 = w10_main(
        100, (-2.0, 1.0), (-1.5, 1.5), size, n_runs=n_runs
    )
    print(f"Median time for w10: {median_w10}, \nw10 variance {var_w10}")
    print("Week 10: GPU with OPENcl (f64)")
    median_w10_f64, mandelbrot_set_w10_f64, var_w10_f64 = w10_main(
        100, (-2.0, 1.0), (-1.5, 1.5), size, n_runs=n_runs, use_f64=True
    )
    print(
        f"Median time for w10 (f64): {median_w10_f64}, \nw10 variance (f64) {var_w10_f64}"
    )

    benhmark_dict = {
        "w1_median": median_w1,
        "w2_median": median_w2,
        "w1_5_median": median_w1_5,
        "w3_f32_median": median_w3_f32,
        "w3_f64_median": median_w3_f64,
        "w4_median": median_w4,
        "w5_median": median_w5,
        "w6_median": median_w6,
        "w10_median": median_w10,
        "w10_f64_median": median_w10_f64,
        "w1_variance": var_w1,
        "w2_variance": var_w2,
        "w1_5_variance": var_w1_5,
        "w3_f32_variance": var_w3_f32,
        "w3_f64_variance": var_w3_f64,
        "w4_variance": var_w4,
        "w5_variance": var_w5,
        "w6_variance": var_w6,
        "w5_LIF": w5_LIF,
        "w6_LIF": w6_LIF,
        "w10_variance": var_w10,
        "w10_f64_variance": var_w10_f64,
    }
    return benhmark_dict


def w10_main(
    max_iters: int = 100,
    x_set: Tuple[float, float] = (-2.0, 1.0),  # Changed to tuple + floats
    y_set: Tuple[float, float] = (-1.5, 1.5),  # Changed to tuple + floats
    win_size: int = 1024,
    n_runs: int = 3,
    use_f64: bool = False,
) -> np.ndarray:
    medians = []
    for _ in range(n_runs):
        if use_f64:
            m, t = gpu_mandelbrot_f64(
                win_size=win_size, x_set=x_set, y_set=y_set, max_iters=max_iters
            )
        else:
            m, t = gpu_mandelbrot(
                win_size=win_size, x_set=x_set, y_set=y_set, max_iters=max_iters
            )
        medians.append(t)
    return np.median(medians), m, np.var(medians)


def bennchmark_parallel(n_runs: int = 3, size: int = 4096) -> dict[str, float]:
    """
    Benchmark only the parallel/distributed implementations (week 4-6).

    Args:
        n_runs (int): Number of runs for median/variance timing.
        size (int): Grid size along each axis.

    Returns:
        dict: Summary dictionary of median, variance, and LIF metrics.
    """
    print("Week 4: parallel computing")
    # W4 main now uses its own timings and we do not want to time the workers in the benchmark
    median_w4, mandelbrot_set_w4, var_w4 = w4_main(
        100, (-2.0, 1.0), (-1.5, 1.5), size, n_runs=n_runs
    )
    print(f"w4 median_w4 {median_w4}, \nw4 variance {var_w4}")
    print("Week 5: parallel computing with pools")
    median_w5, mandelbrot_set_w5, var_w5, w5_LIF = w5_main(
        100, (-2.0, 1.0), (-1.5, 1.5), size, n_runs=n_runs
    )
    print(f"Median time for w5: {median_w5}, \nw5 variance {var_w5}")
    print("Week 6: Dask local")
    median_w6, mandelbrot_set_26, var_w6, w6_LIF = w6_main(
        100, (-2.0, 1.0), (-1.5, 1.5), size, n_runs=n_runs, chunks=48
    )
    print(f"Median time for w6: {median_w6}, \nw6 variance {var_w6}")
    print("Week 10: GPU with OPENcl (f32)")
    median_w10, mandelbrot_set_w10, var_w10 = w10_main(
        100, (-2.0, 1.0), (-1.5, 1.5), size, n_runs=n_runs
    )
    print(f"Median time for w10: {median_w10}, \nw10 variance {var_w10}")
    print("Week 10: GPU with OPENcl (f64)")
    median_w10_f64, mandelbrot_set_w10_f64, var_w10_f64 = w10_main(
        100, (-2.0, 1.0), (-1.5, 1.5), size, n_runs=n_runs, use_f64=True
    )
    print(
        f"Median time for w10 (f64): {median_w10_f64}, \nw10 variance (f64) {var_w10_f64}"
    )

    benhmark_dict = {
        "w4_median": median_w4,
        "w5_median": median_w5,
        "w6_median": median_w6,
        "w4_variance": var_w4,
        "w5_variance": var_w5,
        "w6_variance": var_w6,
        "w5_LIF": w5_LIF,
        "w6_LIF": w6_LIF,
        "w10_median": median_w10,
        "w10_f64_median": median_w10_f64,
        "w10_variance": var_w10,
        "w10_f64_variance": var_w10_f64,
    }
    return benhmark_dict


def benchmark_dtype(
    n_runs: int,
    win_size: int = 1024,
    n_iters: int = 100,
    x_set: Tuple[float, float] = (-2.0, 1.0), 
    y_set: Tuple[float, float] = (-1.5, 1.5), 
) -> tuple[float, npt.NDArray[Any], float, npt.NDArray[Any], float, float]:
    """
    Compare ``w3_main`` performance for float64 versus float32 inputs.

    Args:
        n_runs (int): Number of benchmark runs per dtype.

    Returns:
        tuple: Timing, result arrays, and variance values for both dtypes.
    """
    median_w3_64, mandelbrot_set_w3_64, v64 = benchmark(
        w3_main, n_iters, x_set, y_set, win_size, np.float64, n_runs=n_runs
    )
    median_w3_32, mandelbrot_set_w3_32, v32 = benchmark(
        w3_main, n_iters, x_set, y_set, win_size, np.float32, n_runs=n_runs
    )

    return (
        median_w3_64,
        mandelbrot_set_w3_64,
        median_w3_32,
        mandelbrot_set_w3_32,
        v64,
        v32,
    )

def benchmark_numba_imp(
    n_runs: int = 3,
) -> tuple[float, float, float, float, float, float]:
    """
    Benchmark different Numba-based Mandelbrot implementations.

    Args:
        n_runs (int): Number of runs for each implementation.

    Returns:
        tuple: Median and variance values for full-jit, hybrid, and parallel versions.
    """
    print("full jit approach")
    w3_median, w3_res, w3_var = benchmark(
        w3_main, 100, (-2.0, 1.0), (-1.5, 1.5), 1024, np.float64, n_runs=n_runs
    )
    print("Python loops approach")
    hyb_median, hyb_res, hyb_var = benchmark(
        numba_hybrid, 100, (-2.0, 1.0), (-1.5, 1.5), 1024, np.float64, n_runs=n_runs
    )
    print("Parallel numba")
    par_median, par_res, par_var = benchmark(
        w3_parallel_main, 100, (-2.0, 1.0), (-1.5, 1.5), 1024, np.float64, n_runs=n_runs
    )
    return w3_median, w3_var, hyb_median, hyb_var, par_median, par_var


if __name__ == "__main__":
    # seahorse = w_1_5_main(max_iters=500, x_set=(-0.8, -0.7), y_set=(0.05, 0.15), win_size=1024)
    # elephant = w_1_5_main(max_iters=500, x_set=(0.175, 0.375), y_set=(-0.1, 0.1), win_size=1024)
    # deep_seahorse = w_1_5_main(max_iters=2000, x_set=(-0.7487667139, -0.7487667078), y_set=(0.1236408449, 0.1236408510), win_size=1024)

    median_w4, mandelbrot_set_w4, var_w4 = w4_main(
        100, (-2.0, 1.0), (-1.5, 1.5), 1024, n_runs=3
    )

    # mandelbrot_set, medians = w4_main(win_size=1024)
    # from multiprocessing_helpers import plot_medians
    # plot_medians(medians, range(1, len(medians)+1))
    # w4_monte_carlo(NUM_RUNS=10_000)
    # plt.imshow(w3_mandel, cmap='twilight_shifted_r')
    # plt.colorbar()
    # plt.show()
