import random
import time
from multiprocessing import Pool
from typing import Any, Callable
from numba import njit
import numpy as np
import numpy.typing as npt

def timing(func: Callable[..., Any]) -> Callable[..., Any]:
    """Decorator to measure the execution time of a function."""
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        print(f"Execution took: {end_time - start_time:.6f} seconds")
        return result
    return wrapper

@timing
def estimate_pi_circle(num_samples: int = 100) -> float:
    """
    Estimates the value of pi using the Monte Carlo method by generating random points within a unit circle.

    Args:
        num_samples (int): The number of random points to generate. Defaults to 100.

    Returns:
        float: An estimate of the value of pi.
    """
    inside_circle = 0
    for _ in range(num_samples):
        x, y = random.random(), random.random()
        if x*x + y*y <= 1:
            inside_circle += 1
    return 4 * inside_circle / num_samples

def estimate_pi_chunk(num_samples: int) -> int:
    """
    Estimates the number of points falling within a unit circle using the Monte Carlo method.

    Args:
        num_samples (int): The number of random points to generate.

    Returns:
        int: The number of points falling within a unit circle.
    """
    inside_circle = 0
    for _ in range(num_samples):
        x, y = random.random(), random.random()
        if x*x + y*y <= 1:
            inside_circle += 1
    return inside_circle

def estimate_pi_parallel(num_samples: int, num_processes: int = 4) -> float:
    """
    Estimates the value of pi using the Monte Carlo method by generating random points within a unit circle in parallel.

    Args:
        num_samples (int): The number of random points to generate. Defaults to 100.
        num_processes (int): The number of processes to use in parallel. Defaults to 4.

    Returns:
        float: An estimate of the value of pi.
    """
    samples_per_process = num_samples // num_processes
    tasks = [samples_per_process] * num_processes
    with Pool(processes=num_processes) as pool:
        results = pool.map(estimate_pi_chunk, tasks)
    return 4 * sum(results) / num_samples

def test_granulaity(total_work: int, chunk_size: int, num_processes: int) -> tuple[float, float]:
    """
    Benchmark Monte Carlo pi computation for a given chunk granularity.

    Args:
        total_work (int): Total number of random samples.
        chunk_size (int): Number of samples per chunk.
        num_processes (int): Number of worker processes.

    Returns:
        tuple[float, float]: Execution time and pi estimate.
    """
    n_chunks = total_work // chunk_size
    tasks = [chunk_size] * n_chunks
    t0 = time.perf_counter()
    if num_processes == 1:
        results = [estimate_pi_chunk(t) for t in tasks]
    else:
        with Pool(processes=num_processes) as pool:
            results = pool.map(estimate_pi_chunk, tasks)
    t1 = time.perf_counter()
    return t1 - t0, 4 * sum(results) / total_work

def subtract_seven(n: int) -> int:
    """
    Subtract seven from a numeric value.

    Args:
        n: Input numeric value.

    Returns:
        Numeric value equal to ``n - 7``.
    """
    return n - 7

def MFR(N: int = 1_000_000, ran_range: list[int] = [10, 100]) -> tuple[int, float]:
    """
    Apply map/filter/reduce serially to randomly generated integers.

    Args:
        N (int): Number of random integers to generate.
        ran_range (list[int]): Inclusive range ``[low, high]`` for generation.

    Returns:
        tuple[int, float]: Sum of odd transformed values and execution time.
    """
    rands = [random.randint(ran_range[0], ran_range[1]) for n in range(N)]
    t0 = time.perf_counter()
    nums = map(subtract_seven, rands)
    odds = filter(lambda n: n % 2 == 1, nums)
    return sum(odds), time.perf_counter() - t0

def MPF(N: int = 1_000_000, ran_range: list[int] = [10, 100], num_processes: int = 4) -> tuple[int, float]:
    """
    Apply map/filter workflow with multiprocessing for the map step.

    Args:
        N (int): Number of random integers to generate.
        ran_range (list[int]): Inclusive range ``[low, high]`` for generation.
        num_processes (int): Number of worker processes.

    Returns:
        tuple[int, float]: Sum of odd transformed values and execution time.
    """
    rands = [random.randint(ran_range[0], ran_range[1]) for n in range(N)]
    with Pool(num_processes) as pool:
        t0 = time.perf_counter()
        nums = pool.map(subtract_seven, rands)
    odds = filter(lambda n: n % 2 == 1, nums)
    return sum(odds), time.perf_counter() - t0
    

@njit(cache=True)
def mandelbrot_pixel(c_real: float, c_imag: float, max_iter: int) -> int:
    """
    Compute escape iteration count for a single Mandelbrot pixel.

    Args:
        c_real (float): Real component of the complex point.
        c_imag (float): Imaginary component of the complex point.
        max_iter (int): Maximum number of iterations.

    Returns:
        int: Escape iteration count, or ``max_iter`` if bounded.
    """
    z_real = z_imag = 0.0
    for i in range(max_iter):
        z_sq = z_real*z_real + z_imag*z_imag
        if z_sq > 4.0: 
            return i
        z_imag = 2.0*z_real*z_imag + c_imag
        z_real = z_real*z_real - z_imag*z_imag + c_real
    return max_iter

@njit(cache=True)
def compute_mandelbrot_chunk(
    start_row: int,
    end_row: int,
    num_points: int,
    min_x: float,
    max_x: float,
    min_y: float,
    max_y: float,
    max_iterations: int,
) -> npt.NDArray[np.int32]:
    """
    Computes a chunk of the Mandelbrot set.

    Args:
        start_row (int): The starting row of the chunk.
        end_row (int): The ending row of the chunk.
        num_points (int): The number of points in the x-axis.
        min_x (float): The minimum value of the x-axis.
        max_x (float): The maximum value of the x-axis.
        min_y (float): The minimum value of the y-axis.
        max_y (float): The maximum value of the y-axis.
        max_iterations (int): The maximum number of iterations.

    Returns:
        np.ndarray: A 2D array containing the computed Mandelbrot set values.
    """
    result = np.empty((end_row - start_row, num_points), dtype=np.int32)
    x_step = (max_x - min_x) / num_points
    y_step = (max_y - min_y) / (end_row - start_row)
    for row in range(end_row - start_row):
        c_imag = min_y + (row + start_row) * y_step
        for col in range(num_points):
            c_real = min_x + col * x_step
            result[row, col] = mandelbrot_pixel(c_real, c_imag, max_iterations)
    return result

def mandelbrot_serial(
    N: int,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    max_iter: int = 100,
) -> npt.NDArray[np.int32]:
    """
    Computes the Mandelbrot set serially.

    Args:
        N (int): The number of points in the x-axis.
        x_min (float): The minimum value of the x-axis.
        x_max (float): The maximum value of the x-axis.
        y_min (float): The minimum value of the y-axis.
        y_max (float): The maximum value of the y-axis.
        max_iter (int): The maximum number of iterations. Defaults to 100.

    Returns:
        np.ndarray: A 2D array containing the computed Mandelbrot set values.
    """
    return compute_mandelbrot_chunk(0, N, N, x_min, x_max, y_min, y_max, max_iter)

def _worker(args: tuple[int, int, int, float, float, float, float, int]) -> npt.NDArray[np.int32]:
    """
    Unpack argument tuples for multiprocessing chunk workers.

    Args:
        args (tuple): Positional arguments for ``compute_mandelbrot_chunk``.

    Returns:
        np.ndarray: Computed Mandelbrot chunk.
    """
    return compute_mandelbrot_chunk(*args)

def get_pool(
    n_processes: int,
    grid: list[tuple[int, int, int, float, float, float, float, int]],
) -> Pool:
    """
    Get a multiprocessing pool from the given grid of arguments.

    Args:
        n_processes (int): The number of processes to use in the pool.
        grid (list of tuples): A list of tuples containing the arguments to pass to _worker for warmup.

    Returns:
        multiprocessing.Pool: A multiprocessing pool object.
    """
    pool = Pool(processes=n_processes)
    pool.map(_worker, grid)
    return pool

def mandelbrot_parallel(
    N: int,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    max_iter: int = 100,
    num_processes: int = 4,
    NoRuns: int = 3,
) -> tuple[npt.NDArray[np.int32], list[float]]:
    """
    Computes the Mandelbrot set in parallel.

    Args:
        N (int): The number of points in the x-axis.
        x_min (float): The minimum value of the x-axis.
        x_max (float): The maximum value of the x-axis.
        y_min (float): The minimum value of the y-axis.
        y_max (float): The maximum value of the y-axis.
        max_iter (int): The maximum number of iterations. Defaults to 100.
        num_processes (int): The number of processes to use in parallel. Defaults to 4.
        NoRuns (int): The number of times to run the computation. Defaults to 3.

    Returns:
        total (np.ndarray): A 2D array containing the computed Mandelbrot set values.
        times (list): A list of execution times for each process.
    """
    times = []

    rows_per_process = N // num_processes
    chunks, row = [], 0
    while row < N:
        row_end = min(row + rows_per_process, N)
        chunks.append((row, row_end, N, x_min, x_max, y_min, y_max, max_iter))
        row = row_end

    with Pool(processes=num_processes) as pool:
        pool.map(_worker, chunks)
        for _ in range(NoRuns):
            start_time = time.perf_counter()
            results = pool.map(_worker, chunks)
            total = np.vstack(results)
            end_time = time.perf_counter()
            times.append(end_time - start_time)
    return total, times

def mandelbrot_parallel_chunks(
    N: int,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    max_iter: int = 100,
    n_workers: int = 4,
    n_chunks: int | None = None,
    pool: Pool | None = None,
) -> tuple[npt.NDArray[np.int32], float]:
    """
    Compute the Mandelbrot set with configurable worker/chunk scheduling.

    Args:
        N (int): Output grid size along each axis.
        x_min (float): Minimum real-axis value.
        x_max (float): Maximum real-axis value.
        y_min (float): Minimum imaginary-axis value.
        y_max (float): Maximum imaginary-axis value.
        max_iter (int): Maximum iteration count per point.
        n_workers (int): Number of worker processes.
        n_chunks (int | None): Number of row chunks. Defaults to ``n_workers``.
        pool (Pool | None): Optional pre-warmed process pool.

    Returns:
        tuple[np.ndarray, float]: Mandelbrot array and execution time.
    """
    if n_chunks is None:
        n_chunks = n_workers
    chunk_size = max(1, N // n_chunks)
    chunks, row = [], 0
    while row < N:
        row_end = min(row + chunk_size, N)
        chunks.append((row, row_end, N, x_min, x_max, y_min, y_max, max_iter))
        row = row_end
    if pool is not None: # caller manages Pool; skip startup + warm-up
        t0 = time.perf_counter()
        res = np.vstack(pool.map(_worker, chunks))
        end_time = time.perf_counter() - t0
        return res, end_time
    tiny = [(0, 8, 8, x_min, x_max, y_min, y_max, max_iter)]
    with Pool(processes=n_workers) as p:
        p.map(_worker, tiny) # warm-up: load JIT cache in workers
        t0 = time.perf_counter()
        parts = p.map(_worker, chunks)
    res = np.vstack(parts)
    end_time = time.perf_counter() - t0
    return res, end_time

def plot_medians(median_vals: list[float], cores: list[int]) -> None:
    """
    Plot median execution time as a function of core count.

    Args:
        median_vals (list[float]): Median timings.
        cores (list[int]): Corresponding number of cores.
    """
    from matplotlib import pyplot as plt
    plt.figure(figsize=(10, 6))
    plt.plot(cores, median_vals, marker='o')
    plt.title('Median Execution Time vs Number of Cores')
    plt.xlabel('Number of Cores')
    plt.ylabel('Median Execution Time (seconds)')
    plt.show()
    
if __name__ == "__main__":
    pass