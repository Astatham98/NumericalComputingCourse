import random
import time
from multiprocessing import Pool
from numba import njit
import numpy as np

def timing(func):
    """Decorator to measure the execution time of a function."""
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        print(f"Execution took: {end_time - start_time:.6f} seconds")
        return result
    return wrapper

@timing
def estimate_pi_circle(num_samples: int = 100):
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

def estimate_pi_chunk(num_samples):
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

def estimate_pi_parallel(num_samples, num_processes=4):
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


# mandelbrot jit

@njit
def mandelbrot_pixel(c_real, c_imag, max_iter):
    z_real = z_imag = 0.0
    for i in range(max_iter):
        z_sq = z_real*z_real + z_imag*z_imag
        if z_sq > 4.0: 
            return i
        z_imag = 2.0*z_real*z_imag + c_imag
        z_real = z_real*z_real - z_imag*z_imag + c_real
    return max_iter

@njit
def compute_mandelbrot_chunk(start_row, end_row, num_points,
min_x, max_x, min_y, max_y, max_iterations):
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

def mandelbrot_serial(N, x_min, x_max, y_min, y_max, max_iter=100):
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

def _worker(args):
    return compute_mandelbrot_chunk(*args)

def mandelbrot_parallel(N, x_min, x_max, y_min, y_max, max_iter=100, num_processes=4):
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

    Returns:
        np.ndarray: A 2D array containing the computed Mandelbrot set values.
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
        for _ in range(3):
            start_time = time.perf_counter()
            results = pool.map(_worker, chunks)
            total = np.vstack(results)
            end_time = time.perf_counter()
            times.append(end_time - start_time)
    return total, times

def plot_medians(median_vals, cores):
    from matplotlib import pyplot as plt
    plt.figure(figsize=(10, 6))
    plt.plot(cores, median_vals, marker='o')
    plt.title('Median Execution Time vs Number of Cores')
    plt.xlabel('Number of Cores')
    plt.ylabel('Median Execution Time (seconds)')
    plt.show()