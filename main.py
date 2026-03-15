import numpy as np
import matplotlib.pyplot as plt
import time
from numba import jit, njit, prange
#from line_profiler import profile
from typing import Tuple, List, Dict
from multiprocessing import Pool
import psutil
from multiprocessing_helpers import mandelbrot_serial, mandelbrot_parallel


"""
Mandelbrot Set Generator
Author : Alex Statham
Course : Numerical Scientific Computing 2026
"""
def timing(func):
    """Decorator to measure the execution time of a function."""
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        print(f"Execution took: {end_time - start_time:.6f} seconds")
        return result
    return wrapper

def benchmark(func, *args, n_runs=3) -> Tuple[float, np.ndarray, float]:
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

@timing
def w1_main(max_iters: int = 100, 
        x_set: tuple = (-2.0, 1.0),    # Changed to tuple + floats
        y_set: tuple = (-1.5, 1.5),    # Changed to tuple + floats
        win_size: int = 100) -> np.ndarray :
    """Generate and plot the Mandelbrot set.
    Args: 
        max_iter (int): Maximum number of iterations.
        x_set (tuple): X-axis range.
        y_set (tuple): Y-axis range.
        win_size (int): Number of points in each axis.
    Returns:
        np.ndarray: Mandelbrot set values in a 2D array.
    """
    def w1_f(points: np.ndarray[complex], mandelbrot_Set: np.ndarray[int], max_iters: int = 100) -> np.ndarray:
            for i, c in enumerate(points):
                z = 0
                for j in range(max_iters):
                    z = z**2 + c
                    # Break if |Z| > 2
                    if abs(z) > 2:
                        mandelbrot_set[i] = j
                        break
                else:
                    mandelbrot_set[i] = max_iters
            return mandelbrot_set


    x_set, y_set = [-2, 1], [-1.5, 1.5]
    width = np.linspace(x_set[0], x_set[1], win_size)
    height = np.linspace(y_set[0], y_set[1], win_size)
    points = []
    for w in width:
        for h in height:
            points.append(w + h*1j)
    points = np.array(points)

    # Based on the 100x100 points, compute the mandelbrot set
    mandelbrot_set = np.zeros(points.shape, dtype=int)
    # Get n and c
    mandelbrot_set = w1_f(points, mandelbrot_set, max_iters)
    # Reshape to 2D and plot        
    mandelbrot_set = np.reshape(mandelbrot_set, (len(width), len(height))).T
    return mandelbrot_set

def w2_main(max_iters: int = 100, 
        x_set: tuple = (-2.0, 1.0),    # Changed to tuple + floats
        y_set: tuple = (-1.5, 1.5),    # Changed to tuple + floats
        win_size: int = 100) -> np.ndarray :
    """Generate and plot the Mandelbrot set.
    Args: 
        max_iter (int): Maximum number of iterations.
        x_set (tuple): X-axis range.
        y_set (tuple): Y-axis range.
        win_size (int): Number of points in each axis.
    Returns:
        np.ndarray: Mandelbrot set values in a 2D array.
    """
    @timing
    def f(C: np.ndarray[complex], Z: np.ndarray, M: np.ndarray, max_iters: int) -> np.ndarray:
        for _ in range(max_iters):
            # Calculate the mask for every point in Z
            mask = np.abs(Z) <= 2
            # Calculate the new Z values 
            Z[mask] = Z[mask]**2 + C[mask]
            # Update M for points that are still within the escape radius
            M[mask] += 1
        return M

    width = np.linspace(x_set[0], x_set[1], win_size)
    height = np.linspace(y_set[0], y_set[1], win_size)
    X, Y = np.meshgrid(width, height)
    C = X + 1j * Y
    
    # Based on the 100x100 points, compute the mandelbrot set
    mandelbrot_set = np.zeros(C.shape, dtype=C.dtype)

    # Initialize Z and M arrays
    Z = np.zeros_like(C, dtype=complex)
    M = np.zeros_like(C, dtype=int) 
    # Compute the Mandelbrot set using the function f 
    mandelbrot_set = f(C, Z, M, max_iters)
    return mandelbrot_set

@jit(nopython=True)
def f_jit(c: complex, max_iters: int) -> int:
    z = 0
    for j in range(max_iters):
        z = z**2 + c
        # Break if |Z| > 2
        if abs(z) > 2:
            return j
    return max_iters

@timing
def w_1_5_main(max_iters: int = 100, 
         x_set: tuple = (-2.0, 1.0),    # Changed to tuple + floats
         y_set: tuple = (-1.5, 1.5),    # Changed to tuple + floats
         win_size: int = 100) -> np.ndarray :
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
    
    
def w2_memory_access(N=1000):
    np.random.seed(42)
    A = np.random.rand(N, N)

    @timing
    def row_major_sum(A):
        for i in range(N): s = np.sum(A[i, :])

    @timing
    def column_major_sum(A):
        for j in range(N): s = np.sum(A[:, j])

    print("Using C order:")
    row_major_sum(A)
    column_major_sum(A)

    print("\nUsing Fortran order:")
    A_F = np.asfortranarray(A)
    row_major_sum(A_F)
    column_major_sum(A_F)

def w2_scaling():
    sizes = [1024, 2048, 4096, 8192]
    times = []
    for size in sizes:
        time, _ = benchmark(w2_main, 100, (-2.0, 1.0), (-1.5, 1.5), size, n_runs=1)
        times.append(time)

    plt.figure(figsize=(10, 6))
    plt.plot(sizes, times, marker='o')
    plt.title('Execution Time vs Window Size for w2_main')
    plt.xlabel('Window Size (win_size)')
    plt.ylabel('Execution Time (seconds)')
    plt.show()
    return sizes, times

@timing
@njit
def w3_f(C: np.ndarray[complex], mandelbrot_set: np.ndarray, max_iters: int) -> np.ndarray:
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
                if z. real *z. real + z. imag *z. imag > 4.0: 
                    mandelbrot_set[i,j] = k
                    break
            else:
                mandelbrot_set[i,j] = max_iters
    return mandelbrot_set


@timing
@njit(parallel=True)
def w3_parallel(C: np.ndarray[complex], mandelbrot_set: np.ndarray, winsize: int,  max_iters: int) -> np.ndarray:
    for i in prange(winsize):
        for j in prange(winsize):
            z = 0j
            n = 0
            while n < max_iters and z. real *z . real +z. imag *z . imag <= 4.0:
                z = z*z + C[i, j]
                n += 1
            mandelbrot_set [i , j ] = n
    return mandelbrot_set

def w3_main(max_iters: int = 100, 
        x_set: Tuple[float, float] = (-2.0, 1.0),    # Changed to tuple + floats
        y_set: Tuple[float, float] = (-1.5, 1.5),    # Changed to tuple + floats
        win_size: int = 100,
        dtype: np.dtype = np.float64) -> np.ndarray :
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

def w3_parallel_main(max_iters: int = 100, 
        x_set: tuple = (-2.0, 1.0),    # Changed to tuple + floats
        y_set: tuple = (-1.5, 1.5),    # Changed to tuple + floats
        win_size: int = 100,
        dtype: np.dtype = np.float64) -> np.ndarray :
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
def mandelbrot_calc(c, max_iters=100):
    z = 0j
    for n in range (max_iters) :
        if z.real * z.real + z.imag * z.imag > 4.0:
            return n
        z = z *z + c
    return max_iters

@timing
def numba_hybrid(max_iters: int = 100, 
        x_set: tuple = (-2.0, 1.0),    # Changed to tuple + floats
        y_set: tuple = (-1.5, 1.5),    # Changed to tuple + floats
        win_size: int = 100,
        dtype: np.dtype = np.float64) -> np.ndarray :
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
            mandelbrot_set[i ,j] = mandelbrot_calc(c , max_iters)
    return mandelbrot_set


def w4_testing(max_iters: int = 100, 
        x_set: Tuple[float, float] = (-2.0, 1.0),    # Changed to tuple + floats
        y_set: Tuple[float, float] = (-1.5, 1.5),    # Changed to tuple + floats
        win_size: int = 100,
        dtype: np.dtype = np.float64) -> np.ndarray :
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
    print('Benchmarking serial approach')
    benchmark(mandelbrot_serial, win_size, x_set[0], x_set[1], y_set[0], y_set[1], max_iters, n_runs=4)
    
    medians = []
    print('Benchmarking parallel approach')
    for n_cores in range(1, psutil.cpu_count()+1):
        print(f'Running on {n_cores} cores')
        mandelbrot_set, timings = mandelbrot_parallel(win_size, x_set[0], x_set[1], y_set[0], y_set[1], max_iters, n_cores)
        #[print(f"Execution took: {t:.6f} seconds \n") for t in timings]
        print(f'Median execution time over 3 runs for {n_cores}: {np.median(timings):.6f} seconds')
        medians.append(np.median(timings))
    return mandelbrot_set, medians

def w4_main(max_iters: int = 100, 
        x_set: Tuple[float, float] = (-2.0, 1.0),    # Changed to tuple + floats
        y_set: Tuple[float, float] = (-1.5, 1.5),    # Changed to tuple + floats
        win_size: int = 100,
        dtype: np.dtype = np.float64,
        NoCores: int = psutil.cpu_count(),
        n_runs: int = 3) -> np.ndarray :
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

    mandelbrot_set, timings = mandelbrot_parallel(win_size, x_set[0], x_set[1], y_set[0], y_set[1], max_iters, NoCores, n_runs)
    return np.median(timings), mandelbrot_set, np.var(timings) 



def w5_main(max_iters: int = 100, 
        x_set: Tuple[float, float] = (-2.0, 1.0),    # Changed to tuple + floats
        y_set: Tuple[float, float] = (-1.5, 1.5),    # Changed to tuple + floats
        win_size: int = 100,
        dtype: np.dtype = np.float64) -> np.ndarray :
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
    pass

def benchmark_all(n_runs=3):
    print('Week 1: Naive python implementatio ')
    median_w1, mandelbrot_set_w1, var_w1 = benchmark(w1_main, 100, (-2.0, 1.0), (-1.5, 1.5), 1024, n_runs=n_runs)
    print('Week 2: numpy vectorization')
    median_w2, mandelbrot_set_w2, var_w2 = benchmark(w2_main, 100, (-2.0, 1.0), (-1.5, 1.5), 1024, n_runs=n_runs)
    print('Week 3: Naive numba')
    median_w1_5, mandelbrot_set_w1_5, var_w1_5 = benchmark(w_1_5_main, 100, (-2.0, 1.0), (-1.5, 1.5), 1024, n_runs=n_runs)
    print('Weel 3: optimized numba')
    median_w3, mandelbrot_set_w3, var_w3 = benchmark(w3_main, 100, (-2.0, 1.0), (-1.5, 1.5), 1024, n_runs=n_runs)
    print('Week 4: parallel computing')
    # W4 main now uses its own timings and we do not want to time the workers in the benchmark
    median_w4, mandelbrot_set_w4, var_w4 = w4_main(100, (-2.0, 1.0), (-1.5, 1.5), 1024, n_runs=n_runs)
    print(f'w4 median_w4 {median_w4}, w4 variance {var_w4}')
    #median_w5, mandelbrot_set_w5, var_w5 = benchmark(w5_main, 100, (-2.0, 1.0), (-1.5, 1.5), 1024, n_runs=n_runs)
    
    return median_w1, median_w2, median_w1_5, median_w3, var_w1, var_w2, var_w1_5, var_w3, median_w4, var_w4

def benchmark_dtype(n_runs):
    median_w3_64, mandelbrot_set_w3_64, v64 = benchmark(w3_main, 100, (-2.0, 1.0), (-1.5, 1.5), 1024, np.float64, n_runs=n_runs)
    median_w3_32, mandelbrot_set_w3_32, v32 = benchmark(w3_main, 100, (-2.0, 1.0), (-1.5, 1.5), 1024, np.float32, n_runs=n_runs)

    return median_w3_64, mandelbrot_set_w3_64, median_w3_32, mandelbrot_set_w3_32, v64, v32

def w4_monte_carlo(NUM_RUNS: int = 10_000) -> None:
    """
    Run Monte Carlo simulations for estimation of pi using Circle and Parallel
    implementations.

    Parameters:
    NUM_RUNS (int): Number of Monte Carlo simulations to run. Default is 10_000.
    """
    from multiprocessing_helpers import estimate_pi_circle, estimate_pi_parallel
    benchmark(estimate_pi_circle, NUM_RUNS, n_runs=3)
    for i in range(psutil.cpu_count(logical=False)):
        print(f'Running on {i+1} cores')
        benchmark(estimate_pi_parallel, NUM_RUNS, i+1, n_runs=3)



def benchmark_numba_imp(n_runs=3):
    print('full jit approach')
    w3_median, w3_res, w3_var = benchmark(w3_main, 100, (-2.0, 1.0), (-1.5, 1.5), 1024, np.float64, n_runs=n_runs)
    print('Python loops approach')
    hyb_median, hyb_res, hyb_var = benchmark(numba_hybrid, 100, (-2.0, 1.0), (-1.5, 1.5), 1024, np.float64, n_runs=n_runs)
    print('Parallel numba')
    par_median, par_res, par_var = benchmark(w3_parallel_main, 100, (-2.0, 1.0), (-1.5, 1.5), 1024, np.float64, n_runs=n_runs)
    return w3_median, w3_var, hyb_median, hyb_var, par_median, par_var


if __name__ == "__main__":
    #seahorse = w_1_5_main(max_iters=500, x_set=(-0.8, -0.7), y_set=(0.05, 0.15), win_size=1024)
    #elephant = w_1_5_main(max_iters=500, x_set=(0.175, 0.375), y_set=(-0.1, 0.1), win_size=1024)
    #deep_seahorse = w_1_5_main(max_iters=2000, x_set=(-0.7487667139, -0.7487667078), y_set=(0.1236408449, 0.1236408510), win_size=1024)

    
    benchmark_all(n_runs=3)    
    
    
    # mandelbrot_set, medians = w4_main(win_size=1024)    
    # from multiprocessing_helpers import plot_medians
    # plot_medians(medians, range(1, len(medians)+1))
    # w4_monte_carlo(NUM_RUNS=10_000)
    # plt.imshow(w3_mandel, cmap='twilight_shifted_r')
    # plt.colorbar()
    # plt.show()

    