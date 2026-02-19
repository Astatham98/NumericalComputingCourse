import numpy as np
import matplotlib.pyplot as plt
import time
from numba import jit

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

def benchmark(func, *args, n_runs=3):
    """Run a function multiple times and print the average execution time."""
    times = []
    for _ in range(n_runs):
        start_time = time.perf_counter()
        result = func(*args)
        end_time = time.perf_counter()
        times.append(end_time - start_time)
    median = np.median(times)
    print(f"Median execution time over {n_runs} runs: {median:.6f} seconds")
    return median, result

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
    # Reshape to 2D and plot        
    mandelbrot_set = np.reshape(mandelbrot_set, (len(width), len(height))).T
    return mandelbrot_set

@timing
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

    #points = [complex(x, y) for x in width for y in height]
    
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
    
    

if __name__ == "__main__":
    def benchmark_all(n_runs=3):
        median_w1, mandelbrot_set_w1 = benchmark(w1_main, 100, (-2.0, 1.0), (-1.5, 1.5), 1024, n_runs=n_runs)
        median_w2, mandelbrot_set_w2 = benchmark(w2_main, 100, (-2.0, 1.0), (-1.5, 1.5), 1024, n_runs=n_runs)
        median_w1_5, mandelbrot_set_w1_5 = benchmark(w_1_5_main, 100, (-2.0, 1.0), (-1.5, 1.5), 1024, n_runs=n_runs)
        if np.allclose(mandelbrot_set_w1, mandelbrot_set_w2):
            print("All implementations produce the same result.")
        else:
            print("Results differ between implementations.")

    benchmark_all(n_runs=1)
    # plt.imshow(mandelbrot_set_w1_5, cmap='twilight_shifted_r')
    # plt.colorbar()
    # plt.show()
    