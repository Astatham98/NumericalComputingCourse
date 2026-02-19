import numpy as np
import matplotlib.pyplot as plt
import time
from numba import jit

"""
Mandelbrot Set Generator
Author : Alex Statham
Course : Numerical Scientific Computing 2026
"""
def f(C: np.ndarray[complex], Z: np.ndarray, M: np.ndarray, max_iters: int) -> np.ndarray:
    for _ in range(max_iters):
        mask = np.abs(Z) <= 2
        Z[mask] = Z[mask]**2 + C[mask]
        M[mask] += 1
    return M

def main(max_iters: int = 100, 
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
    X, Y = np.meshgrid(width, height)
    C = X + 1j * Y

    #points = [complex(x, y) for x in width for y in height]
    
    # Based on the 100x100 points, compute the mandelbrot set
    mandelbrot_set = np.zeros(C.shape, dtype=C.dtype)
    # Get n and c

    Z = np.zeros_like(C, dtype=complex)
    M = np.zeros_like(C, dtype=int)  
    mandelbrot_set = f(C, Z, M, max_iters)
    return mandelbrot_set
    

if __name__ == "__main__":
    start_time = time.time()
    mandelbrot_set = main(win_size=1024)
    end_time = time.time()
    print(f"Execution took: {end_time - start_time:.2f} seconds")

    plt.imshow(mandelbrot_set, cmap='twilight_shifted_r')
    plt.colorbar()
    plt.show()
    