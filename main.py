import numpy as np
import matplotlib.pyplot as plt
import time
"""
Mandelbrot Set Generator
Author : Alex Statham
Course : Numerical Scientific Computing 2026
"""

def main(max_iters: int = 100, x_set: list = [-2, 1], y_set: list = [-1.5, 1.5], win_size: int = 100) -> np.ndarraygit :
    """Generate and plot the Mandelbrot set.
    Args: 
        max_iter (int): Maximum number of iterations.
        x_set (list): X-axis range.
        y_set (list): Y-axis range.
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
    

if __name__ == "__main__":
    start_time = time.time()
    mandelbrot_set = main(win_size=1024)
    end_time = time.time()
    print(f"Execution took: {end_time - start_time:.2f} seconds")

    plt.imshow(mandelbrot_set, cmap='twilight_shifted_r')
    plt.colorbar()
    plt.show()
    