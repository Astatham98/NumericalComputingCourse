import numpy as np
import matplotlib.pyplot as plt
import time

def main(max_iters: int = 100) -> None:
    """Generate and plot the Mandelbrot set.
    Args: 
        max_iter (int): Maximum number of iterations.
    """
    start_time = time.time()
    x_set, y_set = [-2, 1], [-1.5, 1.5]
    width = np.linspace(x_set[0], x_set[1], 100)
    height = np.linspace(y_set[0], y_set[1], 100)
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
    
    end_time = time.time()
    print(f"Execution took: {end_time - start_time:.2f} seconds")
    
    plt.imshow(mandelbrot_set, cmap='twilight_shifted_r')
    plt.colorbar()
    plt.show()

if __name__ == "__main__":
    main()
    