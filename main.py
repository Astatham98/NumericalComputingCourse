import numpy as np
import matplotlib.pyplot as plt

def main(max_iters: int = 100) -> None:
    """Generate and plot the Mandelbrot set.
    Args: 
        max_iter (int): Maximum number of iterations.
    """
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
    for i, z in enumerate(points):
        c = z
        for j in range(max_iters):
            # Break if |Z| > 2
            if abs(z) > 2:
                mandelbrot_set[i] = j
                break
            z = z**2 + c
            
    # Reshape to 2D and plot        
    mandelbrot_set = np.reshape(mandelbrot_set, (len(width), len(height)))
    plt.imshow(mandelbrot_set)
    plt.colorbar()
    plt.show()

if __name__ == "__main__":
    main()
