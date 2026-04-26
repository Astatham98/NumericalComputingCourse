import time
from matplotlib import pyplot as plt
import pyopencl as cl
import numpy as np


def vector_addition(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Add two vectors element-wise using PyOpenCL.
    Parameters:
    a (np.ndarray): First input vector.
    b (np.ndarray): Second input vector.
    Returns:
    np.ndarray: The element-wise sum of the input vectors.
    """
    KERNEL_SRC = """
    __kernel void sum(
    __global const float *a_g, __global const float *b_g, __global float *res_g)
    {
    int gid = get_global_id(0);
    res_g[gid] = a_g[gid] + b_g[gid];
    }
    """

    ctx = cl.create_some_context(interactive=False)
    queue = cl.CommandQueue(ctx)
    prog = cl.Program(ctx, KERNEL_SRC).build()

    a = np.array(a, dtype=np.float32)
    b = np.array(b, dtype=np.float32)
    assert a.size == b.size, "Input arrays must have the same size"
    N = a.size
    res = np.empty_like(a)

    a_dev = cl.Buffer(
        ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=a
    )
    b_dev = cl.Buffer(
        ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=b
    )
    res_g = cl.Buffer(ctx, cl.mem_flags.WRITE_ONLY, res.nbytes)

    # Pass the queue, shape, work group size, the output buffer to the kernel
    prog.sum(queue, (N,), None, a_dev, b_dev, res_g)
    # Pass the queue, the output array, and the output buffer to copy the data back to the host
    cl.enqueue_copy(queue, res, res_g)
    queue.finish()

    print(res)  # → [ 0  1  4  9 16 25 36 49]
    assert np.allclose(res, a + b)
    return res


def gpu_mandelbrot(
    max_iters: int = 100,
    x_set: tuple = (-2.0, 1.0),
    y_set: tuple = (-1.5, 1.5),
    win_size: int = 1024,
    gpu_devices: list[cl.Device] | None = None,
) -> np.ndarray:

    KERNEL_SRC = """
    __kernel void mandelbrot(
    __global int *result,
    const float x_min, const float x_max,
    const float y_min, const float y_max,
    const int max_iter, const int win_size)
    {
    int col = get_global_id(0);
    int row = get_global_id(1);
    if (col >= win_size || row >= win_size) return;
    float c_real = x_min + (x_max - x_min) * col / (float)win_size;
    float c_imag = y_min + (y_max - y_min) * row / (float)win_size;
    
    float zr = 0.0f, zi = 0.0f;
    int count = 0;
    while (zr * zr + zi * zi <= 4.0f && count < max_iter) {
        float tmp = zr*zr - zi*zi + c_real;
        zi = 2.0f * zr * zi + c_imag;
        zr = tmp;
        count++;
    }
    result[row * win_size + col] = count;
    }
    """

    x_min, x_max = x_set
    y_min, y_max = y_set
    ctx = cl.Context(devices=gpu_devices)
    queue = cl.CommandQueue(ctx)
    prog = cl.Program(ctx, KERNEL_SRC).build()

    result = np.zeros((win_size, win_size), dtype=np.int32)
    result_g = cl.Buffer(ctx, cl.mem_flags.WRITE_ONLY, result.nbytes)

    # Warmup
    prog.mandelbrot(
        queue,
        (64, 64),
        None,
        result_g,
        np.float32(x_min),
        np.float32(x_max),
        np.float32(y_min),
        np.float32(y_max),
        np.int32(max_iters),
        np.int32(64),
    )
    queue.finish()

    # Actual run
    t0 = time.perf_counter()
    prog.mandelbrot(
        queue,
        (win_size, win_size),
        None,
        result_g,
        np.float32(x_min),
        np.float32(x_max),
        np.float32(y_min),
        np.float32(y_max),
        np.int32(max_iters),
        np.int32(win_size),
    )
    queue.finish()
    elapsed = time.perf_counter() - t0
    print(f"GPU Mandelbrot computed in {elapsed:.4f} seconds")

    cl.enqueue_copy(queue, result, result_g)
    queue.finish()

    return result, elapsed


def gpu_mandelbrot_f64(
    max_iters: int = 100,
    x_set: tuple = (-2.0, 1.0),
    y_set: tuple = (-1.5, 1.5),
    win_size: int = 1024,
    gpu_devices: list[cl.Device] | None = None,
) -> np.ndarray:

    KERNEL_SRC = get_f64_mandelbrot_kernel()

    x_min, x_max = x_set
    y_min, y_max = y_set
    ctx = cl.Context(devices=gpu_devices)
    queue = cl.CommandQueue(ctx)
    prog = cl.Program(ctx, KERNEL_SRC).build()

    result = np.zeros((win_size, win_size), dtype=np.int32)
    result_g = cl.Buffer(ctx, cl.mem_flags.WRITE_ONLY, result.nbytes)

    # Warmup
    prog.mandelbrot_f64(
        queue,
        (64, 64),
        None,
        result_g,
        np.float64(x_min),
        np.float64(x_max),
        np.float64(y_min),
        np.float64(y_max),
        np.int32(max_iters),
        np.int32(64),
    )
    queue.finish()

    # Actual run
    t0 = time.perf_counter()
    prog.mandelbrot_f64(
        queue,
        (win_size, win_size),
        None,
        result_g,
        np.float64(x_min),
        np.float64(x_max),
        np.float64(y_min),
        np.float64(y_max),
        np.int32(max_iters),
        np.int32(win_size),
    )
    queue.finish()
    elapsed = time.perf_counter() - t0
    print(f"GPU Mandelbrot computed in {elapsed:.4f} seconds")

    cl.enqueue_copy(queue, result, result_g)
    queue.finish()

    return result, elapsed


def get_f64_mandelbrot_kernel() -> str:
    return """
    __kernel void mandelbrot_f64(
    __global int *result,
    const double x_min, const double x_max,
    const double y_min, const double y_max,
    const int max_iter, const int win_size)
    {
    int col = get_global_id(0);
    int row = get_global_id(1);
    if (col >= win_size || row >= win_size) return;
    double c_real = x_min + (x_max - x_min) * col / (double)win_size;
    double c_imag = y_min + (y_max - y_min) * row / (double)win_size;
    
    double zr = 0.0, zi = 0.0;
    int count = 0;
    while (zr * zr + zi * zi <= 4.0 && count < max_iter) {
        double tmp = zr*zr - zi*zi + c_real;
        zi = 2.0 * zr * zi + c_imag;
        zr = tmp;
        count++;
    }
    result[row * win_size + col] = count;
    }
    """


if __name__ == "__main__":
    m, t = gpu_mandelbrot_f64(win_size=4096)
    plt.imshow(m, cmap="twilight_shifted_r")
    plt.show()
