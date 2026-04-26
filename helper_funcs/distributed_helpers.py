import dask, random, time, statistics
from dask import delayed
import numpy as np
import numpy.typing as npt
from dask.distributed import Client, LocalCluster
from helper_funcs.multiprocessing_helpers import estimate_pi_chunk, timing, compute_mandelbrot_chunk


def monte_carlo_chunk(
    total: int = 1_000_000, chunksize: int = 8
) -> tuple[float, float]:
    """
    Estimate pi serially by splitting work into equal Monte Carlo chunks.

    Args:
        total (int): Total number of random samples.
        chunksize (int): Number of chunks to split the work into.

    Returns:
        tuple[float, float]: Execution time and pi estimate.
    """
    samples = total // chunksize
    t0 = time.perf_counter()
    results = [estimate_pi_chunk(samples) for _ in range(chunksize)]
    t1 = time.perf_counter()
    print(
        f"Execution for serial took: {t1 - t0:.6f} seconds with total: {4 * sum(results) / total}"
    )
    return t1 - t0, 4 * sum(results) / total


def monte_carlo_dask(total: int = 1_000_000, chunksize: int = 8) -> tuple[float, float]:
    """
    Estimate pi using Dask delayed tasks on the local scheduler.

    Args:
        total (int): Total number of random samples.
        chunksize (int): Number of delayed tasks to create.

    Returns:
        tuple[float, float]: Execution time and pi estimate.
    """
    samples = total // chunksize
    task = [delayed(estimate_pi_chunk)(samples) for _ in range(chunksize)]
    t0 = time.perf_counter()
    results = dask.compute(*task)
    t1 = time.perf_counter()
    print(
        f"Execution  for dask took: {t1 - t0:.6f} seconds with total: {4 * sum(results) / total}"
    )
    return t1 - t0, 4 * sum(results) / total


def load_dask_client_local(
    workers: int = 4, threads_per_worker: int = 1
) -> tuple[Client, LocalCluster]:
    """
    Create a local Dask cluster and client.

    Args:
        workers (int): Number of Dask workers.
        threads_per_worker (int): Threads per worker process.

    Returns:
        tuple[Client, LocalCluster]: Active Dask client and cluster.
    """
    cluster = LocalCluster(n_workers=workers, threads_per_worker=threads_per_worker)
    client = Client(cluster)
    print(f"Dashboard: {client.dashboard_link}")
    return client, cluster


def load_dask_client(ip: str) -> Client:
    """
    Connect to an existing Dask scheduler by IP.

    Args:
        ip (str): Scheduler host IP address.

    Returns:
        Client: Connected Dask client.
    """
    client = Client(f"tcp://{ip}:8786")
    return client


def monte_carlo_dask_client(
    total: int = 1_000_000,
    chunksize: int = 8,
    workers: int = 4,
    threads_per_worker: int = 1,
) -> tuple[float, float]:
    """
    Estimate pi with Dask after explicitly creating a local client.

    Args:
        total (int): Total number of random samples.
        chunksize (int): Number of delayed tasks to create.
        workers (int): Number of Dask workers.
        threads_per_worker (int): Threads per worker process.

    Returns:
        tuple[float, float]: Execution time and pi estimate.
    """
    client, cluster = load_dask_client_local(workers, threads_per_worker)
    samples = total // chunksize
    task = [delayed(estimate_pi_chunk)(samples) for _ in range(chunksize)]
    t0 = time.perf_counter()
    results = dask.compute(*task)
    t1 = time.perf_counter()
    print(
        f"Execution  for dask took: {t1 - t0:.6f} seconds with total: {4 * sum(results) / total}"
    )

    client.close()
    cluster.close()

    return t1 - t0, 4 * sum(results) / total


def mandelbrot_dask_worker(
    N: int,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    max_iter: int = 100,
    chunks: int = 32,
) -> tuple[float, npt.NDArray[np.int32]]:
    """
    Compute Mandelbrot rows in parallel with Dask delayed tasks.

    Args:
        N (int): Output grid size along each axis.
        x_min (float): Minimum real-axis value.
        x_max (float): Maximum real-axis value.
        y_min (float): Minimum imaginary-axis value.
        y_max (float): Maximum imaginary-axis value.
        max_iter (int): Maximum iteration count per point.
        chunks (int): Number of row chunks to schedule.

    Returns:
        tuple[float, np.ndarray]: Execution time and Mandelbrot array.
    """
    chunk_size = max(1, N // chunks)
    tasks, row = [], 0
    while row < N:
        row_end = min(row + chunk_size, N)
        tasks.append(
            delayed(compute_mandelbrot_chunk)(
                row, row_end, N, x_min, x_max, y_min, y_max, max_iter
            )
        )
        row = row_end
    t0 = time.perf_counter()
    parts = dask.compute(*tasks)
    results = np.vstack(parts)
    t = time.perf_counter() - t0
    return t, results


if __name__ == "__main__":
    # total = 1_000_000_000
    # chunksize = 8
    # serial_time, serial_pi = monte_carlo_chunk(total, chunksize)
    # # Warmup
    # monte_carlo_dask(total, chunksize)
    # dask_time, dask_pi = monte_carlo_dask(total, chunksize)

    # dask_client_time, dask_client_pi = monte_carlo_dask_client(total, chunksize, 8, 1)

    N, max_iter = 1024, 100
    X_MIN, X_MAX, Y_MIN, Y_MAX = -2.5, 1.0, -1.25, 1.25
    client, cluster = load_dask_client_local(8, 1)
    client.run(
        lambda: compute_mandelbrot_chunk(0, 8, 8, X_MIN, X_MAX, Y_MIN, Y_MAX, 10)
    )

    times = []
    for _ in range(3):
        timing, results = mandelbrot_dask_worker(
            N, X_MIN, X_MAX, Y_MIN, Y_MAX, max_iter
        )
        times.append(timing)
    print(f"median time for dask worker: {np.median(times)}")

    client.close()
    cluster.close()
