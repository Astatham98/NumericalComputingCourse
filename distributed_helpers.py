import dask, random, time, statistics
from dask import delayed
import numpy as np
from dask.distributed import Client, LocalCluster
from multiprocessing_helpers import estimate_pi_chunk, timing, compute_mandelbrot_chunk

def monte_carlo_chunk(total: int = 1_000_000, chunksize: int = 8):
    samples = total // chunksize
    t0 = time.perf_counter()
    results =  [estimate_pi_chunk(samples) for _ in range(chunksize)]
    t1 = time.perf_counter()
    print(f'Execution for serial took: {t1 - t0:.6f} seconds with total: {4*sum(results)/total}')
    return t1 - t0, 4 * sum(results) / total

def monte_carlo_dask(total: int = 1_000_000, chunksize: int = 8):
    samples = total // chunksize
    task = [delayed(estimate_pi_chunk)(samples) for _ in range(chunksize)]    
    t0 = time.perf_counter()
    results = dask.compute(*task)
    t1 = time.perf_counter()
    print(f'Execution  for dask took: {t1 - t0:.6f} seconds with total: {4*sum(results)/total}')
    return t1 - t0, 4 * sum(results) / total


def load_dask_client_local(workers: int = 4, threads_per_worker: int = 1):
    cluster = LocalCluster(n_workers=workers, threads_per_worker=threads_per_worker)
    client = Client(cluster)
    print(f"Dashboard: {client.dashboard_link}")
    return client, cluster

def load_dask_client(ip):
    client = Client(f"tcp://{ip}:8786")
    return client

def monte_carlo_dask_client(total: int = 1_000_000, chunksize: int = 8, workers: int = 4, threads_per_worker: int = 1):
    client, cluster = load_dask_client_local(workers, threads_per_worker)
    samples = total // chunksize
    task = [delayed(estimate_pi_chunk)(samples) for _ in range(chunksize)]    
    t0 = time.perf_counter()
    results = dask.compute(*task)
    t1 = time.perf_counter()
    print(f'Execution  for dask took: {t1 - t0:.6f} seconds with total: {4*sum(results)/total}')

    client.close()
    cluster.close()

    return t1 - t0, 4 * sum(results) / total


def mandelbrot_dask_worker(N, x_min, x_max, y_min, y_max, max_iter=100, chunks=32):
    chunk_size = max(1, N // chunks)
    tasks, row = [], 0
    while row < N:
        row_end = min(row + chunk_size, N)
        tasks.append(delayed(compute_mandelbrot_chunk)(
            row, row_end, N, x_min, x_max, y_min, y_max, max_iter))
        row = row_end
    t0 = time.perf_counter()
    parts = dask.compute(*tasks)
    results = np.vstack(parts)
    timing = time.perf_counter() - t0
    return timing, results

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
    client.run(lambda: compute_mandelbrot_chunk(0, 8, 8, X_MIN, X_MAX, Y_MIN, Y_MAX, 10))

    times = []
    for _ in range(3):
        timing, results = mandelbrot_dask_worker(N, X_MIN, X_MAX, Y_MIN, Y_MAX, max_iter)
        times.append(timing)
    print(f'median time for dask worker: {np.median(times)}')

    client.close()
    cluster.close()
   