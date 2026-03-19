import dask, random, time, statistics
from dask import delayed
from dask.distributed import Client, LocalCluster
from multiprocessing_helpers import estimate_pi_chunk, estimate_pi_circle, timing

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


def load_dask_client(workers: int = 4, threads_per_worker: int = 1):
    cluster = LocalCluster(n_workers=workers, threads_per_worker=threads_per_worker)
    client = Client(cluster)
    print(f"Dashboard: {client.dashboard_link}")
    return client, cluster

def monte_carlo_dask_client(total: int = 1_000_000, chunksize: int = 8, workers: int = 4, threads_per_worker: int = 1):
    client, cluster = load_dask_client(workers, threads_per_worker)
    samples = total // chunksize
    task = [delayed(estimate_pi_chunk)(samples) for _ in range(chunksize)]    
    t0 = time.perf_counter()
    results = dask.compute(*task)
    t1 = time.perf_counter()
    print(f'Execution  for dask took: {t1 - t0:.6f} seconds with total: {4*sum(results)/total}')

    client.close()
    cluster.close()

    return t1 - t0, 4 * sum(results) / total

if __name__ == "__main__":
    total = 1_000_000
    chunksize = 8
    serial_time, serial_pi = monte_carlo_chunk(total, chunksize)
    # Warmup
    monte_carlo_dask(total, chunksize)
    dask_time, dask_pi = monte_carlo_dask(total, chunksize)

    dask_client_time, dask_client_pi = monte_carlo_dask_client(total, chunksize, 8, 1)

   