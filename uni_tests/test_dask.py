import numpy as np
import pytest
from numpy.testing import assert_array_equal

from distributed_helpers import load_dask_client_local, mandelbrot_dask_worker
from main import w6_main
from multiprocessing_helpers import compute_mandelbrot_chunk, mandelbrot_serial
from uni_tests.conftest import DEFAULT_X_SET, DEFAULT_Y_SET


def test_mandelbrot_dask_worker_matches_serial_small_grid() -> None:
    expected = mandelbrot_serial(
        5,
        DEFAULT_X_SET[0],
        DEFAULT_X_SET[1],
        DEFAULT_Y_SET[0],
        DEFAULT_Y_SET[1],
        20,
    )
    elapsed, actual = mandelbrot_dask_worker(
        5,
        DEFAULT_X_SET[0],
        DEFAULT_X_SET[1],
        DEFAULT_Y_SET[0],
        DEFAULT_Y_SET[1],
        20,
        chunks=3,
    )
    assert elapsed >= 0.0
    assert_array_equal(actual, expected)


@pytest.mark.integration
def test_local_dask_submit_gather_matches_expected_chunk() -> None:
    client, cluster = load_dask_client_local(workers=1, threads_per_worker=1)
    try:
        future = client.submit(
            compute_mandelbrot_chunk,
            0,
            4,
            4,
            DEFAULT_X_SET[0],
            DEFAULT_X_SET[1],
            DEFAULT_Y_SET[0],
            DEFAULT_Y_SET[1],
            20,
        )
        gathered = client.gather(future)
    finally:
        client.close()
        cluster.close()

    expected = mandelbrot_serial(
        4,
        DEFAULT_X_SET[0],
        DEFAULT_X_SET[1],
        DEFAULT_Y_SET[0],
        DEFAULT_Y_SET[1],
        20,
    )
    assert_array_equal(gathered, expected)


def test_w6_main_returns_summary_and_closes_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected = np.arange(9, dtype=np.int32).reshape(3, 3)
    call_log = {"runs": 0}

    class FakeClient:
        def __init__(self) -> None:
            self.closed = False
            self.run_args = []

        def run(self, callback):
            self.run_args.append(callback)
            return None

        def close(self) -> None:
            self.closed = True

    class FakeCluster:
        def __init__(self) -> None:
            self.closed = False

        def close(self) -> None:
            self.closed = True

    fake_client = FakeClient()
    fake_cluster = FakeCluster()

    def fake_load_client_local(processes: int, threads_per_worker: int):
        assert processes == 2
        assert threads_per_worker == 1
        return fake_client, fake_cluster

    def fake_dask_worker(*args, **kwargs):
        call_log["runs"] += 1
        return 0.2 * call_log["runs"], expected

    monkeypatch.setattr("main.load_dask_client_local", fake_load_client_local)
    monkeypatch.setattr("main.mandelbrot_dask_worker", fake_dask_worker)
    monkeypatch.setattr("main.get_mandelbrot_serial_time", lambda *args, **kwargs: 2.0)

    median, result, variance, lif = w6_main(
        max_iters=20,
        x_set=DEFAULT_X_SET,
        y_set=DEFAULT_Y_SET,
        win_size=3,
        NoProcesses=2,
        n_runs=3,
        chunks=4,
    )

    assert len(fake_client.run_args) == 1
    assert fake_client.closed is True
    assert fake_cluster.closed is True
    assert median == pytest.approx(0.4)
    assert variance == pytest.approx(np.var([0.2, 0.4, 0.6]))
    assert lif == pytest.approx(((2 * 0.4) / 2.0) - 1)
    assert_array_equal(result, expected)
