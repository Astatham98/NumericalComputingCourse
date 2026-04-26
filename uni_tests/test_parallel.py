import numpy as np
import pytest
from numpy.testing import assert_array_equal

from main import w4_main, w5_main
from helper_funcs.multiprocessing_helpers import (
    _worker,
    compute_mandelbrot_chunk,
    mandelbrot_parallel,
    mandelbrot_parallel_chunks,
    mandelbrot_pixel,
    mandelbrot_serial,
)
from uni_tests.conftest import DEFAULT_X_SET, DEFAULT_Y_SET, reference_escape_bounded


@pytest.mark.parametrize(
    ("c_real", "c_imag", "max_iters"),
    [
        (0.0, 0.0, 0),
        (0.0, 0.0, 5),
        (2.0, 2.0, 1),
        (2.0, 2.0, 20),
        (-1.0, 0.0, 20),
        (-0.75, 0.1, 50),
    ],
)
def test_mandelbrot_pixel_matches_python_reference(
    c_real: float, c_imag: float, max_iters: int
) -> None:
    expected = reference_escape_bounded(complex(c_real, c_imag), max_iters)
    assert mandelbrot_pixel.py_func(c_real, c_imag, max_iters) == expected
    assert mandelbrot_pixel(c_real, c_imag, max_iters) == expected


@pytest.mark.parametrize("row_bounds", [(0, 2), (2, 5)])
def test_compute_mandelbrot_chunk_matches_serial_rows(
    row_bounds: tuple[int, int],
) -> None:
    start_row, end_row = row_bounds
    expected = mandelbrot_serial(
        5,
        DEFAULT_X_SET[0],
        DEFAULT_X_SET[1],
        DEFAULT_Y_SET[0],
        DEFAULT_Y_SET[1],
        20,
    )[start_row:end_row]
    actual = compute_mandelbrot_chunk(
        start_row,
        end_row,
        5,
        DEFAULT_X_SET[0],
        DEFAULT_X_SET[1],
        DEFAULT_Y_SET[0],
        DEFAULT_Y_SET[1],
        20,
    )
    assert_array_equal(actual, expected)


def test_worker_unpacks_compute_chunk_arguments() -> None:
    args = (
        1,
        4,
        5,
        DEFAULT_X_SET[0],
        DEFAULT_X_SET[1],
        DEFAULT_Y_SET[0],
        DEFAULT_Y_SET[1],
        20,
    )
    assert_array_equal(_worker(args), compute_mandelbrot_chunk(*args))


@pytest.mark.integration
def test_mandelbrot_parallel_matches_serial_small_grid() -> None:
    expected = mandelbrot_serial(
        4,
        DEFAULT_X_SET[0],
        DEFAULT_X_SET[1],
        DEFAULT_Y_SET[0],
        DEFAULT_Y_SET[1],
        20,
    )
    result, timings = mandelbrot_parallel(
        4,
        DEFAULT_X_SET[0],
        DEFAULT_X_SET[1],
        DEFAULT_Y_SET[0],
        DEFAULT_Y_SET[1],
        20,
        num_processes=2,
        NoRuns=1,
    )
    assert len(timings) == 1
    assert_array_equal(result, expected)


@pytest.mark.integration
def test_mandelbrot_parallel_chunks_matches_serial_small_grid() -> None:
    expected = mandelbrot_serial(
        5,
        DEFAULT_X_SET[0],
        DEFAULT_X_SET[1],
        DEFAULT_Y_SET[0],
        DEFAULT_Y_SET[1],
        20,
    )
    result, elapsed = mandelbrot_parallel_chunks(
        5,
        DEFAULT_X_SET[0],
        DEFAULT_X_SET[1],
        DEFAULT_Y_SET[0],
        DEFAULT_Y_SET[1],
        20,
        n_workers=2,
        n_chunks=3,
    )
    assert elapsed >= 0.0
    assert_array_equal(result, expected)


def test_w4_main_returns_summary_from_parallel_helper(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected = np.arange(9, dtype=np.int32).reshape(3, 3)
    recorded = {}

    def fake_parallel(
        win_size: int,
        x_min: float,
        x_max: float,
        y_min: float,
        y_max: float,
        max_iters: int,
        no_cores: int,
        n_runs: int,
    ):
        recorded["args"] = (
            win_size,
            x_min,
            x_max,
            y_min,
            y_max,
            max_iters,
            no_cores,
            n_runs,
        )
        return expected, [0.3, 0.1, 0.2]

    monkeypatch.setattr("main.mandelbrot_parallel", fake_parallel)

    median, result, variance = w4_main(
        max_iters=20,
        x_set=DEFAULT_X_SET,
        y_set=DEFAULT_Y_SET,
        win_size=3,
        NoCores=2,
        n_runs=3,
    )

    assert recorded["args"] == (3, -2.0, 1.0, -1.5, 1.5, 20, 2, 3)
    assert median == pytest.approx(0.2)
    assert variance == pytest.approx(np.var([0.3, 0.1, 0.2]))
    assert_array_equal(result, expected)


def test_w5_main_returns_summary_and_lif(monkeypatch: pytest.MonkeyPatch) -> None:
    expected = np.arange(16, dtype=np.int32).reshape(4, 4)
    recorded = {"calls": []}

    def fake_get_pool(processes: int, grid):
        recorded["pool"] = (processes, grid)
        return object()

    def fake_benchmark(*args, **kwargs):
        recorded["benchmark"] = (args, kwargs)
        return 2.0, expected, 0.0

    def fake_parallel_chunks(
        win_size: int,
        x_min: float,
        x_max: float,
        y_min: float,
        y_max: float,
        max_iters: int,
        processers: int,
        n_chunks: int,
        pool,
    ):
        recorded["calls"].append((win_size, processers, n_chunks, pool))
        return expected, 0.5 + 0.1 * len(recorded["calls"])

    monkeypatch.setattr("main.get_pool", fake_get_pool)
    monkeypatch.setattr("main.benchmark", fake_benchmark)
    monkeypatch.setattr("main.mandelbrot_parallel_chunks", fake_parallel_chunks)

    median, result, variance, lif = w5_main(
        max_iters=20,
        x_set=DEFAULT_X_SET,
        y_set=DEFAULT_Y_SET,
        win_size=4,
        NoProcesses=2,
        n_runs=3,
        chunk=4,
    )

    assert recorded["pool"][0] == 8
    assert recorded["calls"] == [
        (4, 2, 8, recorded["calls"][0][3]),
        (4, 2, 8, recorded["calls"][1][3]),
        (4, 2, 8, recorded["calls"][2][3]),
    ]
    assert median == pytest.approx(0.7)
    assert variance == pytest.approx(np.var([0.6, 0.7, 0.8]))
    assert lif == pytest.approx(((2 * 0.7) / 2.0) - 1)
    assert_array_equal(result, expected)
