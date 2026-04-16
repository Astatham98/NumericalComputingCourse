import io
import statistics
import time
from contextlib import redirect_stderr, redirect_stdout
from typing import Any, Callable

import numpy as np
import pytest

from main import get_mandelbrot_serial_time, w1_main, w2_main, w3_main, w4_main, w5_main, w6_main, w_1_5_main
from uni_tests.conftest import DEFAULT_X_SET, DEFAULT_Y_SET


BENCHMARK_GRID_SIZE = 384
BENCHMARK_ITERS = 50
BENCHMARK_RUNS = 7
BENCHMARK_SMOKE_GRID_SIZE = 64


def timed_call(func: Callable[..., Any], *args: Any, **kwargs: Any) -> tuple[float, Any]:
    sink = io.StringIO()
    with redirect_stdout(sink), redirect_stderr(sink):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
    return elapsed, result


def median_runtime(
    func: Callable[..., Any],
    *args: Any,
    warmup: tuple[tuple[Any, ...], dict[str, Any]] | None = None,
    runs: int = BENCHMARK_RUNS,
    **kwargs: Any,
) -> float:
    if warmup is not None:
        warmup_args, warmup_kwargs = warmup
        timed_call(func, *warmup_args, **warmup_kwargs)

    durations = []
    for _ in range(runs):
        elapsed, _ = timed_call(func, *args, **kwargs)
        durations.append(elapsed)
    return statistics.median(durations)


@pytest.fixture(scope="module")
def naive_baseline_time() -> float:
    return median_runtime(
        w1_main,
        BENCHMARK_ITERS,
        DEFAULT_X_SET,
        DEFAULT_Y_SET,
        BENCHMARK_GRID_SIZE,
    )


@pytest.fixture(scope="module")
def numpy_vectorized_time() -> float:
    return median_runtime(
        w2_main,
        BENCHMARK_ITERS,
        DEFAULT_X_SET,
        DEFAULT_Y_SET,
        BENCHMARK_GRID_SIZE,
    )


@pytest.fixture(scope="module")
def numba_jit_time() -> float:
    return median_runtime(
        w_1_5_main,
        BENCHMARK_ITERS,
        DEFAULT_X_SET,
        DEFAULT_Y_SET,
        BENCHMARK_GRID_SIZE,
        warmup=((BENCHMARK_ITERS, DEFAULT_X_SET, DEFAULT_Y_SET, 16), {}),
    )


@pytest.fixture(scope="module")
def optimized_numba_time() -> float:
    return median_runtime(
        w3_main,
        BENCHMARK_ITERS,
        DEFAULT_X_SET,
        DEFAULT_Y_SET,
        BENCHMARK_GRID_SIZE,
        warmup=((BENCHMARK_ITERS, DEFAULT_X_SET, DEFAULT_Y_SET, 16), {}),
    )


@pytest.mark.performance
def test_w2_is_under_tenth_of_w1_runtime(
    naive_baseline_time: float, numpy_vectorized_time: float
) -> None:
    numpy_time = numpy_vectorized_time

    assert numpy_time < 0.1 * naive_baseline_time


@pytest.mark.performance
def test_w1_5_is_faster_than_w1_by_clear_margin(
    naive_baseline_time: float, numba_jit_time: float
) -> None:
    assert numba_jit_time < 0.2 * naive_baseline_time


@pytest.mark.performance
def test_w3_is_substantially_faster_than_w1(
    naive_baseline_time: float, optimized_numba_time: float
) -> None:
    assert optimized_numba_time < 0.05 * naive_baseline_time


@pytest.mark.performance
@pytest.mark.integration
@pytest.mark.parametrize(
    ("label", "func", "kwargs", "has_lif"),
    [
        (
            "w4",
            w4_main,
            {
                "max_iters": BENCHMARK_ITERS,
                "x_set": DEFAULT_X_SET,
                "y_set": DEFAULT_Y_SET,
                "win_size": BENCHMARK_SMOKE_GRID_SIZE,
                "NoCores": 2,
                "n_runs": 1,
            },
            False,
        ),
        (
            "w5",
            w5_main,
            {
                "max_iters": BENCHMARK_ITERS,
                "x_set": DEFAULT_X_SET,
                "y_set": DEFAULT_Y_SET,
                "win_size": BENCHMARK_SMOKE_GRID_SIZE,
                "NoProcesses": 2,
                "n_runs": 1,
                "chunk": 2,
            },
            True,
        ),
        (
            "w6",
            w6_main,
            {
                "max_iters": BENCHMARK_ITERS,
                "x_set": DEFAULT_X_SET,
                "y_set": DEFAULT_Y_SET,
                "win_size": BENCHMARK_SMOKE_GRID_SIZE,
                "NoProcesses": 2,
                "n_runs": 1,
                "chunks": 4,
            },
            True,
        ),
    ],
)
def test_week_4_to_6_benchmarks_return_finite_relative_timings(
    label: str,
    func: Callable[..., Any],
    kwargs: dict[str, Any],
    has_lif: bool,
    record_property: Callable[[str, object], None],
) -> None:
    serial_time, _ = timed_call(
        get_mandelbrot_serial_time,
        BENCHMARK_SMOKE_GRID_SIZE,
        DEFAULT_X_SET,
        DEFAULT_Y_SET,
        BENCHMARK_ITERS,
    )
    outer_time, payload = timed_call(func, **kwargs)

    if has_lif:
        returned_median, grid, variance, lif = payload
        assert np.isfinite(lif)
        record_property(f"{label}_lif", float(lif))
    else:
        returned_median, grid, variance = payload

    assert grid.shape == (BENCHMARK_SMOKE_GRID_SIZE, BENCHMARK_SMOKE_GRID_SIZE)
    assert np.isfinite(serial_time)
    assert serial_time > 0.0
    assert np.isfinite(returned_median)
    assert returned_median > 0.0
    assert np.isfinite(variance)
    assert variance >= 0.0
    assert outer_time >= returned_median

    record_property(f"{label}_wrapper_time", float(outer_time))
    record_property(f"{label}_returned_median", float(returned_median))
    record_property(f"{label}_wrapper_to_serial_ratio", float(outer_time / serial_time))