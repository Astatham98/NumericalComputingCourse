import numpy as np
import pytest
from numpy.testing import assert_array_equal

from main import (
    benchmark,
    f_jit,
    mandelbrot_calc,
    w1_f,
    w1_main,
    w2_f,
    w2_main,
    w3_f,
    w3_main,
    w3_parallel,
    w3_parallel_main,
    w_1_5_main,
)
from uni_tests.conftest import (
    DEFAULT_X_SET,
    DEFAULT_Y_SET,
    SHIFTED_X_SET,
    SHIFTED_Y_SET,
    complex_grid,
    reference_escape_bounded,
    reference_escape_post_update,
    reference_grid_bounded,
    reference_grid_post_update,
)


@pytest.mark.parametrize(
    ("point", "max_iters"),
    [
        (0j, 0),
        (0j, 5),
        (2 + 2j, 1),
        (2 + 2j, 20),
        (-1 + 0j, 20),
        (-0.75 + 0.1j, 50),
    ],
)
def test_f_jit_matches_python_reference(point: complex, max_iters: int) -> None:
    expected = reference_escape_post_update(point, max_iters)
    assert f_jit.py_func(point, max_iters) == expected
    assert f_jit(point, max_iters) == expected


@pytest.mark.parametrize(
    ("point", "max_iters"),
    [
        (0j, 0),
        (0j, 5),
        (2 + 2j, 1),
        (2 + 2j, 20),
        (-1 + 0j, 20),
        (-0.75 + 0.1j, 50),
    ],
)
def test_mandelbrot_calc_matches_python_reference(
    point: complex, max_iters: int
) -> None:
    expected = reference_escape_bounded(point, max_iters)
    assert mandelbrot_calc.py_func(point, max_iters) == expected
    assert mandelbrot_calc(point, max_iters) == expected


@pytest.mark.parametrize("max_iters", [0, 1, 5, 20])
def test_w1_f_matches_reference_grid(max_iters: int) -> None:
    points = np.array([complex(x, y) for x in (-2.0, -0.5, 0.5) for y in (-1.0, 0.0)])
    expected = np.array(
        [reference_escape_post_update(point, max_iters) for point in points],
        dtype=np.int32,
    )
    actual = w1_f(points, np.zeros(len(points), dtype=np.int32), max_iters)
    assert_array_equal(actual, expected)


@pytest.mark.parametrize("max_iters", [0, 1, 5, 20])
def test_w2_f_matches_reference_grid(max_iters: int) -> None:
    grid = complex_grid(4, DEFAULT_X_SET, DEFAULT_Y_SET)
    actual = w2_f(
        grid.copy(),
        np.zeros_like(grid, dtype=complex),
        np.zeros(grid.shape, dtype=np.int32),
        max_iters,
    )
    expected = reference_grid_bounded(4, max_iters)
    assert_array_equal(actual, expected)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("max_iters", [0, 1, 5, 20])
def test_w3_f_matches_python_and_compiled_versions(
    dtype: np.dtype, max_iters: int
) -> None:
    grid = complex_grid(4, DEFAULT_X_SET, DEFAULT_Y_SET, dtype=dtype)
    expected = reference_grid_post_update(4, max_iters, dtype=dtype)
    python_version = w3_f.__wrapped__.py_func(
        grid.copy(), np.zeros(grid.shape, dtype=np.int32), max_iters
    )
    compiled_version = w3_f(
        grid.copy(), np.zeros(grid.shape, dtype=np.int32), max_iters
    )
    assert_array_equal(python_version, expected)
    assert_array_equal(compiled_version, expected)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("max_iters", [0, 1, 5, 20])
def test_w3_parallel_matches_python_and_compiled_versions(
    dtype: np.dtype, max_iters: int
) -> None:
    grid = complex_grid(4, DEFAULT_X_SET, DEFAULT_Y_SET, dtype=dtype)
    expected = reference_grid_bounded(4, max_iters, dtype=dtype)
    python_version = w3_parallel.__wrapped__.py_func(
        grid.copy(), np.zeros(grid.shape, dtype=np.int32), 4, max_iters
    )
    compiled_version = w3_parallel(
        grid.copy(), np.zeros(grid.shape, dtype=np.int32), 4, max_iters
    )
    assert_array_equal(python_version, expected)
    assert_array_equal(compiled_version, expected)


@pytest.mark.parametrize(
    ("func", "reference_builder"),
    [
        (w1_main, reference_grid_post_update),
        (w2_main, reference_grid_bounded),
        (w_1_5_main, reference_grid_post_update),
        (w3_main, reference_grid_post_update),
        (w3_parallel_main, reference_grid_bounded),
    ],
)
@pytest.mark.parametrize("win_size", [1, 2, 5])
@pytest.mark.parametrize("max_iters", [0, 1, 5])
def test_week_mains_match_expected_reference(
    func, reference_builder, win_size: int, max_iters: int
) -> None:
    actual = func(max_iters, DEFAULT_X_SET, DEFAULT_Y_SET, win_size)
    expected = reference_builder(win_size, max_iters)
    assert actual.shape == (win_size, win_size)
    assert_array_equal(actual, expected)


@pytest.mark.parametrize(
    "func",
    [w1_main, w2_main, w_1_5_main, w3_main, w3_parallel_main],
)
def test_week_mains_respect_custom_axes(func) -> None:
    shifted = func(20, SHIFTED_X_SET, SHIFTED_Y_SET, 5)
    default = func(20, DEFAULT_X_SET, DEFAULT_Y_SET, 5)
    assert not np.array_equal(shifted, default)


def test_w1_main_matches_shifted_reference() -> None:
    actual = w1_main(20, SHIFTED_X_SET, SHIFTED_Y_SET, 5)
    expected = reference_grid_post_update(5, 20, SHIFTED_X_SET, SHIFTED_Y_SET)
    assert_array_equal(actual, expected)


def test_benchmark_returns_last_result_and_statistics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    perf_values = iter([0.0, 0.2, 1.0, 1.3, 2.0, 2.5])

    def fake_perf_counter() -> float:
        return next(perf_values)

    call_results = iter([10, 20, 30])

    def sample(value: int) -> int:
        return value + next(call_results)

    monkeypatch.setattr("main.time.perf_counter", fake_perf_counter)

    median, result, variance = benchmark(sample, 5, n_runs=3)

    assert median == pytest.approx(0.3)
    assert result == 35
    assert variance == pytest.approx(np.var([0.2, 0.3, 0.5]))
