"""Reusable statistics: distribution summaries, bootstrap CIs, regression tests."""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Integral, Real

import numpy as np
import numpy.typing as npt
from scipy import stats as sps

_MAX_RESAMPLE_ELEMENTS = 1_000_000


@dataclass(frozen=True)
class Summary:
    n: int
    mean: float
    stddev: float
    ci_low: float
    ci_high: float
    level: float


@dataclass(frozen=True)
class RegressionResult:
    regressed: bool
    p_value: float
    effect_size: float
    alpha: float
    baseline_mean: float
    candidate_mean: float


def _as_float_array(samples: npt.ArrayLike) -> np.ndarray:
    arr = np.asarray(samples, dtype=float).ravel()
    if arr.size == 0:
        raise ValueError("samples must be non-empty")
    return arr


def _validate_alpha(alpha: object) -> None:
    if (
        isinstance(alpha, bool)
        or not isinstance(alpha, Real)
        or not math.isfinite(alpha)
        or not 0 < float(alpha) < 1
    ):
        raise ValueError("alpha must be finite and strictly between 0 and 1")


def _validate_iterations(iterations: object) -> None:
    if isinstance(iterations, bool) or not isinstance(iterations, Integral) or iterations <= 0:
        raise ValueError("iterations must be a positive integer")


def bootstrap_ci(
    samples: npt.ArrayLike,
    alpha: float = 0.05,
    *,
    iterations: int = 2000,
    rng: np.random.Generator | None = None,
) -> tuple[float, float]:
    """Percentile bootstrap confidence interval for the mean."""
    _validate_alpha(alpha)
    _validate_iterations(iterations)
    arr = _as_float_array(samples)
    if arr.size < 2:
        raise ValueError("bootstrap CI needs at least two samples")
    if rng is None:
        rng = np.random.default_rng()
    means = np.empty(iterations, dtype=float)
    batch_size = max(1, _MAX_RESAMPLE_ELEMENTS // arr.size)
    for start in range(0, iterations, batch_size):
        count = min(batch_size, iterations - start)
        indexes = rng.integers(0, arr.size, size=(count, arr.size))
        means[start : start + count] = arr[indexes].mean(axis=1)
    low, high = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return float(low), float(high)


def summary(
    samples: npt.ArrayLike,
    alpha: float = 0.05,
    *,
    iterations: int = 2000,
    rng: np.random.Generator | None = None,
) -> Summary:
    """Mean, sample stddev (ddof=1), and a bootstrap CI for the mean.

    Pass a seeded ``rng`` for reproducible intervals. With a single sample the
    stddev is NaN and the CI collapses to the point value.
    """
    _validate_alpha(alpha)
    _validate_iterations(iterations)
    arr = _as_float_array(samples)
    n = int(arr.size)
    mean = float(arr.mean())
    stddev = float(arr.std(ddof=1)) if n > 1 else float("nan")
    if n > 1:
        ci_low, ci_high = bootstrap_ci(arr, alpha=alpha, iterations=iterations, rng=rng)
    else:
        ci_low = ci_high = mean
    return Summary(
        n=n,
        mean=mean,
        stddev=stddev,
        ci_low=ci_low,
        ci_high=ci_high,
        level=1 - alpha,
    )


def cohens_d(baseline: npt.ArrayLike, candidate: npt.ArrayLike) -> float:
    """Pooled-SD effect size; positive means the candidate is HIGHER.

    Returns NaN when it is undefined: fewer than three observations in total
    or zero pooled variance.
    """
    base = _as_float_array(baseline)
    cand = _as_float_array(candidate)
    n_base, n_cand = base.size, cand.size
    dof = n_base + n_cand - 2
    if dof <= 0:
        return float("nan")
    pooled_var = (
        (n_base - 1) * (base.var(ddof=1) if n_base > 1 else 0.0)
        + (n_cand - 1) * (cand.var(ddof=1) if n_cand > 1 else 0.0)
    ) / dof
    if pooled_var <= 0.0:
        return float("nan")
    return float((cand.mean() - base.mean()) / np.sqrt(pooled_var))


def regressed(
    baseline: npt.ArrayLike,
    candidate: npt.ArrayLike,
    alpha: float = 0.05,
    *,
    higher_is_better: bool = False,
) -> RegressionResult:
    """Test whether the candidate is WORSE than the baseline.

    By default "worse" means a higher candidate mean (metrics where lower is
    better, e.g. cost or latency). With ``higher_is_better=True`` a lower
    candidate mean is the regression (e.g. accuracy). Uses a one-sided Welch
    t-test plus Cohen's d. ``effect_size`` keeps one sign convention
    regardless of direction: positive means the candidate mean is higher.
    """
    _validate_alpha(alpha)
    base = _as_float_array(baseline)
    cand = _as_float_array(candidate)
    alternative = "less" if higher_is_better else "greater"
    test = sps.ttest_ind(cand, base, equal_var=False, alternative=alternative)
    return RegressionResult(
        regressed=bool(test.pvalue < alpha),
        p_value=float(test.pvalue),
        effect_size=cohens_d(base, cand),
        alpha=alpha,
        baseline_mean=float(base.mean()),
        candidate_mean=float(cand.mean()),
    )
