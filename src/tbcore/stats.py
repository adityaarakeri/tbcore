"""Reusable statistics: distribution summaries, bootstrap CIs, regression tests."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy import stats as sps


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


def _as_float_array(samples: Sequence[float]) -> np.ndarray:
    arr = np.asarray(samples, dtype=float).ravel()
    if arr.size == 0:
        raise ValueError("samples must be non-empty")
    return arr


def bootstrap_ci(
    samples: Sequence[float],
    alpha: float = 0.05,
    *,
    iterations: int = 2000,
    rng: np.random.Generator | None = None,
) -> tuple[float, float]:
    """Percentile bootstrap confidence interval for the mean."""
    arr = _as_float_array(samples)
    if arr.size < 2:
        raise ValueError("bootstrap CI needs at least two samples")
    if rng is None:
        rng = np.random.default_rng()
    indexes = rng.integers(0, arr.size, size=(iterations, arr.size))
    means = arr[indexes].mean(axis=1)
    low, high = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return float(low), float(high)


def summary(
    samples: Sequence[float],
    alpha: float = 0.05,
    *,
    iterations: int = 2000,
    rng: np.random.Generator | None = None,
) -> Summary:
    """Mean, sample stddev (ddof=1), and a bootstrap CI for the mean.

    Pass a seeded ``rng`` for reproducible intervals. With a single sample the
    stddev is NaN and the CI collapses to the point value.
    """
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


def cohens_d(baseline: Sequence[float], candidate: Sequence[float]) -> float:
    """Pooled-SD effect size; positive means the candidate is HIGHER."""
    base = _as_float_array(baseline)
    cand = _as_float_array(candidate)
    n_base, n_cand = base.size, cand.size
    pooled_var = ((n_base - 1) * base.var(ddof=1) + (n_cand - 1) * cand.var(ddof=1)) / (
        n_base + n_cand - 2
    )
    return float((cand.mean() - base.mean()) / np.sqrt(pooled_var))


def regressed(
    baseline: Sequence[float], candidate: Sequence[float], alpha: float = 0.05
) -> RegressionResult:
    """Test whether the candidate is WORSE than the baseline.

    "Worse" means a higher candidate mean (use this layer for metrics where
    lower is better, e.g. cost or latency). Uses a one-sided Welch t-test plus
    Cohen's d as the effect size.
    """
    base = _as_float_array(baseline)
    cand = _as_float_array(candidate)
    test = sps.ttest_ind(cand, base, equal_var=False, alternative="greater")
    return RegressionResult(
        regressed=bool(test.pvalue < alpha),
        p_value=float(test.pvalue),
        effect_size=cohens_d(base, cand),
        alpha=alpha,
        baseline_mean=float(base.mean()),
        candidate_mean=float(cand.mean()),
    )
