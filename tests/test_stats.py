import warnings

import numpy as np
import pytest

from tbcore.stats import bootstrap_ci, cohens_d, regressed, summary

SEED = 1234


def test_summary_known_values():
    result = summary([1, 2, 3, 4, 5])

    assert result.n == 5
    assert result.mean == pytest.approx(3.0)
    assert result.stddev == pytest.approx(1.5811388300841898)
    assert result.level == pytest.approx(0.95)


def test_summary_stddev_exact_for_simple_series():
    result = summary([2, 4, 6])

    assert result.mean == pytest.approx(4.0)
    assert result.stddev == pytest.approx(2.0)


def test_bootstrap_ci_is_seeded_and_deterministic():
    samples = [1.0, 2.0, 3.0, 4.0, 5.0]

    first = bootstrap_ci(samples, rng=np.random.default_rng(SEED))
    second = bootstrap_ci(samples, rng=np.random.default_rng(SEED))

    assert first == second
    low, high = first
    assert 1.0 < low < 3.0
    assert 3.0 < high < 5.0


def test_summary_ci_brackets_the_mean():
    rng = np.random.default_rng(SEED)
    samples = rng.normal(10.0, 2.0, size=200).tolist()

    result = summary(samples, rng=np.random.default_rng(SEED + 1))

    assert result.ci_low < result.mean < result.ci_high


def test_summary_single_sample_collapses_to_point():
    result = summary([42.0])

    assert result.n == 1
    assert result.mean == 42.0
    assert np.isnan(result.stddev)
    assert result.ci_low == result.ci_high == 42.0


def test_regressed_flags_synthetic_worse_candidate():
    rng = np.random.default_rng(SEED)
    baseline = rng.normal(0.0, 1.0, size=200)
    candidate = rng.normal(1.0, 1.0, size=200)

    result = regressed(baseline, candidate, alpha=0.05)

    assert result.regressed is True
    assert result.p_value < 1e-6
    assert result.effect_size == pytest.approx(1.0, abs=0.35)
    assert result.candidate_mean > result.baseline_mean


def test_regressed_clears_synthetic_no_change_candidate():
    rng = np.random.default_rng(SEED)
    baseline = rng.normal(0.0, 1.0, size=200)
    candidate = rng.normal(0.0, 1.0, size=200)

    result = regressed(baseline, candidate, alpha=0.05)

    assert result.regressed is False
    assert result.p_value > 0.05


def test_regressed_does_not_flag_improvement():
    rng = np.random.default_rng(SEED)
    baseline = rng.normal(0.0, 1.0, size=200)
    candidate = rng.normal(-1.0, 1.0, size=200)

    result = regressed(baseline, candidate)

    assert result.regressed is False
    assert result.effect_size < 0


def test_cohens_d_known_value():
    d = cohens_d([-1.0, 1.0], [0.0, 2.0])

    assert d == pytest.approx(1.0 / np.sqrt(2.0))


def test_empty_samples_raise():
    with pytest.raises(ValueError):
        summary([])
    with pytest.raises(ValueError):
        regressed([], [1.0])


def test_regressed_higher_is_better_flags_lower_candidate():
    rng = np.random.default_rng(SEED)
    baseline = rng.normal(1.0, 1.0, size=200)
    candidate = rng.normal(0.0, 1.0, size=200)

    default = regressed(baseline, candidate)
    result = regressed(baseline, candidate, higher_is_better=True)

    assert default.regressed is False
    assert result.regressed is True
    assert result.p_value < 1e-6
    assert result.effect_size < 0  # sign convention: negative means candidate is lower


def test_regressed_higher_is_better_does_not_flag_higher_candidate():
    rng = np.random.default_rng(SEED)
    baseline = rng.normal(0.0, 1.0, size=200)
    candidate = rng.normal(1.0, 1.0, size=200)

    assert regressed(baseline, candidate, higher_is_better=True).regressed is False
    assert regressed(baseline, candidate).regressed is True


def test_cohens_d_is_nan_for_too_little_data():
    assert np.isnan(cohens_d([1.0], [2.0]))


def test_cohens_d_is_nan_for_zero_pooled_variance():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert np.isnan(cohens_d([1.0, 1.0, 1.0], [2.0, 2.0, 2.0]))
        assert np.isnan(cohens_d([3.0, 3.0], [3.0, 3.0]))


def test_cohens_d_with_one_singleton_group_is_finite():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert np.isfinite(cohens_d([1.0], [2.0, 4.0]))


def test_stats_helpers_are_exported():
    import tbcore

    assert tbcore.bootstrap_ci is bootstrap_ci
    assert tbcore.cohens_d is cohens_d


@pytest.mark.parametrize("alpha", [0, 1, -0.1, 1.1, float("nan"), float("inf"), -float("inf")])
@pytest.mark.parametrize(
    "call",
    [
        lambda alpha: bootstrap_ci([1, 2], alpha=alpha),
        lambda alpha: summary([1, 2], alpha=alpha),
        lambda alpha: summary([1], alpha=alpha),
        lambda alpha: regressed([1, 2], [2, 3], alpha=alpha),
    ],
)
def test_invalid_alpha_raises_at_public_entry_points(alpha, call):
    with pytest.raises(ValueError, match="alpha must be finite and strictly between 0 and 1"):
        call(alpha)


@pytest.mark.parametrize("iterations", [0, -1, 1.5, float("nan"), float("inf"), True, False, "3"])
@pytest.mark.parametrize(
    "call",
    [
        lambda iterations: bootstrap_ci([1, 2], iterations=iterations),
        lambda iterations: summary([1, 2], iterations=iterations),
        lambda iterations: summary([1], iterations=iterations),
    ],
)
def test_invalid_iterations_raise_at_public_entry_points(iterations, call):
    with pytest.raises(ValueError, match="iterations must be a positive integer"):
        call(iterations)


def test_numpy_integer_iterations_are_accepted():
    result = bootstrap_ci([1, 2], iterations=np.int64(10), rng=np.random.default_rng(SEED))

    assert result[0] <= result[1]


def test_bootstrap_ci_bounds_resampling_allocation():
    class GuardedRng:
        def __init__(self):
            self.generator = np.random.default_rng(SEED)
            self.calls = 0

        def integers(self, low, high, *, size):
            assert size[0] * size[1] <= 1_000_000
            self.calls += 1
            return self.generator.integers(low, high, size=size)

    rng = GuardedRng()
    low, high = bootstrap_ci(np.arange(10_000), iterations=200, rng=rng)

    assert rng.calls > 1
    assert low <= high


def test_batched_bootstrap_matches_seeded_bulk_reference():
    samples = np.arange(1001, dtype=float)
    iterations = 1000
    indexes = np.random.default_rng(SEED).integers(0, samples.size, size=(iterations, samples.size))
    expected = tuple(np.quantile(samples[indexes].mean(axis=1), [0.025, 0.975]))

    assert bootstrap_ci(samples, iterations=iterations, rng=np.random.default_rng(SEED)) == expected
