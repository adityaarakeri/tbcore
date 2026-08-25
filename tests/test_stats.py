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
