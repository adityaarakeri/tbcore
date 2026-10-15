"""The benchmark reports distributions without hard timing thresholds."""

from scripts.benchmark_experiment import _stream_latency, summarize


def test_summary_preserves_all_samples_and_reports_distribution():
    result = summarize([4.0, 1.0, 2.0, 3.0])
    assert result["count"] == 4
    assert result["samples"] == [4.0, 1.0, 2.0, 3.0]
    assert result["median"] == 2.5
    assert result["p95"] == 4.0
    assert summarize([])["median"] is None


def test_stream_recipe_measures_direct_and_proxied_first_byte():
    result = _stream_latency(2)
    assert result["direct_first_byte_s"]["count"] == 2
    assert result["proxy_first_byte_s"]["count"] == 2
