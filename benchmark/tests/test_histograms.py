"""Unit tests for benchmark/common.py histogram computation."""

from common import compute_histograms


def _detail(audio_sec: float, ref_words: int, errors: int, failed: bool = False) -> dict:
    wer = (errors / ref_words * 100.0) if ref_words > 0 else 0.0
    return {
        "audio_sec": audio_sec,
        "ref_words": ref_words,
        "errors": errors,
        "wer": wer,
        "failed": failed,
    }


def test_histogram_by_audio_duration():
    details = [
        _detail(2.0, 5, 0),
        _detail(8.0, 10, 1),
        _detail(20.0, 15, 3),
        _detail(45.0, 20, 5),
    ]
    hists = compute_histograms(details)
    buckets = {b["bucket"]: b for b in hists["audio_duration"]}

    assert buckets["0-5s"]["samples"] == 1
    assert buckets["5-15s"]["samples"] == 1
    assert buckets["15-30s"]["samples"] == 1
    assert buckets["30s+"]["samples"] == 1

    assert buckets["0-5s"]["ref_words"] == 5
    assert buckets["0-5s"]["errors"] == 0
    assert buckets["0-5s"]["wer"] == 0.0

    assert buckets["30s+"]["ref_words"] == 20
    assert buckets["30s+"]["errors"] == 5
    assert buckets["30s+"]["wer"] == 25.0


def test_histogram_by_ref_words():
    details = [
        _detail(2.0, 3, 0),
        _detail(8.0, 8, 1),
        _detail(20.0, 20, 2),
        _detail(45.0, 40, 10),
    ]
    hists = compute_histograms(details)
    buckets = {b["bucket"]: b for b in hists["ref_words"]}

    assert buckets["1-5 words"]["samples"] == 1
    assert buckets["6-15 words"]["samples"] == 1
    assert buckets["16-30 words"]["samples"] == 1
    assert buckets["30+ words"]["samples"] == 1


def test_histogram_by_wer():
    details = [
        _detail(2.0, 10, 0),    # 0%
        _detail(3.0, 10, 1),    # 10%
        _detail(4.0, 10, 2),    # 20%
        _detail(5.0, 10, 4),    # 40%
        _detail(6.0, 10, 7),    # 70%
        _detail(7.0, 10, 10),   # 100%
        _detail(8.0, 10, 12),   # 120%
    ]
    hists = compute_histograms(details)
    buckets = {b["bucket"]: b for b in hists["wer"]}

    # Bucket upper bounds are exclusive, so boundary values fall into the next
    # bucket: 10% → 10-20%, 20% → 20-50%, 100% → 100%+.
    assert buckets["0%"]["samples"] == 1
    assert buckets["1-10%"]["samples"] == 0
    assert buckets["10-20%"]["samples"] == 1
    assert buckets["20-50%"]["samples"] == 2
    assert buckets["50-100%"]["samples"] == 1
    assert buckets["100%+"]["samples"] == 2


def test_histogram_bucket_boundaries():
    # Exact boundary values should land in the expected bucket.
    details = [
        _detail(5.0, 10, 1),   # 5s is low-inclusive for 5-15s
        _detail(15.0, 10, 1),  # 15s is low-inclusive for 15-30s
        _detail(30.0, 10, 1),  # 30s is low-inclusive for 30s+
    ]
    hists = compute_histograms(details)
    buckets = {b["bucket"]: b for b in hists["audio_duration"]}

    assert buckets["0-5s"]["samples"] == 0
    assert buckets["5-15s"]["samples"] == 1
    assert buckets["15-30s"]["samples"] == 1
    assert buckets["30s+"]["samples"] == 1


def test_histogram_ref_words_zero_goes_to_first_bucket():
    # Values below the first bucket's lower bound currently fall into the first
    # bucket because _bucket_value returns the last bucket label only after all
    # buckets have been tested.
    details = [_detail(1.0, 0, 0)]
    hists = compute_histograms(details)
    buckets = {b["bucket"]: b for b in hists["ref_words"]}
    assert buckets["1-5 words"]["samples"] == 1


def test_histogram_empty_details():
    hists = compute_histograms([])
    for dim in ("audio_duration", "ref_words", "wer"):
        buckets = hists[dim]
        assert all(b["samples"] == 0 for b in buckets)
        assert all(b["wer"] == 0.0 for b in buckets)


def test_histogram_includes_failed_samples():
    # Failed samples report 100% WER for that sample.
    details = [
        _detail(2.0, 5, 0),
        {**_detail(2.0, 5, 0), "errors": 5, "wer": 100.0, "failed": True},
    ]
    hists = compute_histograms(details)
    wer_buckets = {b["bucket"]: b for b in hists["wer"]}

    assert wer_buckets["0%"]["samples"] == 1
    assert wer_buckets["100%+"]["samples"] == 1
