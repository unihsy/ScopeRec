import numpy as np
import pyarrow as pa
import pyarrow.parquet as parquet
import pytest

from scoperec.protocol import build_protocol, classify_items, earlier_histories, timestamp_ms


def test_histories_exclude_equal_timestamps_and_future():
    assert earlier_histories([(1, 1), (2, 2), (2, 3), (3, 4), (4, 5)], 2) == [
        [], [1], [1], [2, 3], [3, 4],
    ]


def test_time_boundaries_are_exclusive():
    assert classify_items(np.array([9, 10, 19, 20, 29, 30]), 10, 20, 30).tolist() == [
        0, 1, 1, 2, 2, -1,
    ]
    with pytest.raises(ValueError, match="timezone"):
        timestamp_ms("2018-01-01")


def test_real_protocol_preserves_time_catalog_and_no_history_rules(tmp_path):
    base = timestamp_ms("2018-01-01T00:00:00+00:00")
    start = base + 86400000
    edit = base + 2 * 86400000
    end = base + 3 * 86400000
    rows = [
        {"user_id": "alice", "parent_asin": "old", "rating": 1.0, "timestamp": base - 3},
        {"user_id": "alice", "parent_asin": "peer", "rating": 5.0, "timestamp": base - 3},
        {"user_id": "alice", "parent_asin": "old2", "rating": 5.0, "timestamp": base - 1},
        {"user_id": "alice", "parent_asin": "new", "rating": 5.0, "timestamp": start},
        {"user_id": "bob", "parent_asin": "old", "rating": 5.0, "timestamp": base - 1},
        {"user_id": "bob", "parent_asin": "new", "rating": 5.0, "timestamp": edit},
        {"user_id": "bob", "parent_asin": "future", "rating": 5.0, "timestamp": edit + 1},
    ]
    parquet.write_table(pa.Table.from_pylist(rows), tmp_path / "interactions.parquet")
    metadata = [
        {"parent_asin": item, "title": item, "features": "", "description": ""}
        for item in ("old", "peer", "old2", "new", "future")
    ]
    parquet.write_table(pa.Table.from_pylist(metadata), tmp_path / "items.parquet")
    config = {
        "base_cutoff": "2018-01-01T00:00:00+00:00", "history_items": 2,
        "sampling_seed": 42, "max_train_requests": 50, "max_holdout_requests": 50,
        "max_eval_requests": 50, "adaptation": {"few_shot_per_item": 2},
        "windows": {"validation": {
            "arrival_start": "2018-01-02T00:00:00+00:00",
            "edit_at": "2018-01-03T00:00:00+00:00", "end": "2018-01-04T00:00:00+00:00",
        }},
    }
    report = build_protocol(tmp_path, tmp_path / "experiment", config)
    assert report["eligible_pre_base_requests"] == 1
    assert report["validation"]["requests"] == 1
    assert report["validation"]["requests_outside_catalog"] == 1
    assert report["validation_few_shot"]["requests"] == 0
    dataset = np.load(tmp_path / "experiment" / "validation.npz")
    assert dataset["group"].tolist() == [2]
    assert dataset["timestamp"].tolist() == [edit]
    assert int(np.sum(dataset["history"] >= 0)) == 1
    assert end > edit > start > base