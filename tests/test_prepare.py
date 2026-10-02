import gzip
import json

import pyarrow.parquet as parquet
import pytest

from scoperec.prepare import (
    convert_interactions,
    convert_metadata,
    metadata_record,
    normalize_text,
    profile_dataset,
)


@pytest.fixture
def tiny_dataset(tmp_path):
    interaction_source = tmp_path / "Software.csv.gz"
    with gzip.open(interaction_source, "wt", encoding="utf-8") as stream:
        stream.write("user_id,parent_asin,rating,timestamp\n")
        stream.write("user-1,item-old,1.0,1609459200000\n")
        stream.write("user-1,item-new,5.0,1640995200001\n")
        stream.write("user-2,item-missing,3.0,1640995200002\n")
        stream.write("user-1,item-old,4.0,1640995200003\n")
    metadata_source = tmp_path / "meta_Software.jsonl.gz"
    records = [
        {
            "parent_asin": "item-old", "title": "  Old\n software ",
            "features": ["Local", " Fast "], "description": ["Useful tool"],
            "average_rating": 4.9, "rating_number": 10000, "price": 2.0,
        },
        {"parent_asin": "item-new", "title": "New software", "description": None},
        {"parent_asin": "item-without-events", "title": ""},
    ]
    with gzip.open(metadata_source, "wt", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record) + "\n")
    return interaction_source, metadata_source


def test_interactions_preserve_ratings_timestamps_and_repeats(tmp_path, tiny_dataset):
    source, _ = tiny_dataset
    destination = tmp_path / "interactions.parquet"
    assert convert_interactions(source, destination) == 4
    rows = parquet.read_table(destination).to_pylist()
    assert rows[0]["rating"] == 1.0
    assert rows[1]["timestamp"] == 1640995200001
    assert rows[0]["parent_asin"] == rows[3]["parent_asin"]
    assert not destination.with_name(destination.name + ".part").exists()


def test_metadata_retains_only_allowed_static_fields(tmp_path, tiny_dataset):
    _, source = tiny_dataset
    destination = tmp_path / "items.parquet"
    assert convert_metadata(source, destination, batch_size=1) == 3
    rows = parquet.read_table(destination).to_pylist()
    assert rows[0] == {
        "parent_asin": "item-old", "title": "Old software",
        "features": "Local Fast", "description": "Useful tool",
    }
    assert rows[1]["description"] == ""
    assert rows[2]["parent_asin"] == "item-without-events"


def test_profile_reports_missing_metadata_without_filtering(tmp_path, tiny_dataset):
    interaction_source, metadata_source = tiny_dataset
    interactions = tmp_path / "interactions.parquet"
    items = tmp_path / "items.parquet"
    convert_interactions(interaction_source, interactions)
    convert_metadata(metadata_source, items)
    report = profile_dataset(interactions, items, threads=1)
    assert report["totals"]["interactions"] == 4
    assert report["totals"]["users"] == 2
    assert report["totals"]["items"] == 3
    assert report["totals"]["repeated_user_item_rows"] == 1
    assert report["metadata_coverage"] == {
        "interaction_items": 3, "items_with_metadata": 2, "items_with_text": 2,
        "events_missing_metadata": 1, "metadata_rows": 3,
    }
    assert report["yearly_interactions"][0]["year"] == 2021
    assert report["yearly_interactions"][0]["interactions"] == 1
    assert report["yearly_first_observed_items"] == [
        {"year": 2021, "first_observed_items": 1},
        {"year": 2022, "first_observed_items": 2},
    ]
    assert report["user_histories_full_period_diagnostic_only"]["users_with_one_event"] == 1
    assert report["split_created"] is False
    assert report["local_filters_applied"] == []


def test_metadata_without_identifier_is_rejected():
    with pytest.raises(ValueError, match="parent_asin"):
        metadata_record({"title": "No identifier"})


def test_metadata_normalizes_lists_and_rejects_other_types():
    assert normalize_text([" Useful ", None, ["tool", " today "]]) == "Useful tool today"
    with pytest.raises(ValueError, match="text type"):
        normalize_text({"not": "text"})


def test_duplicate_metadata_is_rejected(tmp_path, tiny_dataset):
    interaction_source, metadata_source = tiny_dataset
    with gzip.open(metadata_source, "at", encoding="utf-8") as stream:
        stream.write(json.dumps({"parent_asin": "item-old", "title": "Duplicate"}) + "\n")
    interactions = tmp_path / "interactions.parquet"
    items = tmp_path / "items.parquet"
    convert_interactions(interaction_source, interactions)
    convert_metadata(metadata_source, items)
    with pytest.raises(ValueError, match="duplicate metadata"):
        profile_dataset(interactions, items, threads=1)


def test_invalid_rating_is_rejected(tmp_path, tiny_dataset):
    interaction_source, metadata_source = tiny_dataset
    with gzip.open(interaction_source, "at", encoding="utf-8") as stream:
        stream.write("user-3,item-old,7.0,1640995200003\n")
    interactions = tmp_path / "interactions.parquet"
    items = tmp_path / "items.parquet"
    convert_interactions(interaction_source, interactions)
    convert_metadata(metadata_source, items)
    with pytest.raises(ValueError, match="invalid interaction"):
        profile_dataset(interactions, items, threads=1)