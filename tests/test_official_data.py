import numpy as np
import pytest

from scoperec.official_data import official_metadata, request_arrays, selected_test_users


def test_released_half_test_selection_includes_first_exceeding_row():
    records = [{"user_id": str(index)} for index in range(6)]
    assert selected_test_users(records, 0.5) == {"0", "1", "2", "3"}
    assert selected_test_users(records, 1) == {str(index) for index in range(6)}
    with pytest.raises(ValueError):
        selected_test_users(records, 0)


def test_official_training_expands_each_source_row_and_preserves_duplicates():
    rows = [
        {"user": 1, "sequence": [0, 1], "timestamp": 20, "source_row": 0},
        {"user": 1, "sequence": [0, 1, 2], "timestamp": 30, "source_row": 1},
    ]
    train = request_arrays(rows, 3, train=True)
    assert train["target"].tolist() == [1, 1, 2]
    assert train["history"].tolist() == [[0, -1, -1], [0, -1, -1], [0, 1, -1]]
    assert train["source_row_timestamp"].tolist() == [20, 30, 30]
    evaluation = request_arrays(rows, 3, train=False)
    np.testing.assert_array_equal(evaluation["target"], [1, 2])


def test_official_metadata_includes_categories_but_not_aggregate_ratings():
    assert (
        official_metadata(
            {
                "title": "<b>App &amp; tool</b>",
                "features": ["Fast"],
                "categories": ["Software"],
                "description": ["Works"],
                "rating_number": 10000,
            }
        )
        == "App & tool Fast. Software. Works. "
    )
