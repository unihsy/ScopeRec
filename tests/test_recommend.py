import pytest

from scoperec.recommend import resolve_history


def test_custom_history_keeps_recency_and_validates_ids():
    catalog = [{"parent_asin": "old"}, {"parent_asin": "new"}, {"parent_asin": "latest"}]
    assert resolve_history(["old", "new", "latest"], catalog, 2).tolist() == [[1, 2]]
    assert resolve_history(["old"], catalog, 3).tolist() == [[0, -1, -1]]
    with pytest.raises(ValueError, match="Unknown"):
        resolve_history(["missing"], catalog, 3)
    with pytest.raises(ValueError, match="At least"):
        resolve_history([], catalog, 3)
