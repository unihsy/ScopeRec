import pytest

from scoperec.recommend_v2 import load_serving_model, recommend


def test_unknown_method_and_invalid_cutoff_fail_before_loading_artifacts():
    with pytest.raises(ValueError, match="Unknown v2"):
        load_serving_model("incorrect", 17, "cpu")
    with pytest.raises(ValueError, match="top-k"):
        recommend("tiger", 17, "cpu", 0, None, 51, False)
