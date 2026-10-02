import numpy as np
import pytest

from scoperec.official_tokens import (
    encode_requests,
    metric_rows,
    metrics,
    released_codes,
    token_layout,
)


def test_released_collision_starts_at_one_before_offset():
    semantic = np.array([[0, 1, 2], [0, 1, 2], [255, 255, 255]])
    assert released_codes(semantic).tolist() == [
        [1, 258, 515, 770],
        [1, 258, 515, 771],
        [256, 512, 768, 770],
    ]
    assert token_layout() == {
        "padding": 0,
        "decoder_start": 0,
        "user_base": 1025,
        "user_tokens": 1,
        "eos": 1026,
        "vocab_size": 1027,
        "sid_length": 4,
    }


def test_released_input_includes_user_and_eos_with_five_label_tokens():
    codes = released_codes(np.array([[0, 1, 2], [3, 4, 5]]))
    requests = {
        "history": np.array([[0, -1], [0, 1]]),
        "user": np.array([19, 300]),
        "target": np.array([1, 0]),
    }
    inputs, labels = encode_requests(requests, codes, token_layout())
    assert inputs.tolist() == [
        [1025, 1, 258, 515, 770, 1026, 0, 0, 0, 0],
        [1025, 1, 258, 515, 770, 4, 261, 518, 770, 1026],
    ]
    assert labels.tolist() == [[4, 261, 518, 770, 1026], [1, 258, 515, 770, 1026]]


def test_prefix_hit_is_not_exact_item_hit_and_duplicates_only_count_once():
    predictions = np.array(
        [
            [[1, 258, 515, 999], [1, 258, 515, 770], [4, 261, 518, 770]],
            [[4, 261, 518, 770], [1, 258, 516, 770], [1, 258, 516, 771]],
        ]
    )
    targets = np.array([[1, 258, 515, 770], [1, 258, 515, 770]])
    results = metrics(predictions, targets, np.array([2, 0]), cutoffs=(1, 3))
    assert results["prefix"]["cold"]["recall@1"] == 1.0
    assert results["exact"]["cold"]["recall@1"] == 0.0
    assert results["prefix"]["overall"]["recall@3"] == 0.5
    assert results["exact"]["cold"]["ndcg@3"] == 1 / np.log2(3)
    with pytest.raises(ValueError, match="cutoff"):
        metric_rows(predictions, targets, 3)
