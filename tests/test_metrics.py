import numpy as np
import pytest

from scoperec.metrics import (
    aggregate_metrics,
    paired_user_bootstrap,
    per_request_metrics,
    validation_utility,
)


def test_metrics_use_first_hit_and_full_denominator():
    predictions = np.array([[1, 2, 3], [2, 1, 3], [3, 4, 5], [0, 2, 4]])
    targets = np.array([1, 1, 1, 2])
    result = per_request_metrics(predictions, targets, (1, 3))
    assert result["recall@1"].tolist() == [1, 0, 0, 0]
    assert result["recall@3"].mean() == 0.75
    assert result["ndcg@3"][1] == 1 / np.log2(3)
    groups = np.array([2, 0, 0, 1])
    summary = aggregate_metrics(predictions, targets, groups, cutoffs=(1, 3))
    assert summary["new"]["recall@3"] == 1.0
    assert summary["old"]["requests"] == 3
    with pytest.raises(ValueError, match="Not enough"):
        per_request_metrics(predictions, targets)


def test_bootstrap_pairs_users_not_independent_requests():
    result = paired_user_bootstrap([1.0, 1.0, 1.0], ["alice", "alice", "bob"], repetitions=100)
    assert result["users"] == 2
    assert result["low"] == result["high"] == result["difference"] == 1.0


def test_selection_preserves_old_metric_floor():
    frozen = {"old": {"ndcg@20": 0.1}}
    unacceptable = {"new": {"recall@20": 0.9}, "old": {"ndcg@20": 0.08}}
    acceptable = {"new": {"recall@20": 0.1}, "old": {"ndcg@20": 0.095}}
    assert validation_utility(unacceptable, frozen) == -1
    assert validation_utility(acceptable, frozen) == pytest.approx(0.0095)
