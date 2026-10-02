import numpy as np

from scoperec.experiment import experiment_signature, rerank_new_bias


def test_bias_only_reranks_available_candidates():
    trace = {
        "predictions": np.array([[0, 1, 2, -1]]),
        "beam_scores": np.array([[-1.0, -2.0, -3.0, -np.inf]]),
    }
    groups = np.array([0, 2, 0, 2])
    assert rerank_new_bias(trace, groups, 2.0, cutoff=3).tolist() == [[1, 0, 2]]
    assert 3 not in rerank_new_bias(trace, groups, 100.0)


def test_config_signature_is_order_invariant_but_not_value_invariant():
    assert experiment_signature({"depth": 1, "rank": 8}) == experiment_signature(
        {"rank": 8, "depth": 1}
    )
    assert experiment_signature({"depth": 1}) != experiment_signature({"depth": 2})
