import numpy as np
import torch

from scoperec.adapt import cache_hidden, fit_patch, pseudo_requests
from scoperec.model import GenerativeRecommender


def test_pseudo_requests_only_reuse_old_histories_and_do_not_add_target():
    training = {
        "history": np.array([[0, -1], [1, -1], [0, 1]]),
        "target": np.array([1, 0, 1]),
        "timestamp": np.array([10, 20, 30]),
        "user": np.array([11, 22, 33]),
    }
    vectors = np.array([[1, 0], [0, 1], [1, 0]], dtype=np.float32)
    pseudo = pseudo_requests(training, vectors, np.array([2]), 1, 10, 42)
    assert pseudo["target"].tolist() == [2]
    assert pseudo["source_target"].tolist() == [0]
    assert pseudo["history"].tolist() == [[1, -1]]
    assert pseudo["timestamp"].tolist() == [20]


def test_cached_adaptation_updates_only_patch():
    torch.manual_seed(17)
    model = GenerativeRecommender(16, d_model=16, d_ff=32, num_layers=1, num_heads=2)
    codes = np.array([[3, 5, 7], [3, 5, 8], [4, 6, 9], [3, 5, 10]])
    requests = {"history": np.array([[0, -1], [1, -1]]), "target": np.array([3, 3])}
    old = {"history": np.array([[0, -1], [1, -1]]), "target": np.array([1, 2])}
    weights = {name: value.clone() for name, value in model.state_dict().items()}
    caches = {"new": cache_hidden(model, requests, codes), "old": cache_hidden(model, old, codes)}
    patch, report = fit_patch(
        model,
        codes,
        np.array([0, 0, 0, 2]),
        caches,
        1,
        2,
        1.0,
        {"steps": 5, "batch_size": 2, "learning_rate": 0.01},
        17,
    )
    assert torch.count_nonzero(patch.output_factor) > 0
    assert report["effective_old_requests"] == 1
    for name, value in model.state_dict().items():
        assert torch.equal(weights[name], value)
    assert all(parameter.grad is None for parameter in model.parameters())
