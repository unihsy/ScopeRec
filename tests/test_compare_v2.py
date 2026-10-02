import json

import numpy as np

from scoperec.compare_v2 import v2_utility
from scoperec.evaluate import save_evaluation
from scoperec.final_v2 import choose_working_points


def test_v2_selection_uses_fixed_retention_floor():
    frozen = {"old": {"ndcg@20": 0.1}}
    good = {"new": {"recall@20": 0.05}, "old": {"ndcg@20": 0.096}}
    bad = {"new": {"recall@20": 0.9}, "old": {"ndcg@20": 0.09}}
    assert v2_utility(good, frozen) > 0
    assert v2_utility(bad, frozen) == -1.0


def test_evaluation_filenames_preserve_decimal_hyperparameters(tmp_path):
    for alpha in (0.01, 0.1, 0.2):
        path = tmp_path / f"model_alpha{alpha}"
        save_evaluation(path, {"alpha": alpha}, {"values": np.array([alpha])})
    for alpha in (0.01, 0.1, 0.2):
        assert json.loads((tmp_path / f"model_alpha{alpha}.json").read_text())["alpha"] == alpha
        assert np.load(tmp_path / f"model_alpha{alpha}.npz")["values"][0] == alpha


def test_working_points_are_chosen_before_test_and_exclude_invalid_candidates():
    def candidate(name, utility, overall, new):
        return {
            "name": name,
            "utility": utility,
            "settings": {"alpha": 0.1},
            "metrics": {"overall": {"ndcg@20": overall}, "new": {"recall@20": new}},
        }

    options = [
        candidate("new_priority", 0.1, 0.08, 0.2),
        candidate("balanced", 0.05, 0.09, 0.1),
        candidate("unacceptable_retention", -1, 0.1, 0.9),
    ]
    chosen = choose_working_points({"candidates": {"scoperec_content": options}})
    assert chosen["scoperec_content"]["name"] == "new_priority"
    assert chosen["scoperec_content_balanced"]["name"] == "balanced"
    assert chosen["uniform_content_ablation"]["settings"]["uniform_new_distribution"]
