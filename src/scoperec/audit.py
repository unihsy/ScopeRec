import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as parquet
import torch

from scoperec.experiment import experiment_signature, write_json
from scoperec.metrics import aggregate_metrics
from scoperec.prepare import file_fingerprint
from scoperec.protocol import timestamp_ms

METHODS = (
    "frozen",
    "local",
    "global",
    "local_few_shot",
    "global_few_shot",
    "content",
    "popularity",
    "frozen_beam200",
    "new_bias",
    "local_no_replay",
    "global_no_replay",
)


def audit_project(root: Path) -> dict:
    config = json.loads((root / "configs/experiment.json").read_text())
    selection = json.loads((root / "reports/validation/selection.json").read_text())
    if selection["provenance"]["experiment_sha256"] != experiment_signature(config):
        raise ValueError("Final configuration differs from locked validation selection")
    if not selection["locked"] or selection["test_used_for_selection"]:
        raise ValueError("Invalid model selection provenance")
    dataset = root / "data/experiment"
    train = np.load(dataset / "train.npz")
    holdout = np.load(dataset / "holdout.npz")
    base = timestamp_ms(config["base_cutoff"])
    if train["timestamp"].max() >= base or holdout["timestamp"].max() >= base:
        raise ValueError("Post-cutoff examples in baseline training or holdout")
    if np.intersect1d(train["user"], holdout["user"]).size:
        raise ValueError("Held-out baseline users overlap training users")
    catalog = parquet.read_table(dataset / "catalog.parquet", columns=["first_seen"])
    first_seen = catalog["first_seen"].to_numpy()
    for name, window in config["windows"].items():
        requests = np.load(dataset / f"{name}.npz")
        edit = timestamp_ms(window["edit_at"])
        end = timestamp_ms(window["end"])
        if requests["timestamp"].min() < edit or requests["timestamp"].max() >= end:
            raise ValueError(f"Request outside its time window: {name}")
        if np.any(first_seen[requests["target"]] >= edit):
            raise ValueError(f"Target absent from editing catalog: {name}")
        valid = requests["history"] >= 0
        history_times = first_seen[np.maximum(requests["history"], 0)]
        if np.any(valid & (history_times >= requests["timestamp"][:, None])):
            raise ValueError(f"Future item in request history: {name}")
        few = np.load(dataset / f"{name}_few_shot.npz")
        if np.any(few["timestamp"] >= edit) or np.any(
            few["timestamp"] <= first_seen[few["target"]]
        ):
            raise ValueError("Few-shot examples include arrival events or post-edit labels")
    codes = np.load(root / "artifacts/sid/codes.npy")
    if len(np.unique(codes, axis=0)) != len(codes):
        raise ValueError("Ambiguous full SID mapping")
    sid_sha = file_fingerprint(root / "artifacts/sid/codes.npy")["sha256"]
    requests = np.load(dataset / "test.npz")
    groups = np.load(dataset / "test_groups.npy")
    checked = []
    for seed in config["seeds"]:
        baseline = root / f"artifacts/baseline/seed_{seed}/model.pt"
        checkpoint = torch.load(baseline, weights_only=True, map_location="cpu")
        if checkpoint["sid_sha256"] != sid_sha:
            raise ValueError("Baseline SID hash mismatch")
        completed = json.loads((root / f"reports/test/seed_{seed}/completed.json").read_text())
        if completed["provenance"]["base_sha256"] != file_fingerprint(baseline)["sha256"]:
            raise ValueError("Baseline changed after evaluation")
        if (
            completed["selection_sha256"]
            != file_fingerprint(root / "reports/validation/selection.json")["sha256"]
        ):
            raise ValueError("Selection was changed after final evaluation")
        for method in METHODS:
            path = root / f"reports/test/seed_{seed}/{method}"
            report = json.loads(path.with_suffix(".json").read_text())
            predictions = np.load(path.with_suffix(".npz"))
            for field in ("target", "group", "user"):
                np.testing.assert_array_equal(predictions[field], requests[field])
            ranked = predictions["predictions"]
            if ranked.shape[1] < 50 or np.any(ranked < 0) or np.any(groups[ranked] < 0):
                raise ValueError("Invalid recommendation candidates")
            if np.any(np.diff(np.sort(ranked, axis=1), axis=1) == 0):
                raise ValueError("Duplicate candidates in recommendation list")
            reproduced = aggregate_metrics(ranked, requests["target"], requests["group"])
            for group_name, metrics in reproduced.items():
                for metric, value in metrics.items():
                    observed = report["metrics"][group_name][metric]
                    if value is None:
                        if observed is not None:
                            raise ValueError("Nonempty metric for empty group")
                    elif not np.isclose(value, observed, atol=1e-12, rtol=0):
                        raise ValueError(f"Metric mismatch: {seed}/{method}/{group_name}/{metric}")
            checked.append(f"{seed}/{method}")
        pseudo = np.load(root / f"artifacts/edits/test_seed_{seed}/pseudo.npz")
        if np.any(pseudo["timestamp"] >= base) or np.any(groups[pseudo["target"]] != 2):
            raise ValueError("Invalid zero-shot pseudo source")
        if np.intersect1d(pseudo["target"], train["target"]).size:
            raise ValueError("New target appeared in baseline training")
        rollback = json.loads((root / f"reports/diagnostics/seed_{seed}/rollback.json").read_text())
        if not rollback["same_predictions"] or rollback["max_path_score_change"] != 0:
            raise ValueError("Patch rollback changed base behavior")
    return {
        "passed": True,
        "primary_evaluations_checked": len(checked),
        "evaluations": checked,
        "checks": [
            "locked validation selection",
            "pre-base disjoint-user holdout",
            "time windows and catalog availability",
            "strictly earlier item availability",
            "few-shot availability",
            "unique SID",
            "model and selection hashes",
            "same requests for all methods",
            "legal unique recommendations",
            "metrics recomputed from saved predictions",
            "zero-shot source boundary",
            "identical rollback",
        ],
        "caveat": "History event ordering is additionally covered by protocol unit tests",
    }


def main():
    parser = argparse.ArgumentParser(description="Audit actual experiment artifacts and metrics.")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    arguments = parser.parse_args()
    result = audit_project(arguments.root)
    write_json(arguments.root / "reports/audit.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
