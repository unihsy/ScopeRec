import argparse
import json
from pathlib import Path

import numpy as np
import torch

from scoperec.experiment import experiment_signature, write_json
from scoperec.metrics import aggregate_metrics
from scoperec.prepare import file_fingerprint
from scoperec.protocol import timestamp_ms

METHODS_V2 = (
    "tiger",
    "genrecedit",
    "genrecedit_balanced",
    "scoperec_v1",
    "scoperec_v1_balanced",
    "scoperec_content",
    "scoperec_content_balanced",
    "scoperec_content_fixed",
    "uniform_content_ablation",
    "content_retrieval",
)


def validate_ranking(predictions, targets, available):
    if predictions.ndim != 2 or len(predictions) != len(targets) or predictions.shape[1] < 50:
        raise ValueError("Misaligned or incomplete candidate lists")
    if np.any(predictions < 0) or np.any(predictions >= len(available)):
        raise ValueError("Invalid item identifier in recommendations")
    if np.any(~available[predictions]):
        raise ValueError("Candidate not in editing snapshot")
    if np.any(np.diff(np.sort(predictions, axis=1), axis=1) == 0):
        raise ValueError("Repeated item in recommendations")


def verify_provenance(provenance, expected):
    for key, value in expected.items():
        if provenance.get(key) != value:
            raise ValueError(f"Artifact provenance mismatch: {key}")


def run_audit(root: Path):
    config = json.loads((root / "configs/experiment_v2.json").read_text())
    selection_path = root / "reports/v2/validation/selection.json"
    selection = json.loads(selection_path.read_text())
    if not selection["locked"] or selection["test_used_for_selection"]:
        raise ValueError("Final test was not protected by a selection lock")
    if selection["configuration_sha256"] != experiment_signature(config):
        raise ValueError("Post-selection configuration mutation")
    if (
        selection["search_sha256"]
        != file_fingerprint(root / "reports/v2/validation/search.json")["sha256"]
    ):
        raise ValueError("Validation search changed after locking")
    parity_path = root / "reports/v2/genrecedit_reference_parity.json"
    parity = json.loads(parity_path.read_text())
    if (
        not parity["passed"]
        or selection["official_parity_sha256"] != file_fingerprint(parity_path)["sha256"]
    ):
        raise ValueError("Official function parity gate changed")
    for position in parity["position_optimization"]:
        if not position["success_mask_identical"] or position["max_delta_abs_difference"] > 5e-4:
            raise ValueError("Official optimizer comparison outside tolerance")
    manifest_path = root / "artifacts/v2/upstream/source_manifest.json"
    if file_fingerprint(manifest_path)["sha256"] != parity["source_manifest"]["sha256"]:
        raise ValueError("Upstream source manifest changed")
    manifest = json.loads(manifest_path.read_text())
    for entry in manifest["files"]:
        if file_fingerprint(root / entry["path"])["sha256"] != entry["sha256"]:
            raise ValueError("Official reference code changed")
    dataset = root / "data/experiment_v2"
    training = np.load(dataset / "train.npz")
    holdout = np.load(dataset / "holdout.npz")
    test = np.load(dataset / "test.npz")
    groups = np.load(dataset / "test_groups.npy")
    codes = np.load(root / "artifacts/sid/codes.npy")
    if len(np.unique(codes, axis=0)) != len(codes):
        raise ValueError("Ambiguous SID mapping")
    if np.intersect1d(training["user"], holdout["user"]).size:
        raise ValueError("Pre-base holdout users overlap training")
    cutoff = timestamp_ms(config["base_cutoff"])
    if max(training["timestamp"].max(), holdout["timestamp"].max()) >= cutoff:
        raise ValueError("Future data in base training/holdout")
    window = config["windows"]["test"]
    if test["timestamp"].min() < timestamp_ms(window["edit_at"]) or test["timestamp"].max() >= (
        timestamp_ms(window["end"])
    ):
        raise ValueError("Test request outside the fresh confirmation window")
    np.testing.assert_array_equal(test["group"], groups[test["target"]])
    checked = []
    sources = [
        root / "configs/experiment_v2.json",
        selection_path,
        parity_path,
        dataset / "test.npz",
        root / "artifacts/sid/codes.npy",
    ]
    for seed in config["seeds"]:
        base_path = root / f"artifacts/v2/baseline/seed_{seed}/model.pt"
        expected = {
            "base_sha256": file_fingerprint(base_path)["sha256"],
            "sid_sha256": file_fingerprint(root / "artifacts/sid/codes.npy")["sha256"],
            "configuration_sha256": experiment_signature(config),
            "selection_sha256": file_fingerprint(selection_path)["sha256"],
            "split": "test",
        }
        directory = root / f"reports/v2/test/seed_{seed}"
        completed = json.loads((directory / "completed.json").read_text())
        verify_provenance(completed["provenance"], expected)
        if completed["methods"] != len(METHODS_V2):
            raise ValueError("Missing predeclared methods")
        for method in METHODS_V2:
            report_path = directory / f"{method}.json"
            trace_path = directory / f"{method}.npz"
            report = json.loads(report_path.read_text())
            verify_provenance(report["provenance"], expected)
            with np.load(trace_path) as trace:
                for field in ("target", "group", "user"):
                    np.testing.assert_array_equal(trace[field], test[field])
                validate_ranking(trace["predictions"], test["target"], groups >= 0)
                recomputed = aggregate_metrics(trace["predictions"], test["target"], test["group"])
                for group, metrics in recomputed.items():
                    for metric, value in metrics.items():
                        observed = report["metrics"][group][metric]
                        if value is None:
                            if observed is not None:
                                raise ValueError("Non-null metric for empty group")
                        elif not np.isclose(value, observed, atol=1e-12, rtol=0):
                            raise ValueError(
                                f"Metric mismatch for {seed}/{method}/{group}/{metric}"
                            )
            chosen = selection["chosen"].get(method)
            if chosen is not None:
                if method.startswith("genrecedit"):
                    if report["cov_lambda"] != chosen["settings"]["cov_lambda"]:
                        raise ValueError("GenRecEdit cov_lambda differs from selected value")
                elif method.startswith("scoperec_v1"):
                    for key in ("depth", "rank", "replay_weight"):
                        if report["training"][key] != chosen["settings"][key]:
                            raise ValueError("Learned patch differs from selected value")
                else:
                    for key, value in chosen["settings"].items():
                        if report["settings"].get(key) != value:
                            raise ValueError("Content mixture differs from selected value")
            checked.append(f"{seed}/{method}")
            sources.extend([report_path, trace_path])
        sample_directory = root / f"artifacts/v2/samples/test/seed_{seed}"
        with np.load(sample_directory / "pseudo.npz") as pseudo:
            if pseudo["timestamp"].max() >= cutoff or np.any(groups[pseudo["target"]] != 2):
                raise ValueError("Incorrect zero-shot editing sample boundary")
            if np.intersect1d(pseudo["target"], training["target"]).size:
                raise ValueError("New item already in base training targets")
        for position in range(4):
            cache = torch.load(
                root / f"artifacts/v2/genrecedit/test/seed_{seed}/position_{position}.pt",
                weights_only=True,
                map_location="cpu",
            )
            verify_provenance(cache["provenance"], expected)
            if cache["settings"] != config["genrecedit"]:
                raise ValueError("GenRecEdit optimization settings changed")
        integrity = json.loads(
            (root / f"reports/v2/diagnostics/seed_{seed}/integrity_and_latency.json").read_text()
        )
        for method, values in integrity["methods"].items():
            if (
                not all(
                    values[key]
                    for key in (
                        "reloaded_predictions_match",
                        "rollback_predictions_match",
                        "backbone_unchanged",
                    )
                )
                or values["rollback_max_log_probability_difference"] != 0
            ):
                raise ValueError(f"Editor persistence or rollback failed: {seed}/{method}")
        sources.append(base_path)
    return {
        "passed": True,
        "confirmation_evaluations_checked": len(checked),
        "evaluations": checked,
        "test_requests": len(test["target"]),
        "new_test_requests": int(np.sum(test["group"] == 2)),
        "checks": [
            "Official function numerical parity",
            "Pinned source hashes",
            "Selection locked before fresh 2020 final window",
            "Checkpoint and SID hashes",
            "Same requests and catalog for all methods",
            "Full ranking metric recomputation",
            "Legal distinct candidates",
            "Hyperparameters equal locked choices",
            "Pre-base zero-interaction pseudo sources",
            "Exact restore and reload",
        ],
        "artifacts": [file_fingerprint(path) for path in sources],
        "limitations": [
            "Observed local artifact audit, not independent third-party certification",
            "Same dataset family and previously used validation window",
        ],
    }


def main():
    parser = argparse.ArgumentParser(
        description="Audit all v2 confirmation results and official parity."
    )
    parser.add_argument("--root", type=Path, default=Path.cwd())
    arguments = parser.parse_args()
    report = run_audit(arguments.root)
    write_json(arguments.root / "reports/v2/audit.json", report)
    print(
        json.dumps({name: value for name, value in report.items() if name != "artifacts"}, indent=2)
    )


if __name__ == "__main__":
    main()
