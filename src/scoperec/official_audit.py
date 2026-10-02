import argparse
import json
from pathlib import Path

import numpy as np
import torch

from scoperec.experiment import experiment_signature, write_json
from scoperec.official_experiment import METHODS, ROOT
from scoperec.official_tokens import metrics, resolve_items
from scoperec.official_train import read_requests
from scoperec.prepare import file_fingerprint


def check_metrics(observed, expected):
    for definition, groups in expected.items():
        for group, values in groups.items():
            for metric, value in values.items():
                actual = observed[definition][group][metric]
                if value is None:
                    if actual is not None:
                        raise ValueError("Nonempty metric on empty subgroup")
                elif not np.isclose(actual, value, rtol=0, atol=1e-12):
                    raise ValueError(f"Metric mismatch: {definition}/{group}/{metric}")


def run_audit():
    config = json.loads(Path("configs/official_software.json").read_text())
    selection_path = ROOT / "selection.json"
    selection = json.loads(selection_path.read_text())
    config_sha = experiment_signature(config)
    if not selection["locked"] or selection["test_metrics_used"]:
        raise ValueError("Validation-only selection lock missing")
    if selection["configuration_sha256"] != config_sha:
        raise ValueError("Configuration changed after selection")
    if file_fingerprint(ROOT / "parity.json")["sha256"] != selection["official_parity_sha256"]:
        raise ValueError("Official tokenizer/decoder parity record changed after selection")
    parity = json.loads((ROOT / "parity.json").read_text())
    if not parity["passed"]:
        raise ValueError("Official functional parity was not passed")
    upstream = Path("artifacts/official_benchmark/upstream/source_manifest.json")
    manifest = json.loads(upstream.read_text())
    for record in manifest["files"]:
        if file_fingerprint(Path(record["path"]))["sha256"] != record["sha256"]:
            raise ValueError("Pinned upstream source changed")
    data = Path("data/official_software")
    protocol = json.loads((data / "protocol.json").read_text())
    for record in protocol["source_files"]:
        if file_fingerprint(Path(record["path"]))["sha256"] != record["sha256"]:
            raise ValueError("Official split archive changed")
    sid = Path("artifacts/official_benchmark/sid")
    sid_info = json.loads((sid / "sid.json").read_text())
    codes = np.load(sid / "codes.npy")
    for name in ("codes", "catalog", "embedding_file", "training_mask"):
        record = sid_info[name]
        if file_fingerprint(Path(record["path"]))["sha256"] != record["sha256"]:
            raise ValueError(f"Official feature artifact changed: {name}")
    if len(np.unique(codes, axis=0)) != len(codes):
        raise ValueError("Complete item codes are not unique")
    training = read_requests(data / "train.npz")
    test = read_requests(data / "test.npz")
    validation = read_requests(data / "valid.npz")
    if len(test["target"]) != protocol["split_summaries"]["test"]["retained_rows"]:
        raise ValueError("Test row count mismatch")
    for entry in selection["candidates"]:
        candidate = json.loads((ROOT / f"validation/{entry['name']}.json").read_text())
        with np.load(ROOT / f"validation/{entry['name']}.npz") as trace:
            check_metrics(
                candidate["metrics"],
                metrics(trace["codes"], codes[validation["target"]], validation["group"]),
            )
        if entry["metrics"] != candidate["metrics"] or entry["settings"] != candidate["settings"]:
            raise ValueError("Validation candidate record changed")
    selected = max(
        selection["candidates"],
        key=lambda entry: (
            entry["metrics"]["prefix"]["overall"]["ndcg@10"],
            entry["metrics"]["prefix"]["cold"]["recall@20"],
        ),
    )
    if selected != selection["selected"]:
        raise ValueError("Selection does not follow the declared metric")
    checked = []
    files = [file_fingerprint(selection_path), file_fingerprint(sid / "codes.npy")]
    for seed in config["seeds"]:
        baseline_dir = Path(f"artifacts/official_benchmark/baseline/seed_{seed}")
        checkpoint_path = baseline_dir / "model.pt"
        checkpoint = torch.load(checkpoint_path, weights_only=True, map_location="cpu")
        if checkpoint["epoch"] != config["training"]["epochs"] or checkpoint["config"] != config:
            raise ValueError("Baseline checkpoint does not match released training configuration")
        if checkpoint["data_sha256"] != file_fingerprint(data / "train.npz")["sha256"]:
            raise ValueError("Baseline training data changed")
        expected = {
            "base_sha256": file_fingerprint(checkpoint_path)["sha256"],
            "sid_sha256": sid_info["codes"]["sha256"],
            "configuration_sha256": config_sha,
            "selection_sha256": file_fingerprint(selection_path)["sha256"],
            "seed": seed,
        }
        directory = ROOT / f"test/seed_{seed}"
        completed = json.loads((directory / "completed.json").read_text())
        if completed["provenance"] != expected or tuple(completed["methods"]) != METHODS:
            raise ValueError("Final experiment completeness/provenance mismatch")
        for method in METHODS:
            result_path = directory / f"{method}.json"
            result = json.loads(result_path.read_text())
            if result["provenance"] != expected or not result["unmasked_full_vocabulary"]:
                raise ValueError("Final protocol mismatch")
            with np.load(directory / f"{method}.npz") as trace:
                for name in ("target", "group", "user", "source_row"):
                    np.testing.assert_array_equal(test[name], trace[name])
                if trace["sequences"].shape != (len(test["target"]), config["beam_size"], 5):
                    raise ValueError("Incorrect generation shape")
                if np.any(trace["sequences"] < 0) or np.any(trace["sequences"] >= 1027):
                    raise ValueError("Generated token outside full vocabulary")
                np.testing.assert_array_equal(trace["codes"], trace["sequences"][:, :, :4])
                resolved = resolve_items(trace["codes"], codes)
                np.testing.assert_array_equal(resolved, trace["predictions"])
                if float(np.mean(resolved < 0)) != result["invalid_full_sid_fraction"]:
                    raise ValueError("Invalid-item fraction mismatch")
                check_metrics(
                    result["metrics"], metrics(trace["codes"], codes[test["target"]], test["group"])
                )
            if method == "scoperec_c" and result["settings"] != selected["settings"]:
                raise ValueError("ScopeRec used a post-selection setting")
            checked.append(f"{seed}/{method}")
            files.append(file_fingerprint(result_path))
        pseudo = read_requests(
            Path(f"artifacts/official_benchmark/samples/test/seed_{seed}/pseudo.npz")
        )
        np.testing.assert_array_equal(
            pseudo["history"], training["history"][pseudo["source_training_row"]]
        )
        if not np.isin(pseudo["target"], test["target"][test["group"] == 2]).all():
            raise ValueError("Edited target not in released cold split")
        integrity = json.loads((ROOT / f"diagnostics/seed_{seed}.json").read_text())
        for values in integrity["methods"].values():
            if not all(
                values[name]
                for name in (
                    "predictions_reloaded_identically",
                    "disabled_predictions_match",
                    "backbone_unchanged",
                )
            ):
                raise ValueError("Editor integrity check failed")
            if values["path_score_rollback_error"] != 0:
                raise ValueError("Editor did not restore exact base scores")
        if seed == 2024 and not integrity["official_editor_parity"]["passed"]:
            raise ValueError("Official optimizer parity not passed")
    return {
        "passed": True,
        "final_evaluations_checked": len(checked),
        "evaluations": checked,
        "validation_candidates_checked": len(selection["candidates"]),
        "test_requests": len(test["target"]),
        "checks": [
            "Pinned official source and split hashes",
            "Sentence-T5/SID/catalog hashes",
            "Released tokenizer, five-step unmasked Beam and prefix NDCG parity",
            "Actual official target-optimization and matrix-update parity",
            "Last-epoch checkpoints, common data and models",
            "Validation-only selection",
            "All prefix and exact metrics recomputed",
            "Invalid paths retained and counted",
            "Same requests for every method",
            "Pseudo histories from released training",
            "Identical editor reload and rollback",
        ],
        "explicit_nonclaim": "Artifact consistency does not imply paper-table reproduction",
        "artifacts": files,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Audit the released Software benchmark experiment."
    )
    parser.parse_args()
    result = run_audit()
    write_json(ROOT / "audit.json", result)
    print(json.dumps({key: value for key, value in result.items() if key != "artifacts"}, indent=2))


if __name__ == "__main__":
    main()
