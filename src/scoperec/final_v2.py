import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from scoperec.adapt import branch_diagnostics, cache_hidden, fit_patch, prepare_edits
from scoperec.compare_v2 import setup_comparison
from scoperec.content_scope import ContentScopeModel
from scoperec.evaluate import (
    content_predictions,
    evaluate_model,
    load_requests,
    save_evaluation,
)
from scoperec.experiment import experiment_signature, write_json
from scoperec.genrecedit import construct_editor, prepare_edit_statistics
from scoperec.metrics import aggregate_metrics
from scoperec.patches import save_patch
from scoperec.prepare import file_fingerprint

SELECTION = Path("reports/v2/validation/selection.json")


def choose_working_points(search):
    chosen = {}
    for method, options in search["candidates"].items():
        viable = [entry for entry in options if entry["utility"] >= 0]
        if not viable:
            raise ValueError(f"No viable validation candidate for {method}")
        chosen[method] = max(viable, key=lambda entry: entry["utility"])
        chosen[f"{method}_balanced"] = max(
            viable,
            key=lambda entry: (
                entry["metrics"]["overall"]["ndcg@20"],
                entry["metrics"]["new"]["recall@20"],
            ),
        )
    content_options = search["candidates"]["scoperec_content"]
    fixed = [
        entry
        for entry in content_options
        if entry["settings"].get("confidence_threshold") is None and entry["utility"] >= 0
    ]
    chosen["scoperec_content_fixed"] = max(fixed, key=lambda entry: entry["utility"])
    chosen["uniform_content_ablation"] = {
        "settings": {**chosen["scoperec_content"]["settings"], "uniform_new_distribution": True},
        "selection": "Same main settings; uniform instead of content-weighted branch distribution",
    }
    return chosen


def lock_selection():
    if SELECTION.is_file():
        existing = json.loads(SELECTION.read_text())
        if existing.get("locked"):
            raise FileExistsError("Selection is already locked; do not overwrite after testing")
    if any(Path("reports/v2/test").glob("seed_*/*.json")):
        raise RuntimeError("Cannot claim an untouched final window after test results exist")
    config = json.loads(Path("configs/experiment_v2.json").read_text())
    search_path = Path("reports/v2/validation/search.json")
    search = json.loads(search_path.read_text())
    if search["provenance"]["configuration_sha256"] != experiment_signature(config):
        raise ValueError("Validation search used a different configuration")
    parity = json.loads(Path("reports/v2/genrecedit_reference_parity.json").read_text())
    if not parity["passed"]:
        raise ValueError("Official GenRecEdit parity gate has not passed")
    chosen = choose_working_points(search)
    selection = {
        "locked": True,
        "locked_at": datetime.now(timezone.utc).isoformat(),
        "test_used_for_selection": False,
        "previously_seen_test": "2019 v1; not used as final v2",
        "confirmation_window": config["windows"]["test"],
        "configuration": config,
        "configuration_sha256": experiment_signature(config),
        "search_sha256": file_fingerprint(search_path)["sha256"],
        "official_parity_sha256": file_fingerprint(
            Path("reports/v2/genrecedit_reference_parity.json")
        )["sha256"],
        "selection_seed": 17,
        "tuning_requests": config["selection"]["tuning_requests"],
        "rules": {
            "new_priority": "Maximize new R@20 times old NDCG@20, old >= 95% of frozen",
            "balanced": "Maximize overall NDCG@20 among same retention-eligible candidates",
            "secondary_status": "Balanced and fixed/uniform variants declared before final testing",
        },
        "chosen": chosen,
    }
    write_json(SELECTION, selection)
    return {
        "locked": True,
        "choices": {name: entry.get("name", "explicit_ablation") for name, entry in chosen.items()},
    }


def verify_selection(config):
    selection = json.loads(SELECTION.read_text())
    if not selection["locked"] or selection["test_used_for_selection"]:
        raise ValueError("Final evaluation requires locked validation-only choices")
    if selection["configuration_sha256"] != experiment_signature(config):
        raise ValueError("Configuration changed after locking")
    return selection


def execute_final(seed, device):
    config, base, codes, provenance = setup_comparison(seed, device)
    selection = verify_selection(config)
    torch.set_float32_matmul_precision("highest")
    provenance.update({"selection_sha256": file_fingerprint(SELECTION)["sha256"], "split": "test"})
    directory = Path(f"reports/v2/test/seed_{seed}")
    if (directory / "completed.json").exists():
        raise FileExistsError("Completed final experiment exists; keep it immutable")
    directory.mkdir(parents=True, exist_ok=True)
    data = Path("data/experiment_v2")
    groups = np.load(data / "test_groups.npy")
    requests = load_requests(data, "test")
    vocab_size = base.specification["vocab_size"]
    frozen, frozen_trace = evaluate_model(base, requests, codes, groups, vocab_size)
    frozen.update({"method": "TIGER common backbone", "provenance": provenance, "seed": seed})
    save_evaluation(directory / "tiger", frozen, frozen_trace)
    write_json(
        directory / "prefix_diagnostics.json",
        branch_diagnostics(codes, groups, vocab_size, (1, 2, 3), frozen_trace),
    )
    samples, preparation = prepare_edits(
        data, Path(f"artifacts/v2/samples/test/seed_{seed}"), groups, config, "test"
    )
    positions, elapsed = prepare_edit_statistics(
        base,
        samples,
        codes,
        config["genrecedit"],
        Path(f"artifacts/v2/genrecedit/test/seed_{seed}"),
        provenance,
    )
    for name in ("genrecedit", "genrecedit_balanced"):
        weight = selection["chosen"][name]["settings"]["cov_lambda"]
        editor, details = construct_editor(base, positions, weight, device)
        result, trace = evaluate_model(editor, requests, codes, groups, vocab_size)
        result.update(
            {
                "method": name,
                "seed": seed,
                "provenance": provenance,
                "cov_lambda": weight,
                "position_diagnostics": details,
                "optimization_seconds": sum(position["seconds"] for position in positions),
                "statistics_load_or_compute_seconds": elapsed,
                "parameters": int(editor.weight_deltas.numel()),
                "reference_revision": config["genrecedit"]["upstream_revision"],
            }
        )
        save_evaluation(directory / name, result, trace)
        torch.save(
            {
                "position_layers": editor.position_layers,
                "weight_deltas": editor.weight_deltas.cpu(),
                "provenance": provenance,
                "cov_lambda": weight,
            },
            directory / f"{name}.pt",
        )
        print(
            json.dumps(
                {
                    "method": name,
                    "new_r20": result["metrics"]["new"]["recall@20"],
                    "old_n20": result["metrics"]["old"]["ndcg@20"],
                }
            ),
            flush=True,
        )
        del editor
    caches = {name: cache_hidden(base, samples[name], codes) for name in ("new", "old")}
    for name in ("scoperec_v1", "scoperec_v1_balanced"):
        settings = selection["chosen"][name]["settings"]
        patch, training = fit_patch(
            base,
            codes,
            groups,
            caches,
            settings["depth"],
            settings["rank"],
            settings["replay_weight"],
            config["adaptation"],
            seed,
        )
        result, trace = evaluate_model(base, requests, codes, groups, vocab_size, patch=patch)
        result.update(
            {
                "method": name,
                "seed": seed,
                "training": training,
                "provenance": provenance,
                "preparation": preparation,
            }
        )
        save_evaluation(directory / name, result, trace)
        save_patch(
            patch,
            directory / f"{name}.pt",
            provenance["base_sha256"],
            provenance["sid_sha256"],
            training,
        )
    del caches, patch
    vectors = np.load(data / "embeddings.npy")
    for name in (
        "scoperec_content",
        "scoperec_content_balanced",
        "scoperec_content_fixed",
        "uniform_content_ablation",
    ):
        settings = selection["chosen"][name]["settings"].copy()
        uniform = settings.pop("uniform_new_distribution", False)
        started = time.perf_counter()
        adapted = ContentScopeModel(base, codes, groups, vectors, **settings).to(device)
        adapted.uniform_new_distribution = uniform
        build_seconds = time.perf_counter() - started
        result, trace = evaluate_model(adapted, requests, codes, groups, vocab_size)
        result.update(
            {
                "method": name,
                "seed": seed,
                "provenance": provenance,
                "settings": {**settings, "uniform_new_distribution": uniform},
                "trainable_parameters": 0,
                "build_seconds": build_seconds,
                "new_index_bytes": int(vectors[groups == 2].nbytes),
                "total_vectors_bytes": int(vectors.nbytes),
            }
        )
        save_evaluation(directory / name, result, trace)
        print(
            json.dumps(
                {
                    "method": name,
                    "new_r20": result["metrics"]["new"]["recall@20"],
                    "old_n20": result["metrics"]["old"]["ndcg@20"],
                    "overall_n20": result["metrics"]["overall"]["ndcg@20"],
                }
            ),
            flush=True,
        )
        del adapted
    predictions, seconds = content_predictions(
        requests["history"], vectors, groups, device, mode="last"
    )
    save_evaluation(
        directory / "content_retrieval",
        {
            "method": "content_retrieval",
            "provenance": provenance,
            "seed": seed,
            "inference_seconds": seconds,
            "metrics": aggregate_metrics(predictions, requests["target"], requests["group"]),
        },
        {
            "predictions": predictions,
            "target": requests["target"],
            "group": requests["group"],
            "user": requests["user"],
        },
    )
    write_json(
        directory / "completed.json",
        {
            "seed": seed,
            "provenance": provenance,
            "requests": len(requests["target"]),
            "methods": 10,
            "completed": True,
        },
    )
    return {"seed": seed, "completed": True, "directory": str(directory)}


def main():
    parser = argparse.ArgumentParser(
        description="Lock validation choices and run untouched v2 tests."
    )
    parser.add_argument("stage", choices=("lock", "test"))
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--device", default="cuda:3")
    arguments = parser.parse_args()
    result = (
        lock_selection()
        if arguments.stage == "lock"
        else execute_final(arguments.seed, arguments.device)
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
