import argparse
import json
import time
from pathlib import Path

import numpy as np

from scoperec.adapt import cache_hidden, fit_patch, prepare_edits
from scoperec.content_scope import ContentScopeModel
from scoperec.evaluate import (
    content_predictions,
    evaluate_model,
    load_requests,
    save_evaluation,
)
from scoperec.experiment import experiment_signature, write_json
from scoperec.metrics import aggregate_metrics
from scoperec.patches import save_patch
from scoperec.prepare import file_fingerprint
from scoperec.train import load_base, setup_torch


def v2_utility(metrics, frozen, minimum_old=0.95):
    if metrics["old"]["ndcg@20"] < minimum_old * frozen["old"]["ndcg@20"]:
        return -1.0
    return metrics["new"]["recall@20"] * metrics["old"]["ndcg@20"]


def setup_comparison(seed, device):
    setup_torch(seed, device)
    config = json.loads(Path("configs/experiment_v2.json").read_text())
    checkpoint_path = Path(f"artifacts/v2/baseline/seed_{seed}/model.pt")
    model, checkpoint = load_base(checkpoint_path, device)
    model.eval().requires_grad_(False)
    codes = np.load("artifacts/sid/codes.npy")
    sid_sha = file_fingerprint(Path("artifacts/sid/codes.npy"))["sha256"]
    if checkpoint["sid_sha256"] != sid_sha:
        raise ValueError("Baseline/SID version mismatch")
    return (
        config,
        model,
        codes,
        {
            "base_sha256": file_fingerprint(checkpoint_path)["sha256"],
            "sid_sha256": sid_sha,
            "configuration_sha256": experiment_signature(config),
        },
    )


def run_validation(device, content_only=False):
    config, base, codes, provenance = setup_comparison(17, device)
    data = Path("data/experiment_v2")
    groups = np.load(data / "validation_groups.npy")
    dataset = load_requests(data, "validation", config["selection"]["tuning_requests"])
    directory = Path("reports/v2/validation/seed_17")
    directory.mkdir(parents=True, exist_ok=True)
    vocab_size = base.specification["vocab_size"]
    if not (directory / "tiger.json").exists():
        result, trace = evaluate_model(base, dataset, codes, groups, vocab_size)
        result.update({"provenance": provenance, "method": "TIGER unified temporal baseline"})
        save_evaluation(directory / "tiger", result, trace)
    frozen = json.loads((directory / "tiger.json").read_text())
    print(json.dumps({"method": "TIGER", "metrics": frozen["metrics"]}), flush=True)
    if not content_only:
        samples, preparation = prepare_edits(
            data, Path("artifacts/v2/samples/validation/seed_17"), groups, config, "validation"
        )
        caches = {name: cache_hidden(base, samples[name], codes) for name in ("new", "old")}
        for depth in (1, 2, 3):
            for replay_weight in (0.2, 1.0, 5.0):
                name = f"scoperec_v1_d{depth}_w{replay_weight:g}"
                if (directory / f"{name}.json").exists():
                    continue
                patch, training = fit_patch(
                    base,
                    codes,
                    groups,
                    caches,
                    depth,
                    8,
                    replay_weight,
                    config["adaptation"],
                    17,
                )
                result, trace = evaluate_model(
                    base, dataset, codes, groups, vocab_size, patch=patch
                )
                result.update(
                    {
                        "method": "ScopeRec v1 low-rank",
                        "training": training,
                        "preparation": preparation,
                        "provenance": provenance,
                        "settings": {"depth": depth, "replay_weight": replay_weight, "rank": 8},
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
        del caches
    vectors = np.load(data / "embeddings.npy")
    for depth in (1, 2):
        for temperature in (0.025, 0.05, 0.1):
            for alpha in (0.01, 0.025, 0.05, 0.1, 0.2):
                name = f"content_scope_d{depth}_t{temperature:g}_a{alpha:g}".replace(".", "p")
                if (directory / f"{name}.json").exists():
                    continue
                settings = {
                    "depth": depth,
                    "alpha": alpha,
                    "temperature": temperature,
                    "query_mode": "last",
                }
                adapted = ContentScopeModel(base, codes, groups, vectors, **settings).to(device)
                result, trace = evaluate_model(adapted, dataset, codes, groups, vocab_size)
                result.update(
                    {
                        "method": "ScopeRec content-conditional mass mixture",
                        "settings": settings,
                        "provenance": provenance,
                        "trainable_parameters": 0,
                        "content_index_bytes": int(vectors[groups == 2].nbytes),
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
    content, seconds = content_predictions(dataset["history"], vectors, groups, device, mode="last")
    save_evaluation(
        directory / "content_retrieval",
        {
            "metrics": aggregate_metrics(content, dataset["target"], dataset["group"]),
            "inference_seconds": seconds,
            "method": "Content last-item retrieval",
        },
        {
            "predictions": content,
            "target": dataset["target"],
            "group": dataset["group"],
            "user": dataset["user"],
        },
    )
    summarize_validation(config, directory, provenance, frozen)


def summarize_validation(config, directory, provenance, frozen):
    candidates = {"genrecedit": [], "scoperec_v1": [], "scoperec_content": []}
    for method, pattern in (
        ("genrecedit", "genrecedit_cov*.json"),
        ("scoperec_v1", "scoperec_v1_*.json"),
        ("scoperec_content", "content_scope_*.json"),
    ):
        for path in sorted(directory.glob(pattern)):
            if method == "scoperec_content" and "." in path.stem:
                continue
            result = json.loads(path.read_text())
            candidates[method].append(
                {
                    "name": path.stem,
                    "settings": result.get("settings", {"cov_lambda": result.get("cov_lambda")}),
                    "metrics": result["metrics"],
                    "utility": v2_utility(
                        result["metrics"],
                        frozen["metrics"],
                        config["selection"]["minimum_old_ndcg_ratio"],
                    ),
                }
            )
    chosen = {}
    for method, options in candidates.items():
        viable = [entry for entry in options if entry["utility"] >= 0]
        if viable:
            chosen[method] = max(viable, key=lambda entry: entry["utility"])
        else:
            chosen[method] = {"no_candidate_satisfies_old_floor": True}
    report = {
        "candidates": candidates,
        "suggested": chosen,
        "provenance": provenance,
        "configuration": config,
        "locked": False,
        "test_evaluated": False,
        "selection_rule": config["selection"],
        "generated_at_unix": time.time(),
    }
    write_json(Path("reports/v2/validation/search.json"), report)
    print(
        json.dumps(
            {
                "validation_summary": {
                    name: entry.get("name", entry) for name, entry in chosen.items()
                }
            },
            indent=2,
        )
    )


def main():
    parser = argparse.ArgumentParser(
        description="Compare GenRecEdit, TIGER and scoped content edits."
    )
    parser.add_argument("--device", default="cuda:3")
    parser.add_argument("--content-only", action="store_true")
    arguments = parser.parse_args()
    run_validation(arguments.device, arguments.content_only)


if __name__ == "__main__":
    main()
