import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from scoperec.adapt import (
    branch_diagnostics,
    cache_hidden,
    fit_patch,
    merge_caches,
    prepare_edits,
)
from scoperec.evaluate import (
    content_predictions,
    evaluate_model,
    load_requests,
    popularity_predictions,
    save_evaluation,
)
from scoperec.metrics import aggregate_metrics, validation_utility
from scoperec.patches import save_patch
from scoperec.prepare import file_fingerprint
from scoperec.train import load_base, setup_torch


def write_json(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def experiment_signature(config):
    return hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()


def rerank_new_bias(trace, groups, bias, cutoff=50):
    items = trace["predictions"]
    scores = trace["beam_scores"].copy()
    scores += bias * (groups[np.maximum(items, 0)] == 2)
    scores[items < 0] = -np.inf
    order = np.argsort(-scores, axis=1, kind="stable")[:, :cutoff]
    return np.take_along_axis(items, order, axis=1)


def setup_run(config_path, data, sid, seed, device):
    setup_torch(seed, device)
    config = json.loads(config_path.read_text())
    checkpoint_path = Path(f"artifacts/baseline/seed_{seed}/model.pt")
    model, checkpoint = load_base(checkpoint_path, device)
    model.eval().requires_grad_(False)
    codes = np.load(sid / "codes.npy")
    sid_sha = file_fingerprint(sid / "codes.npy")["sha256"]
    if checkpoint["sid_sha256"] != sid_sha:
        raise ValueError("Baseline and SID versions disagree")
    provenance = {
        "base_sha256": file_fingerprint(checkpoint_path)["sha256"],
        "sid_sha256": sid_sha,
        "experiment_sha256": experiment_signature(config),
    }
    return config, model, codes, provenance


def train_and_evaluate(
    model,
    codes,
    groups,
    caches,
    requests,
    specification,
    config,
    seed,
    directory,
    name,
    provenance,
    preparation_seconds,
):
    patch, training = fit_patch(
        model,
        codes,
        groups,
        caches,
        specification["depth"],
        specification["rank"],
        specification["replay_weight"],
        config["adaptation"],
        seed,
        specification["mode"],
    )
    patch_path = directory / f"{name}.pt"
    started = time.perf_counter()
    save_patch(patch, patch_path, provenance["base_sha256"], provenance["sid_sha256"], training)
    result, trace = evaluate_model(
        model,
        requests,
        codes,
        groups,
        model.specification["vocab_size"],
        config["beam_size"],
        patch,
    )
    training["artifact_bytes"] = patch_path.stat().st_size
    training["artifact_save_and_evaluation_seconds"] = time.perf_counter() - started
    training["sample_preparation_seconds"] = preparation_seconds
    training["update_seconds_excluding_shared_text_encoding"] = (
        preparation_seconds + training["cache_seconds"] + training["optimizer_seconds"]
    )
    result.update({"training": training, "provenance": provenance, "seed": seed})
    save_evaluation(directory / name, result, trace)
    print(
        json.dumps(
            {
                "name": name,
                "new_r20": result["metrics"]["new"]["recall@20"],
                "old_n20": result["metrics"]["old"]["ndcg@20"],
                "parameters": training["parameters"],
            }
        ),
        flush=True,
    )
    return patch, result, trace


def validation(config_path: Path, data: Path, sid: Path, device: str, limit: int):
    seed = 17
    config, model, codes, provenance = setup_run(config_path, data, sid, seed, device)
    directory = Path("reports/validation/grid")
    directory.mkdir(parents=True, exist_ok=True)
    groups = np.load(data / "validation_groups.npy")
    requests = load_requests(data, "validation", limit)
    vocab_size = model.specification["vocab_size"]
    frozen, frozen_trace = evaluate_model(model, requests, codes, groups, vocab_size)
    save_evaluation(directory / "frozen", frozen, frozen_trace)
    write_json(
        directory / "prefix_diagnostics.json",
        branch_diagnostics(codes, groups, vocab_size, config["adaptation"]["depths"], frozen_trace),
    )
    samples, preparation = prepare_edits(
        data, Path("artifacts/edits/validation"), groups, config, "validation"
    )
    print(json.dumps(preparation), flush=True)
    caches = {name: cache_hidden(model, samples[name], codes) for name in ("new", "old")}
    candidates = {"local": [], "global": []}
    for mode, rank in (("local", 8), ("global", 8), ("global", 64)):
        for depth in config["adaptation"]["depths"]:
            for weight in config["adaptation"]["replay_weights"]:
                name = f"{mode}_d{depth}_r{rank}_w{weight:g}"
                specification = {
                    "mode": mode,
                    "depth": depth,
                    "rank": rank,
                    "replay_weight": weight,
                }
                patch, result, _ = train_and_evaluate(
                    model,
                    codes,
                    groups,
                    caches,
                    requests,
                    specification,
                    config,
                    seed,
                    directory,
                    name,
                    provenance,
                    preparation["seconds"],
                )
                candidates[mode].append(
                    {
                        "name": name,
                        "specification": specification,
                        "utility": validation_utility(result["metrics"], frozen["metrics"]),
                        "metrics": result["metrics"],
                        "parameters": result["training"]["parameters"],
                    }
                )
                del patch
    selected = {}
    for mode in candidates:
        viable = [entry for entry in candidates[mode] if entry["utility"] >= 0]
        if not viable:
            raise RuntimeError(f"No {mode} candidate respects the validation old-item floor")
        selected[mode] = max(viable, key=lambda entry: (entry["utility"], -entry["parameters"]))
    vectors = np.load(data / "embeddings.npy")
    content_options = []
    for mode in ("mean", "last"):
        predictions, seconds = content_predictions(
            requests["history"], vectors, groups, device, mode=mode
        )
        result = {
            "metrics": aggregate_metrics(predictions, requests["target"], requests["group"]),
            "inference_seconds": seconds,
        }
        save_evaluation(
            directory / f"content_{mode}",
            result,
            {
                "predictions": predictions,
                "target": requests["target"],
                "group": requests["group"],
                "user": requests["user"],
            },
        )
        content_options.append(
            {"mode": mode, "overall_ndcg20": result["metrics"]["overall"]["ndcg@20"]}
        )
    selected["content"] = max(content_options, key=lambda entry: entry["overall_ndcg20"])
    broad, broad_trace = evaluate_model(model, requests, codes, groups, vocab_size, beam_size=200)
    save_evaluation(directory / "frozen_beam200", broad, broad_trace)
    bias_options = []
    for bias in (0.0, 1.0, 2.0, 4.0, 8.0):
        predictions = rerank_new_bias(broad_trace, groups, bias)
        metrics = aggregate_metrics(predictions, requests["target"], requests["group"])
        bias_options.append(
            {
                "bias": bias,
                "utility": validation_utility(metrics, frozen["metrics"]),
                "metrics": metrics,
            }
        )
    selected["bias"] = max(bias_options, key=lambda entry: entry["utility"])
    for label in ("local", "global"):
        setting = selected[label]["specification"]
        few_cache = cache_hidden(model, samples["few"], codes)
        if few_cache is not None:
            few_caches = {"new": merge_caches(caches["new"], few_cache), "old": caches["old"]}
            train_and_evaluate(
                model,
                codes,
                groups,
                few_caches,
                requests,
                setting,
                config,
                seed,
                directory,
                f"{label}_few_shot",
                provenance,
                preparation["seconds"],
            )
        train_and_evaluate(
            model,
            codes,
            groups,
            caches,
            requests,
            {**setting, "replay_weight": 0.0},
            config,
            seed,
            directory,
            f"{label}_no_replay",
            provenance,
            preparation["seconds"],
        )
    selection = {
        "locked": True,
        "selection_seed": seed,
        "tuning_requests": len(requests["target"]),
        "provenance": provenance,
        "configuration": config,
        "rule": (
            "Max new Recall@20 * old NDCG@20, subject to old NDCG >= 90% frozen; "
            "ties: fewer params"
        ),
        "content_rule": "Highest overall validation NDCG@20 among mean and last history queries",
        "bias_candidate_pool": "Natural beam 200 over full catalog; rerank only those items",
        "selected": selected,
        "candidates": candidates,
        "bias_candidates": bias_options,
        "test_used_for_selection": False,
        "budget_note": "Same data source and optimization steps; total parameters are not equal",
    }
    write_json(Path("reports/validation/selection.json"), selection)
    print(
        json.dumps(
            {
                "locked_selection": {
                    name: value.get("name", value.get("mode", value.get("bias")))
                    for name, value in selected.items()
                }
            },
            indent=2,
        )
    )
    return selection


def test_experiment(config_path: Path, data: Path, sid: Path, device: str, seed: int):
    selection_path = Path("reports/validation/selection.json")
    selection = json.loads(selection_path.read_text())
    config, model, codes, provenance = setup_run(config_path, data, sid, seed, device)
    if (
        not selection["locked"]
        or experiment_signature(config) != selection["provenance"]["experiment_sha256"]
    ):
        raise ValueError("Final experiment requires an unchanged, locked validation selection")
    directory = Path(f"reports/test/seed_{seed}")
    directory.mkdir(parents=True, exist_ok=True)
    groups = np.load(data / "test_groups.npy")
    requests = load_requests(data, "test")
    vocab_size = model.specification["vocab_size"]
    frozen, frozen_trace = evaluate_model(model, requests, codes, groups, vocab_size)
    frozen.update({"seed": seed, "provenance": provenance})
    save_evaluation(directory / "frozen", frozen, frozen_trace)
    write_json(
        directory / "prefix_diagnostics.json",
        branch_diagnostics(codes, groups, vocab_size, config["adaptation"]["depths"], frozen_trace),
    )
    samples, preparation = prepare_edits(
        data, Path(f"artifacts/edits/test_seed_{seed}"), groups, config, "test"
    )
    caches = {name: cache_hidden(model, samples[name], codes) for name in ("new", "old")}
    few_cache = cache_hidden(model, samples["few"], codes)
    for mode in ("local", "global"):
        specification = selection["selected"][mode]["specification"]
        for setting_name, training_caches, replay in (
            (mode, caches, specification["replay_weight"]),
            (f"{mode}_no_replay", caches, 0.0),
            (
                f"{mode}_few_shot",
                {"new": merge_caches(caches["new"], few_cache), "old": caches["old"]},
                specification["replay_weight"],
            ),
        ):
            patch, result, trace = train_and_evaluate(
                model,
                codes,
                groups,
                training_caches,
                requests,
                {**specification, "replay_weight": replay},
                config,
                seed,
                directory,
                setting_name,
                provenance,
                preparation["seconds"],
            )
            if mode == "local" and setting_name == "local":
                outside = ~trace["active"]
                result["outside_target_score_max_abs_difference"] = (
                    float(
                        np.max(
                            np.abs(
                                trace["target_log_probs"][outside]
                                - frozen_trace["target_log_probs"][outside]
                            )
                        )
                    )
                    if outside.any()
                    else None
                )
                write_json(directory / f"{setting_name}.json", result)
            del patch
    vectors = np.load(data / "embeddings.npy")
    content, seconds = content_predictions(
        requests["history"], vectors, groups, device, mode=selection["selected"]["content"]["mode"]
    )
    save_evaluation(
        directory / "content",
        {
            "metrics": aggregate_metrics(content, requests["target"], requests["group"]),
            "inference_seconds": seconds,
        },
        {
            "predictions": content,
            "target": requests["target"],
            "group": requests["group"],
            "user": requests["user"],
        },
    )
    popular = popularity_predictions(
        Path("data/processed/software"),
        data,
        config["base_cutoff"],
        groups,
        len(requests["target"]),
    )
    save_evaluation(
        directory / "popularity",
        {
            "metrics": aggregate_metrics(popular, requests["target"], requests["group"]),
            "counts_cutoff": config["base_cutoff"],
        },
        {
            "predictions": popular,
            "target": requests["target"],
            "group": requests["group"],
            "user": requests["user"],
        },
    )
    broad, broad_trace = evaluate_model(model, requests, codes, groups, vocab_size, beam_size=200)
    save_evaluation(directory / "frozen_beam200", broad, broad_trace)
    biased = rerank_new_bias(broad_trace, groups, selection["selected"]["bias"]["bias"])
    save_evaluation(
        directory / "new_bias",
        {
            "metrics": aggregate_metrics(biased, requests["target"], requests["group"]),
            "bias": selection["selected"]["bias"]["bias"],
            "candidate_pool": "frozen beam 200",
            "inference_seconds": broad["inference_seconds"],
        },
        {
            "predictions": biased,
            "target": requests["target"],
            "group": requests["group"],
            "user": requests["user"],
        },
    )
    write_json(
        directory / "completed.json",
        {
            "seed": seed,
            "selection_sha256": file_fingerprint(selection_path)["sha256"],
            "provenance": provenance,
            "test_requests": len(requests["target"]),
        },
    )
    return {"seed": seed, "directory": str(directory), "completed": True}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run validation selection and locked final experiments."
    )
    parser.add_argument("stage", choices=("validation", "test"))
    parser.add_argument("--config", type=Path, default=Path("configs/experiment.json"))
    parser.add_argument("--data", type=Path, default=Path("data/experiment"))
    parser.add_argument("--sid", type=Path, default=Path("artifacts/sid"))
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--device", default="cuda:3")
    parser.add_argument("--tuning-limit", type=int, default=4096)
    arguments = parser.parse_args()
    if arguments.stage == "validation":
        validation(
            arguments.config,
            arguments.data,
            arguments.sid,
            arguments.device,
            arguments.tuning_limit,
        )
    else:
        print(
            json.dumps(
                test_experiment(
                    arguments.config,
                    arguments.data,
                    arguments.sid,
                    arguments.device,
                    arguments.seed,
                )
            )
        )


if __name__ == "__main__":
    main()
