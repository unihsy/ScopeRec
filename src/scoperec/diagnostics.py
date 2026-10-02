import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as parquet
import torch

from scoperec.adapt import cache_hidden, fit_patch, merge_caches
from scoperec.decoding import numpy_keys
from scoperec.evaluate import evaluate_model, load_requests, save_evaluation
from scoperec.experiment import setup_run, write_json
from scoperec.metrics import aggregate_metrics
from scoperec.patches import extend_patch, load_patch, save_patch
from scoperec.protocol import timestamp_ms


def read_samples(path):
    with np.load(path) as archive:
        return {name: archive[name] for name in archive.files}


def subset_requests(samples, selected):
    return {name: values[selected] for name, values in samples.items()}


def subset_cache(cache, selected):
    return {
        name: values[selected] if isinstance(values, torch.Tensor) else values
        for name, values in cache.items()
    }


def gate_ablation(config, model, codes, data, provenance, device):
    directory = Path("reports/diagnostics/gate")
    groups = np.load(data / "validation_groups.npy")
    requests = load_requests(data, "validation", 4096)
    pseudo = read_samples(Path("artifacts/edits/validation/pseudo.npz"))
    replay = read_samples(Path("artifacts/edits/validation/replay.npz"))
    keys, counts = np.unique(codes[pseudo["target"], :2], axis=0, return_counts=True)
    branches = keys[[np.argmax(counts)]]
    caches = {"new": cache_hidden(model, pseudo, codes), "old": cache_hidden(model, replay, codes)}
    patch, training = fit_patch(
        model, codes, groups, caches, 2, 8, 1.0, config["adaptation"], 17, branches=branches
    )
    save_patch(
        patch,
        directory / "single_branch.pt",
        provenance["base_sha256"],
        provenance["sid_sha256"],
        training,
    )
    results = {}
    outside_changes = {}
    original_trace = None
    for name, current in (("frozen", None), ("gated", patch), ("ungated", patch)):
        if name == "ungated":
            patch.gate_enabled = False
        result, trace = evaluate_model(
            model, requests, codes, groups, model.specification["vocab_size"], patch=current
        )
        if name == "frozen":
            original_trace = trace
        else:
            outside = ~trace["active"]
            difference = np.abs(
                trace["target_log_probs"][outside] - original_trace["target_log_probs"][outside]
            )
            outside_changes[name] = float(difference.max())
        save_evaluation(directory / name, result, trace)
        results[name] = result["metrics"]
    write_json(
        directory / "summary.json",
        {
            "branch": branches.tolist(),
            "selection": "Most pseudo requests; no evaluation labels",
            "same_weights_for_gate_toggle": True,
            "training": training,
            "metrics": results,
            "outside_path_token_score_max_abs_change": outside_changes,
            "device": device,
            "interpretation": "Scope ablation, not separate training algorithms",
        },
    )
    del caches


def historical_and_oracle(config, model, codes, data, provenance, seed):
    device = next(model.parameters()).device
    directory = Path(f"reports/diagnostics/seed_{seed}")
    groups = np.load(data / "test_groups.npy")
    vocab_size = model.specification["vocab_size"]
    patch = load_patch(
        Path(f"reports/test/seed_{seed}/local.pt"),
        provenance["base_sha256"],
        provenance["sid_sha256"],
        device,
    )
    global_patch = load_patch(
        Path(f"reports/test/seed_{seed}/global.pt"),
        provenance["base_sha256"],
        provenance["sid_sha256"],
        device,
    )
    historical = load_requests(data, "holdout")
    historical["group"] = np.zeros(len(historical["target"]), dtype=np.int8)
    old_catalog = np.where(groups == 0, 0, -1)
    for name, current in (("frozen", None), ("local", patch), ("global", global_patch)):
        result, trace = evaluate_model(
            model, historical, codes, old_catalog, vocab_size, patch=current
        )
        result["catalog"] = "Original base catalog; held-out pre-base users; no future items"
        save_evaluation(directory / f"historical_{name}", result, trace)
    requests = load_requests(data, "test")
    new_requests = subset_requests(requests, requests["group"] == 2)
    for name, current in (("frozen", None), ("local", patch)):
        result, trace = evaluate_model(
            model, new_requests, codes, groups, vocab_size, patch=current, oracle_depth=patch.depth
        )
        result["diagnostic_only"] = "Correct prefix forced; not a deployable recommendation metric"
        save_evaluation(directory / f"oracle_{name}", result, trace)
    sample = subset_requests(requests, np.arange(256))
    baseline, before = evaluate_model(model, sample, codes, groups, vocab_size)
    patch.enabled = False
    restored, after = evaluate_model(model, sample, codes, groups, vocab_size, patch=patch)
    write_json(
        directory / "rollback.json",
        {
            "requests": 256,
            "same_predictions": bool(np.array_equal(before["predictions"], after["predictions"])),
            "max_path_score_change": float(
                np.max(np.abs(before["target_log_probs"] - after["target_log_probs"]))
            ),
            "base_sha256": provenance["base_sha256"],
            "same_history_and_catalog": True,
        },
    )


def sequential_updates(config, model, codes, data, provenance):
    seed = 17
    directory = Path("reports/diagnostics/sequential")
    groups = np.load(data / "test_groups.npy")
    first_seen = parquet.read_table(data / "catalog.parquet", columns=["first_seen"])[
        "first_seen"
    ].to_numpy()
    split_time = timestamp_ms("2019-04-01T00:00:00+00:00")
    first_items = (groups == 2) & (first_seen < split_time)
    second_items = (groups == 2) & ~first_items
    first_groups = groups.copy()
    first_groups[first_seen >= split_time] = -1
    pseudo = read_samples(Path("artifacts/edits/test_seed_17/pseudo.npz"))
    replay = read_samples(Path("artifacts/edits/test_seed_17/replay.npz"))
    first_requests = subset_requests(pseudo, first_items[pseudo["target"]])
    second_requests = subset_requests(pseudo, second_items[pseudo["target"]])
    first_caches = {
        "new": cache_hidden(model, first_requests, codes),
        "old": cache_hidden(model, replay, codes),
    }
    selection = json.loads(Path("reports/validation/selection.json").read_text())
    specification = selection["selected"]["local"]["specification"]
    depth = specification["depth"]
    first_patch, first_training = fit_patch(
        model,
        codes,
        first_groups,
        first_caches,
        depth,
        specification["rank"],
        specification["replay_weight"],
        config["adaptation"],
        seed,
    )
    save_patch(
        first_patch,
        directory / "batch1.pt",
        provenance["base_sha256"],
        provenance["sid_sha256"],
        first_training,
    )
    branches = np.unique(codes[groups == 2, :depth], axis=0)
    half_budget = config["adaptation"]["replay_requests"] // 2
    rng = np.random.default_rng(config["sampling_seed"])
    earlier_rows = rng.choice(
        len(first_requests["target"]),
        min(half_budget, len(first_requests["target"])),
        replace=False,
    )
    warm_count = config["adaptation"]["replay_requests"] - len(earlier_rows)
    replay_cache = merge_caches(
        subset_cache(first_caches["old"], np.arange(warm_count)),
        subset_cache(first_caches["new"], earlier_rows),
    )
    second_cache = cache_hidden(model, second_requests, codes)
    snapshots = {"batch1_only": first_patch}
    training = {"batch1": first_training}
    for label, old_cache in (
        ("with_earlier_replay", replay_cache),
        ("without_earlier_replay", first_caches["old"]),
    ):
        extended = extend_patch(first_patch, branches)
        trained, cost = fit_patch(
            model,
            codes,
            groups,
            {"new": second_cache, "old": old_cache},
            depth,
            specification["rank"],
            specification["replay_weight"],
            config["adaptation"],
            seed,
            branches=branches,
            existing=extended,
        )
        snapshots[label] = trained
        training[label] = cost
        save_patch(
            trained,
            directory / f"{label}.pt",
            provenance["base_sha256"],
            provenance["sid_sha256"],
            cost,
        )
    requests = load_requests(data, "test")
    metrics = {}
    for label, current in snapshots.items():
        result, trace = evaluate_model(
            model, requests, codes, groups, model.specification["vocab_size"], patch=current
        )
        for batch_name, item_mask in (("first_batch", first_items), ("second_batch", second_items)):
            chosen = item_mask[requests["target"]]
            result[batch_name] = aggregate_metrics(
                trace["predictions"][chosen], requests["target"][chosen], requests["group"][chosen]
            )["overall"]
        save_evaluation(directory / label, result, trace)
        metrics[label] = {name: result[name] for name in ("first_batch", "second_batch")}
    first_keys = numpy_keys(codes[first_items, :depth], model.specification["vocab_size"])
    second_keys = numpy_keys(codes[second_items, :depth], model.specification["vocab_size"])
    write_json(
        directory / "summary.json",
        {
            "seed": seed,
            "first_batch_items": int(first_items.sum()),
            "second_batch_items": int(second_items.sum()),
            "first_edit_at": "2019-04-01",
            "second_edit_at": "2019-07-01",
            "fixed_final_catalog_comparison": True,
            "replay_total_requests": config["adaptation"]["replay_requests"],
            "earlier_batch_replay_requests": len(earlier_rows),
            "shared_prefix_count": len(np.intersect1d(first_keys, second_keys)),
            "metrics": metrics,
            "training": training,
            "selection": "Locked validation configuration; post-primary diagnostic only",
            "caveat": "Earlier-new replay uses content pseudo labels, never future labels",
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run mechanism, rollback, and sequential diagnostics."
    )
    parser.add_argument("--config", type=Path, default=Path("configs/experiment.json"))
    parser.add_argument("--data", type=Path, default=Path("data/experiment"))
    parser.add_argument("--sid", type=Path, default=Path("artifacts/sid"))
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--device", default="cuda:3")
    arguments = parser.parse_args()
    config, model, codes, provenance = setup_run(
        arguments.config, arguments.data, arguments.sid, arguments.seed, arguments.device
    )
    if arguments.seed == 17:
        gate_ablation(config, model, codes, arguments.data, provenance, arguments.device)
        sequential_updates(config, model, codes, arguments.data, provenance)
    historical_and_oracle(config, model, codes, arguments.data, provenance, arguments.seed)
    print(json.dumps({"seed": arguments.seed, "diagnostics_complete": True}))


if __name__ == "__main__":
    main()
