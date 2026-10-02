import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from scoperec.evaluate import save_evaluation
from scoperec.experiment import experiment_signature, write_json
from scoperec.official_content import ReleasedContentScope
from scoperec.official_edits import (
    collect_official_statistics,
    fit_released_editor,
    prepare_samples,
)
from scoperec.official_evaluate import evaluate_released
from scoperec.official_train import load_released_checkpoint, read_requests
from scoperec.prepare import file_fingerprint
from scoperec.train import setup_torch

ROOT = Path("reports/official_benchmark")
METHODS = ("tiger", "genrecedit_3000", "genrecedit_1000", "scoperec_c", "uniform_ablation")


def setup(seed, device):
    config = json.loads(Path("configs/official_software.json").read_text())
    setup_torch(seed, device)
    torch.set_float32_matmul_precision("highest")
    checkpoint_path = Path(f"artifacts/official_benchmark/baseline/seed_{seed}/model.pt")
    base, checkpoint = load_released_checkpoint(checkpoint_path, device)
    sid_path = Path("artifacts/official_benchmark/sid/codes.npy")
    if file_fingerprint(sid_path)["sha256"] != checkpoint["sid_sha256"]:
        raise ValueError("Official SID/checkpoint mismatch")
    return (
        config,
        base,
        np.load(sid_path),
        {
            "base_sha256": file_fingerprint(checkpoint_path)["sha256"],
            "sid_sha256": checkpoint["sid_sha256"],
            "configuration_sha256": experiment_signature(config),
        },
    )


def obtain_statistics(config, base, codes, split, seed, provenance):
    pseudo, report = prepare_samples(split, seed)
    cache_provenance = {
        "base_sha256": provenance["base_sha256"],
        "sid_sha256": provenance["sid_sha256"],
        "samples_sha256": report["source"]["sha256"],
        "split": split,
        "seed": seed,
    }
    statistics, covariance_seconds = collect_official_statistics(
        base,
        pseudo,
        codes,
        config,
        Path(f"artifacts/official_benchmark/edits/{split}/seed_{seed}"),
        cache_provenance,
        seed,
    )
    return statistics, {
        "samples": report,
        "covariance_seconds": covariance_seconds,
        "target_optimization_seconds": sum(entry["seconds"] for entry in statistics),
    }


def summarize(result):
    return {
        "prefix_overall_n10": result["metrics"]["prefix"]["overall"]["ndcg@10"],
        "prefix_cold_r20": result["metrics"]["prefix"]["cold"]["recall@20"],
        "exact_cold_r20": result["metrics"]["exact"]["cold"]["recall@20"],
        "invalid_fraction": result["invalid_full_sid_fraction"],
    }


def validation(device):
    selection_path = ROOT / "selection.json"
    if selection_path.is_file():
        raise FileExistsError("Official protocol selection already locked")
    if any((ROOT / "test").glob("seed_*/*.json")):
        raise RuntimeError("Cannot choose parameters after final test output exists")
    config, base, codes, provenance = setup(2024, device)
    data = Path("data/official_software")
    requests = read_requests(data / "valid.npz")
    groups = np.load(data / "valid_groups.npy")
    vectors = np.load(data / "sentence_t5.npy")
    directory = ROOT / "validation"
    directory.mkdir(parents=True, exist_ok=True)
    baseline, trace = evaluate_released(base, requests, codes)
    baseline["provenance"] = provenance
    save_evaluation(directory / "tiger", baseline, trace)
    positions, costs = obtain_statistics(config, base, codes, "valid", 2024, provenance)
    for weight in (
        config["genrecedit"]["cov_lambda"],
        config["genrecedit"]["released_script_cov_lambda"],
    ):
        editor, details = fit_released_editor(base, positions, weight)
        result, trace = evaluate_released(editor, requests, codes)
        result.update(
            {
                "provenance": provenance,
                "cov_lambda": weight,
                "editing": costs,
                "position_checks": details,
            }
        )
        save_evaluation(directory / f"genrecedit_{weight}", result, trace)
        print(json.dumps({"method": f"GenRecEdit {weight}", **summarize(result)}), flush=True)
        del editor
    candidates = []
    settings = config["scoperec"]
    for alpha in settings["alphas"]:
        for temperature in settings["temperatures"]:
            for threshold in settings["confidence_thresholds"]:
                chosen = {
                    "depth": settings["depth"],
                    "alpha": alpha,
                    "temperature": temperature,
                    "confidence_threshold": threshold,
                    "confidence_width": settings["confidence_width"],
                }
                name = f"scope_a{alpha:g}_t{temperature:g}_c{threshold:g}".replace(".", "p")
                model = ReleasedContentScope(base, codes, groups, vectors, **chosen).eval()
                result, trace = evaluate_released(model, requests, codes)
                result.update({"provenance": provenance, "settings": chosen})
                save_evaluation(directory / name, result, trace)
                candidates.append({"name": name, "settings": chosen, "metrics": result["metrics"]})
                print(json.dumps({"method": name, **summarize(result)}), flush=True)
                del model
    selected = max(
        candidates,
        key=lambda entry: (
            entry["metrics"]["prefix"]["overall"]["ndcg@10"],
            entry["metrics"]["prefix"]["cold"]["recall@20"],
        ),
    )
    write_json(
        selection_path,
        {
            "locked": True,
            "locked_at": datetime.now(timezone.utc).isoformat(),
            "test_metrics_used": False,
            "configuration": config,
            "configuration_sha256": experiment_signature(config),
            "selected": selected,
            "candidates": candidates,
            "selection_metric": settings["selection"],
            "test_methods": list(METHODS),
            "validation_seed": 2024,
            "official_parity_sha256": file_fingerprint(ROOT / "parity.json")["sha256"],
            "caveat": "Released benchmark uses held-out users and cold-candidate identities",
        },
    )
    return {"locked": True, "selected": selected["name"]}


def final_test(seed, device):
    selection_path = ROOT / "selection.json"
    selection = json.loads(selection_path.read_text())
    config, base, codes, provenance = setup(seed, device)
    if not selection["locked"] or selection["test_metrics_used"]:
        raise ValueError("Official test requires a validation-only locked selection")
    if selection["configuration_sha256"] != experiment_signature(config):
        raise ValueError("Official config changed after selection")
    provenance["selection_sha256"] = file_fingerprint(selection_path)["sha256"]
    provenance["seed"] = seed
    directory = ROOT / f"test/seed_{seed}"
    if (directory / "completed.json").exists():
        raise FileExistsError("Official final result is already complete")
    directory.mkdir(parents=True, exist_ok=True)
    data = Path("data/official_software")
    requests = read_requests(data / "test.npz")
    groups = np.load(data / "test_groups.npy")
    vectors = np.load(data / "sentence_t5.npy")
    result, trace = evaluate_released(base, requests, codes)
    result.update({"provenance": provenance, "method": "tiger"})
    save_evaluation(directory / "tiger", result, trace)
    print(json.dumps({"method": "tiger", **summarize(result)}), flush=True)
    positions, costs = obtain_statistics(config, base, codes, "test", seed, provenance)
    for weight in (
        config["genrecedit"]["cov_lambda"],
        config["genrecedit"]["released_script_cov_lambda"],
    ):
        editor, details = fit_released_editor(base, positions, weight)
        result, trace = evaluate_released(editor, requests, codes)
        result.update(
            {
                "provenance": provenance,
                "cov_lambda": weight,
                "editing": costs,
                "method": f"genrecedit_{weight}",
                "position_checks": details,
            }
        )
        save_evaluation(directory / f"genrecedit_{weight}", result, trace)
        torch.save(
            {
                "weight_deltas": editor.weight_deltas.cpu(),
                "position_layers": editor.position_layers,
                "provenance": provenance,
                "cov_lambda": weight,
            },
            directory / f"genrecedit_{weight}.pt",
        )
        print(json.dumps({"method": f"GenRecEdit {weight}", **summarize(result)}), flush=True)
        del editor
    for name, uniform in (("scoperec_c", False), ("uniform_ablation", True)):
        model = ReleasedContentScope(
            base, codes, groups, vectors, **selection["selected"]["settings"]
        ).eval()
        model.uniform_new_distribution = uniform
        result, trace = evaluate_released(model, requests, codes)
        result.update(
            {
                "provenance": provenance,
                "settings": selection["selected"]["settings"],
                "method": name,
                "uniform": uniform,
                "embeddings_sha256": file_fingerprint(data / "sentence_t5.npy")["sha256"],
                "groups_sha256": file_fingerprint(data / "test_groups.npy")["sha256"],
                "vector_bytes": int(vectors.nbytes),
                "trainable_parameters": 0,
            }
        )
        save_evaluation(directory / name, result, trace)
        print(json.dumps({"method": name, **summarize(result)}), flush=True)
        del model
    write_json(
        directory / "completed.json",
        {
            "provenance": provenance,
            "methods": list(METHODS),
            "requests": len(requests["target"]),
            "completed": True,
        },
    )
    return {"seed": seed, "completed": True, "directory": str(directory)}


def main():
    parser = argparse.ArgumentParser(
        description="Run released Software validation and locked comparisons."
    )
    parser.add_argument("stage", choices=("validation", "test"))
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--device", default="cuda:3")
    arguments = parser.parse_args()
    result = (
        validation(arguments.device)
        if arguments.stage == "validation"
        else final_test(arguments.seed, arguments.device)
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
