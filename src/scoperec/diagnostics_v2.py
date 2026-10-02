import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from scoperec.compare_v2 import setup_comparison
from scoperec.content_scope import load_content_artifact, save_content_artifact
from scoperec.decoding import SIDTree
from scoperec.evaluate import evaluate_model, load_requests, save_evaluation
from scoperec.experiment import write_json
from scoperec.final_v2 import verify_selection
from scoperec.genrecedit import load_editor
from scoperec.model import history_tokens
from scoperec.prepare import file_fingerprint
from scoperec.train import autocast_for, trim_padding


def tensor_fingerprint(model):
    import hashlib

    digest = hashlib.sha256()
    for name, tensor in model.t5.state_dict().items():
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


@torch.no_grad()
def latency(model, codes, groups, histories, batch_size, repetitions=15):
    device = next(model.parameters()).device
    tree = SIDTree(
        codes[groups >= 0], np.flatnonzero(groups >= 0), model.specification["vocab_size"], device
    )
    inputs = trim_padding(
        torch.as_tensor(history_tokens(histories[:batch_size], codes), device=device)
    )
    with autocast_for(device):
        model.generate(inputs, tree, beam_size=50)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    durations = []
    for _ in range(repetitions):
        started = time.perf_counter()
        with autocast_for(device):
            model.generate(inputs, tree, beam_size=50)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        durations.append(time.perf_counter() - started)
    return {
        "batch_size": batch_size,
        "repetitions": repetitions,
        "median_batch_ms": float(np.median(durations) * 1000),
        "p95_batch_ms": float(np.quantile(durations, 0.95) * 1000),
        "peak_cuda_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0,
        "includes": ["encoder", "beam decoding", "content query if applicable"],
        "excludes": ["disk loading", "tree construction", "history tokenization", "target scoring"],
    }


def validation_confirmation(
    base, codes, groups, vectors, selection, requests, provenance, directory
):
    if len(requests["target"]) == 0:
        raise ValueError("No reserved validation requests")
    vocab_size = base.specification["vocab_size"]
    result, trace = evaluate_model(base, requests, codes, groups, vocab_size)
    save_evaluation(directory / "reserved_validation_tiger", result, trace)
    for name in ("scoperec_content", "scoperec_content_balanced"):
        artifact_path = directory / f"reserved_{name}.json"
        save_content_artifact(artifact_path, selection["chosen"][name]["settings"], provenance)
        adapted = load_content_artifact(base, artifact_path, codes, groups, vectors, provenance)
        result, trace = evaluate_model(adapted, requests, codes, groups, vocab_size)
        result["status"] = "Unused validation requests, diagnostic only after choices were locked"
        save_evaluation(directory / f"reserved_validation_{name}", result, trace)


def run(seed, device):
    config, base, codes, provenance = setup_comparison(seed, device)
    selection = verify_selection(config)
    torch.set_float32_matmul_precision("highest")
    data = Path("data/experiment_v2")
    groups = np.load(data / "test_groups.npy")
    vectors = np.load(data / "embeddings.npy")
    requests = load_requests(data, "test", 256)
    provenance.update(
        {
            "embeddings_sha256": file_fingerprint(data / "embeddings.npy")["sha256"],
            "catalog_groups_sha256": file_fingerprint(data / "test_groups.npy")["sha256"],
        }
    )
    directory = Path(f"reports/v2/diagnostics/seed_{seed}")
    directory.mkdir(parents=True, exist_ok=True)
    base_signature = tensor_fingerprint(base)
    _, frozen_trace = evaluate_model(
        base, requests, codes, groups, base.specification["vocab_size"]
    )
    report = {"seed": seed, "methods": {}, "provenance": provenance}
    for name in (
        "genrecedit",
        "genrecedit_balanced",
        "scoperec_content",
        "scoperec_content_balanced",
    ):
        started = time.perf_counter()
        if name.startswith("genrecedit"):
            adapted = load_editor(
                base, Path(f"reports/v2/test/seed_{seed}/{name}.pt"), provenance, device
            )
            artifact_bytes = Path(f"reports/v2/test/seed_{seed}/{name}.pt").stat().st_size
        else:
            artifact_path = Path(f"artifacts/v2/content/seed_{seed}/{name}.json")
            save_content_artifact(artifact_path, selection["chosen"][name]["settings"], provenance)
            adapted = load_content_artifact(base, artifact_path, codes, groups, vectors, provenance)
            artifact_bytes = artifact_path.stat().st_size
        load_seconds = time.perf_counter() - started
        _, actual_trace = evaluate_model(
            adapted, requests, codes, groups, base.specification["vocab_size"]
        )
        expected = np.load(f"reports/v2/test/seed_{seed}/{name}.npz")
        np.testing.assert_array_equal(actual_trace["predictions"], expected["predictions"][:256])
        adapted.enabled = False
        _, restored = evaluate_model(
            adapted, requests, codes, groups, base.specification["vocab_size"]
        )
        np.testing.assert_array_equal(restored["predictions"], frozen_trace["predictions"])
        max_difference = float(
            np.max(np.abs(restored["target_log_probs"] - frozen_trace["target_log_probs"]))
        )
        if max_difference > 1e-5:
            raise AssertionError("Rollback failed to restore baseline path scores")
        if tensor_fingerprint(base) != base_signature:
            raise AssertionError("An editor mutated the common backbone")
        adapted.enabled = True
        entries = {
            "reloaded_predictions_match": True,
            "rollback_predictions_match": True,
            "rollback_max_log_probability_difference": max_difference,
            "backbone_unchanged": True,
            "artifact_bytes": artifact_bytes,
            "load_seconds": load_seconds,
        }
        if name.startswith("scoperec"):
            before_prefix = frozen_trace["target_log_probs"][:, : adapted.depth]
            after_prefix = actual_trace["target_log_probs"][:, : adapted.depth]
            entries["max_pre_gate_log_probability_change"] = float(
                np.max(np.abs(before_prefix - after_prefix))
            )
            rows = np.flatnonzero(requests["group"] != 2)
            alpha = adapted.alpha
            observed_ratio = np.exp(
                actual_trace["target_log_probs"][rows].sum(1)
                - frozen_trace["target_log_probs"][rows].sum(1)
            )
            entries["minimum_old_path_probability_ratio"] = float(observed_ratio.min())
            entries["guaranteed_lower_bound_one_minus_alpha"] = 1 - alpha
            if observed_ratio.min() < 1 - alpha - 2e-5:
                raise AssertionError("Old-path probability budget bound violated")
        if seed == 17:
            entries["latency"] = {
                f"batch_{size}": latency(adapted, codes, groups, requests["history"], size)
                for size in (1, 32)
            }
        report["methods"][name] = entries
        del adapted
    if seed == 17:
        report["tiger_latency"] = {
            f"batch_{size}": latency(base, codes, groups, requests["history"], size)
            for size in (1, 32)
        }
        validation = load_requests(data, "validation")
        validation = {name: values[4096:] for name, values in validation.items()}
        validation_groups = np.load(data / "validation_groups.npy")
        validation_provenance = {
            **provenance,
            "catalog_groups_sha256": file_fingerprint(data / "validation_groups.npy")["sha256"],
        }
        validation_confirmation(
            base,
            codes,
            validation_groups,
            vectors,
            selection,
            validation,
            validation_provenance,
            directory,
        )
    write_json(directory / "integrity_and_latency.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(
        description="Check reloading, rollback and serving of v2 editors."
    )
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--device", default="cuda:3")
    arguments = parser.parse_args()
    print(json.dumps(run(arguments.seed, arguments.device), indent=2))


if __name__ == "__main__":
    main()
