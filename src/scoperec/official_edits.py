import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from scoperec.experiment import write_json
from scoperec.genrecedit import edit_module, optimize_deltas, solve_weight_delta
from scoperec.official_model import ReleasedGenRecEdit
from scoperec.official_tokens import encode_requests
from scoperec.official_train import load_released_checkpoint, read_requests
from scoperec.prepare import file_fingerprint
from scoperec.train import setup_torch, trim_padding


def build_pseudo(training, evaluation, vectors, train_mask, neighbors, per_item, seed):
    cold_targets = list(dict.fromkeys(evaluation["target"][evaluation["group"] == 2].tolist()))
    available = np.flatnonzero(train_mask)
    normalized = vectors / np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12)
    reverse = defaultdict(list)
    mapping = {}
    for target in cold_targets:
        similarities = normalized[available] @ normalized[target]
        count = min(neighbors, len(available))
        selected = np.argpartition(-similarities, count - 1)[:count]
        selected = selected[np.argsort(-similarities[selected], kind="stable")]
        similar = available[selected].tolist()
        mapping[int(target)] = similar
        for item in similar:
            reverse[item].append(target)
    source_rows = {}
    for row, target in enumerate(training["target"]):
        for new_target in reverse[int(target)]:
            source_rows.setdefault(new_target, []).append(row)
    rng = np.random.default_rng(seed)
    selected_rows = []
    targets = []
    for target, rows in source_rows.items():
        chosen = (
            rng.choice(len(rows), per_item, replace=False)
            if len(rows) > per_item
            else range(len(rows))
        )
        selected_rows.extend(rows[int(index)] for index in chosen)
        targets.extend([target] * len(chosen))
    selected_rows = np.asarray(selected_rows, dtype=np.int64)
    result = {name: values[selected_rows] for name, values in training.items()}
    result["source_target"] = result["target"].copy()
    result["source_training_row"] = selected_rows
    result["target"] = np.asarray(targets, dtype=np.int32)
    result["group"] = np.full(len(targets), 2, dtype=np.int8)
    return result, mapping


def prepare_samples(split, seed):
    config = json.loads(Path("configs/official_software.json").read_text())
    data = Path("data/official_software")
    output = Path(f"artifacts/official_benchmark/samples/{split}/seed_{seed}")
    output.mkdir(parents=True, exist_ok=True)
    if (output / "samples.json").exists():
        return read_requests(output / "pseudo.npz"), json.loads(
            (output / "samples.json").read_text()
        )
    started = time.perf_counter()
    training = read_requests(data / "train.npz")
    evaluation = read_requests(data / f"{split}.npz")
    vectors = np.load(data / "sentence_t5.npy")
    train_mask = np.load(data / "quantizer_train_mask.npy")
    pseudo, mapping = build_pseudo(
        training,
        evaluation,
        vectors,
        train_mask,
        config["genrecedit"]["neighbors"],
        config["genrecedit"]["examples_per_item"],
        seed,
    )
    np.savez_compressed(output / "pseudo.npz", **pseudo)
    write_json(output / "neighbors.json", mapping)
    report = {
        "split": split,
        "seed": seed,
        "pseudo_requests": len(pseudo["target"]),
        "target_items": len(mapping),
        "covered_targets": len(np.unique(pseudo["target"])),
        "self_neighbor_targets": sum(target in items for target, items in mapping.items()),
        "duration_seconds": time.perf_counter() - started,
        "training_rows": len(training["target"]),
        "configuration": config["genrecedit"],
        "sampling_note": "Official default seed=None is nondeterministic; explicit run seed used",
        "neighbor_order_note": "Deterministic item-ID order replaces Python set order",
        "target_visibility": "Cold identities come from the evaluated split, as in released code",
        "source": file_fingerprint(output / "pseudo.npz"),
    }
    write_json(output / "samples.json", report)
    return pseudo, report


@torch.no_grad()
def covariance_moments(model, tokens, labels, layers, batch_size=256):
    captured = {}
    handles = []
    device = model.device
    for position, layer in enumerate(layers):

        def capture(module, arguments, output, position=position):
            captured[position] = arguments[0][:, position].detach().double()

        handles.append(edit_module(model, layer).register_forward_hook(capture))
    width = model.t5.config.d_ff
    moments = [torch.zeros(width, width, dtype=torch.float64, device=device) for _ in layers]
    try:
        for offset in range(0, len(tokens), batch_size):
            inputs = trim_padding(
                torch.as_tensor(tokens[offset : offset + batch_size], device=device)
            )
            targets = torch.as_tensor(labels[offset : offset + batch_size], device=device)
            model(inputs, targets)
            for position in range(len(layers)):
                moments[position].add_(captured[position].T @ captured[position])
            if offset % (batch_size * 200) == 0:
                print(f"Covariance examples {offset + len(inputs)}/{len(tokens)}", flush=True)
    finally:
        for handle in handles:
            handle.remove()
    return [(moment / len(tokens)).float().cpu() for moment in moments]


def collect_official_statistics(model, pseudo, codes, config, output, provenance, seed):
    output.mkdir(parents=True, exist_ok=True)
    settings = config["genrecedit"]
    layers = settings["pos2layer"]
    model.eval().requires_grad_(False)
    covariance_path = output.parent.parent / f"covariance_seed{seed}.pt"
    if covariance_path.exists():
        cached = torch.load(covariance_path, weights_only=True, map_location="cpu")
        if cached["base_sha256"] != provenance["base_sha256"]:
            raise ValueError("Covariance cache checkpoint mismatch")
        moments = cached["moments"]
        covariance_seconds = cached["seconds"]
    else:
        started = time.perf_counter()
        training = read_requests(Path("data/official_software/train.npz"))
        count = min(settings["covariance_requests"], len(training["target"]))
        generator = torch.Generator().manual_seed(seed)
        rows = torch.randperm(len(training["target"]), generator=generator)[:count].numpy()
        training = {name: values[rows] for name, values in training.items()}
        tokens, labels = encode_requests(training, codes, model.layout)
        moments = covariance_moments(model, tokens, labels, layers, settings["batch_size"])
        covariance_seconds = time.perf_counter() - started
        covariance_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "moments": moments,
                "seconds": covariance_seconds,
                "base_sha256": provenance["base_sha256"],
                "examples": count,
                "sampling": "One fixed random 400k sample shared across position moments",
            },
            covariance_path,
        )
    tokens, labels = encode_requests(pseudo, codes, model.layout)
    statistics = []
    for position, layer in enumerate(layers):
        path = output / f"position_{position}.pt"
        if path.is_file():
            cached = torch.load(path, weights_only=True, map_location="cpu")
            if cached["provenance"] != provenance:
                raise ValueError("Stale official editing cache")
            statistics.append(cached["statistics"])
            continue
        started = time.perf_counter()
        accepted_keys = []
        accepted_deltas = []
        captured = {}

        def capture(module, arguments, output):
            captured["key"] = arguments[0][:, -1].detach().float().cpu()

        for offset in range(0, len(tokens), settings["batch_size"]):
            inputs = trim_padding(
                torch.as_tensor(
                    tokens[offset : offset + settings["batch_size"]], device=model.device
                )
            )
            targets = torch.as_tensor(
                labels[offset : offset + settings["batch_size"]], device=model.device
            )
            handle = edit_module(model, layer).register_forward_hook(capture)
            try:
                with torch.no_grad():
                    encoder = model.t5.encoder(
                        input_ids=inputs, attention_mask=inputs.ne(0)
                    ).last_hidden_state
                    model.decode_logits(encoder, inputs.ne(0), targets[:, :position])
            finally:
                handle.remove()
            keys = captured["key"]
            delta, success = optimize_deltas(model, inputs, targets, position, layer, settings)
            accepted_keys.append(keys[success.cpu()])
            accepted_deltas.append(delta[success].cpu())
        entry = {
            "position": position,
            "layer": layer,
            "keys": torch.cat(accepted_keys),
            "deltas": torch.cat(accepted_deltas),
            "covariance": moments[position],
            "requests": len(tokens),
            "seconds": time.perf_counter() - started,
        }
        torch.save({"statistics": entry, "provenance": provenance}, path)
        print(
            json.dumps(
                {
                    "position": position,
                    "successful": len(entry["keys"]),
                    "requests": len(tokens),
                    "seconds": entry["seconds"],
                }
            ),
            flush=True,
        )
        statistics.append(entry)
    return statistics, covariance_seconds


def fit_released_editor(base, statistics, weight):
    updates = []
    checks = []
    for entry in statistics:
        if len(entry["keys"]):
            delta, residual = solve_weight_delta(
                entry["keys"].to(base.device),
                entry["deltas"].to(base.device),
                entry["covariance"].to(base.device),
                weight,
            )
        else:
            delta = torch.zeros_like(edit_module(base, entry["layer"]).weight)
            residual = 0.0
        updates.append(delta.to(base.device))
        checks.append(
            {
                "position": entry["position"],
                "successful": len(entry["keys"]),
                "requests": entry["requests"],
                "solve_relative_residual": residual,
            }
        )
    return ReleasedGenRecEdit(base, [entry["layer"] for entry in statistics], updates), checks


def main():
    parser = argparse.ArgumentParser(
        description="Fit released benchmark GenRecEdit edit statistics."
    )
    parser.add_argument("--split", choices=("valid", "test"), default="valid")
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--device", default="cuda:3")
    arguments = parser.parse_args()
    setup_torch(arguments.seed, arguments.device)
    torch.set_float32_matmul_precision("highest")
    config = json.loads(Path("configs/official_software.json").read_text())
    checkpoint = Path(f"artifacts/official_benchmark/baseline/seed_{arguments.seed}/model.pt")
    base, saved = load_released_checkpoint(checkpoint, arguments.device)
    codes = np.load("artifacts/official_benchmark/sid/codes.npy")
    pseudo, samples = prepare_samples(arguments.split, arguments.seed)
    provenance = {
        "base_sha256": file_fingerprint(checkpoint)["sha256"],
        "sid_sha256": saved["sid_sha256"],
        "samples_sha256": samples["source"]["sha256"],
        "split": arguments.split,
        "seed": arguments.seed,
    }
    positions, seconds = collect_official_statistics(
        base,
        pseudo,
        codes,
        config,
        Path(f"artifacts/official_benchmark/edits/{arguments.split}/seed_{arguments.seed}"),
        provenance,
        arguments.seed,
    )
    print(
        json.dumps(
            {
                "split": arguments.split,
                "seed": arguments.seed,
                "samples": samples,
                "covariance_seconds": seconds,
                "successful_per_position": [len(entry["keys"]) for entry in positions],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
