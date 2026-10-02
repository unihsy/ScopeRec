import json
import time
from collections import defaultdict
from pathlib import Path

import faiss
import numpy as np
import torch

from scoperec.decoding import SIDTree, numpy_keys
from scoperec.evaluate import active_targets
from scoperec.model import history_tokens
from scoperec.patches import PrefixPatch
from scoperec.train import autocast_for, trim_padding


def pseudo_requests(
    training: dict,
    vectors: np.ndarray,
    new_items: np.ndarray,
    neighbors: int,
    examples_per_item: int,
    seed: int,
) -> dict:
    faiss.omp_set_num_threads(4)
    source_rows = defaultdict(list)
    for row_index, item in enumerate(training["target"]):
        source_rows[int(item)].append(row_index)
    old_items = np.array(sorted(source_rows), dtype=np.int64)
    if np.intersect1d(old_items, new_items).size:
        raise ValueError("Pseudo request targets must be unseen in the source training data")
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(np.ascontiguousarray(vectors[old_items], dtype=np.float32))
    _, nearest = index.search(
        np.ascontiguousarray(vectors[new_items], dtype=np.float32), min(neighbors, len(old_items))
    )
    rng = np.random.default_rng(seed)
    selected_rows = []
    target_items = []
    for new_item, nearest_rows in zip(new_items, nearest, strict=True):
        available = np.array(
            [row for neighbor in old_items[nearest_rows] for row in source_rows[int(neighbor)]],
            dtype=np.int64,
        )
        chosen = rng.choice(available, min(examples_per_item, len(available)), replace=False)
        selected_rows.extend(chosen.tolist())
        target_items.extend([int(new_item)] * len(chosen))
    selected_rows = np.asarray(selected_rows, dtype=np.int64)
    return {
        "history": training["history"][selected_rows],
        "target": np.asarray(target_items, dtype=np.int32),
        "timestamp": training["timestamp"][selected_rows],
        "user": training["user"][selected_rows],
        "source_target": training["target"][selected_rows],
    }


def prepare_edits(data: Path, output: Path, groups: np.ndarray, config: dict, split: str):
    started = time.perf_counter()
    output.mkdir(parents=True, exist_ok=True)
    training_file = np.load(data / "train.npz")
    training = {key: training_file[key] for key in training_file.files}
    vectors = np.load(data / "embeddings.npy")
    settings = config["adaptation"]
    new_items = np.flatnonzero(groups == 2)
    pseudo = pseudo_requests(
        training,
        vectors,
        new_items,
        settings["neighbors"],
        settings["examples_per_item"],
        config["sampling_seed"],
    )
    np.savez_compressed(output / "pseudo.npz", **pseudo)
    rng = np.random.default_rng(config["sampling_seed"])
    chosen = rng.choice(
        len(training["target"]),
        min(settings["replay_requests"], len(training["target"])),
        replace=False,
    )
    replay = {key: values[chosen] for key, values in training.items()}
    np.savez_compressed(output / "replay.npz", **replay)
    few_file = np.load(data / f"{split}_few_shot.npz")
    few = {key: few_file[key] for key in few_file.files}
    np.savez_compressed(output / "few_shot.npz", **few)
    report = {
        "split": split,
        "pseudo_requests": len(pseudo["target"]),
        "new_items": len(new_items),
        "covered_new_items": len(np.unique(pseudo["target"])),
        "replay_requests": len(chosen),
        "few_shot_requests": len(few["target"]),
        "few_shot_items": len(np.unique(few["target"])),
        "seconds": time.perf_counter() - started,
        "source": "Only pre-base training histories; no validation/test request is used",
        "new_target_real_interactions_in_zero_shot": 0,
    }
    (output / "samples.json").write_text(json.dumps(report, indent=2) + "\n")
    return {"new": pseudo, "old": replay, "few": few}, report


@torch.no_grad()
def cache_hidden(model, requests, codes, batch_size=256):
    started = time.perf_counter()
    model.eval().requires_grad_(False)
    device = next(model.parameters()).device
    inputs = history_tokens(requests["history"], codes)
    hidden_batches = []
    logits_batches = []
    for offset in range(0, len(inputs), batch_size):
        selected_inputs = trim_padding(
            torch.as_tensor(inputs[offset : offset + batch_size], device=device)
        )
        selected_codes = torch.as_tensor(
            codes[requests["target"][offset : offset + batch_size]], device=device
        )
        with autocast_for(device):
            logits, hidden = model.teacher(selected_inputs, selected_codes)
        hidden_batches.append(hidden.float())
        logits_batches.append(logits.float())
    if not len(inputs):
        return None
    cache = {
        "hidden": torch.cat(hidden_batches),
        "logits": torch.cat(logits_batches),
        "codes": torch.as_tensor(codes[requests["target"]], device=device),
        "targets": torch.as_tensor(requests["target"], device=device),
    }
    cache["seconds"] = time.perf_counter() - started
    cache["bytes"] = sum(
        value.numel() * value.element_size()
        for value in cache.values()
        if isinstance(value, torch.Tensor)
    )
    return cache


def merge_caches(first, second):
    if second is None:
        return first
    return {
        name: torch.cat([first[name], second[name]])
        if isinstance(first[name], torch.Tensor)
        else first[name] + second[name]
        for name in first
    }


def cached_suffix_loss(patch, tree, cache, selected):
    targets = cache["codes"][selected]
    losses = []
    for position in range(patch.depth, tree.length):
        prefix = targets[:, :position]
        logits = cache["logits"][selected, position] + patch(
            cache["hidden"][selected, position], prefix
        )
        log_probabilities = tree.log_probabilities(logits, prefix)
        losses.append(-log_probabilities.gather(1, targets[:, position : position + 1]).squeeze(1))
    return torch.stack(losses, dim=1).sum(dim=1).mean()


def fit_patch(
    model,
    codes,
    groups,
    caches,
    depth,
    rank,
    replay_weight,
    settings,
    seed,
    mode="local",
    branches=None,
    existing=None,
):
    torch.manual_seed(seed)
    device = next(model.parameters()).device
    vocab_size = model.specification["vocab_size"]
    if branches is None:
        branches = np.unique(codes[groups == 2, :depth], axis=0)
    patch = (
        existing
        if existing is not None
        else PrefixPatch(model.specification["d_model"], vocab_size, branches, rank, mode).to(
            device
        )
    )
    if patch.depth != depth or patch.mode != mode:
        raise ValueError("Existing patch does not match the requested update")
    tree = SIDTree(codes[groups >= 0], np.flatnonzero(groups >= 0), vocab_size, device)
    new_targets = caches["new"]["targets"].cpu().numpy()
    new_indices = np.flatnonzero(active_targets(new_targets, codes, branches, vocab_size))
    old_targets = caches["old"]["targets"].cpu().numpy()
    if mode == "local":
        old_indices = np.flatnonzero(active_targets(old_targets, codes, branches, vocab_size))
    else:
        old_indices = np.arange(len(old_targets))
    if not len(new_indices):
        raise ValueError("No new training requests in active branches")
    new_indices = torch.as_tensor(new_indices, device=device)
    old_indices = torch.as_tensor(old_indices, device=device)
    optimizer = torch.optim.AdamW(patch.parameters(), lr=settings["learning_rate"], weight_decay=0)
    trace = []
    started = time.perf_counter()
    for step in range(settings["steps"]):
        optimizer.zero_grad(set_to_none=True)
        selected = new_indices[
            torch.randint(len(new_indices), (settings["batch_size"],), device=device)
        ]
        new_loss = cached_suffix_loss(patch, tree, caches["new"], selected)
        old_loss = torch.zeros((), device=device)
        if len(old_indices) and replay_weight:
            selected_old = old_indices[
                torch.randint(len(old_indices), (settings["batch_size"],), device=device)
            ]
            old_loss = cached_suffix_loss(patch, tree, caches["old"], selected_old)
        loss = new_loss + replay_weight * old_loss
        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite adaptation loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(patch.parameters(), 5)
        optimizer.step()
        if step % 50 == 0 or step + 1 == settings["steps"]:
            trace.append({"step": step + 1, "new_nll": new_loss.item(), "old_nll": old_loss.item()})
    seconds = time.perf_counter() - started
    active_items = active_targets(np.arange(len(codes)), codes, branches, vocab_size)
    old_items = (groups >= 0) & (groups != 2)
    report = {
        "depth": depth,
        "rank": rank,
        "mode": mode,
        "replay_weight": replay_weight,
        "steps": settings["steps"],
        "seed": seed,
        "branches": len(branches),
        "parameters": sum(parameter.numel() for parameter in patch.parameters()),
        "optimizer_seconds": seconds,
        "cache_seconds": caches["new"]["seconds"] + caches["old"]["seconds"],
        "cache_bytes": caches["new"]["bytes"] + caches["old"]["bytes"],
        "effective_new_requests": len(new_indices),
        "effective_old_requests": len(old_indices),
        "old_item_coverage": float(active_items[old_items].mean()),
        "old_items_inside": int(np.sum(active_items & old_items)),
        "training_trace": trace,
    }
    return patch.eval(), report


def branch_diagnostics(codes, groups, vocab_size, depths, trace):
    new_mask = trace["group"] == 2
    old_mask = ~new_mask
    report = {}
    for depth in depths:
        branches = np.unique(codes[groups == 2, :depth], axis=0)
        covered = active_targets(
            np.flatnonzero((groups >= 0) & (groups != 2)), codes, branches, vocab_size
        )
        covered_requests = active_targets(trace["target"], codes, branches, vocab_size)
        prefix_mass = np.exp(trace["target_log_probs"][:, :depth].sum(axis=1))
        base_keys = numpy_keys(codes[groups == 0, :depth], vocab_size)
        new_keys = numpy_keys(codes[groups == 2, :depth], vocab_size)
        report[str(depth)] = {
            "active_branches": len(branches),
            "old_catalog_coverage": float(covered.mean()),
            "old_request_coverage": float(covered_requests[old_mask].mean()),
            "new_prefix_survival": float(trace["survival"][new_mask, depth - 1].mean()),
            "new_target_prefix_mass_mean": float(prefix_mass[new_mask].mean()),
            "new_target_prefix_mass_median": float(np.median(prefix_mass[new_mask])),
            "new_items_with_pre_base_prefix": float(np.isin(new_keys, base_keys).mean()),
        }
    return report
