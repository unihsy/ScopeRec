import argparse
import json
import time
from pathlib import Path

import duckdb
import numpy as np
import torch

from scoperec.decoding import SIDTree, numpy_keys
from scoperec.metrics import aggregate_metrics
from scoperec.model import history_tokens
from scoperec.patches import load_patch
from scoperec.prepare import file_fingerprint
from scoperec.protocol import timestamp_ms
from scoperec.train import autocast_for, load_base, setup_torch, trim_padding


def load_requests(data: Path, split: str, limit=None):
    dataset = np.load(data / f"{split}.npz")
    return {name: dataset[name][:limit] for name in dataset.files}


def active_targets(targets, codes, branches, vocab_size):
    depth = branches.shape[1]
    return np.isin(numpy_keys(codes[targets, :depth], vocab_size), numpy_keys(branches, vocab_size))


@torch.no_grad()
def evaluate_model(
    model,
    dataset,
    codes,
    groups,
    vocab_size,
    beam_size=50,
    patch=None,
    batch_size=32,
    oracle_depth=0,
):
    device = next(model.parameters()).device
    tree = SIDTree(codes[groups >= 0], np.flatnonzero(groups >= 0), vocab_size, device)
    inputs = history_tokens(dataset["history"], codes)
    predictions = []
    score_batches = []
    survival = []
    target_scores = []
    model.eval()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    for offset in range(0, len(inputs), batch_size):
        token_batch = trim_padding(
            torch.as_tensor(inputs[offset : offset + batch_size], device=device)
        )
        target_batch = torch.as_tensor(
            codes[dataset["target"][offset : offset + batch_size]], device=device
        )
        with autocast_for(device):
            generated = model.generate(
                token_batch,
                tree,
                beam_size=beam_size,
                patch=patch,
                targets=target_batch,
                forced_depth=oracle_depth,
            )
            path_scores = model.path_scores(token_batch, target_batch, tree, patch)
        predictions.append(generated["items"].cpu().numpy())
        score_batches.append(generated["scores"].cpu().numpy())
        survival.append(generated["survival"].cpu().numpy())
        target_scores.append(path_scores.float().cpu().numpy())
    elapsed = time.perf_counter() - started
    predicted = np.concatenate(predictions)
    if predicted.shape[1] < 50:
        predicted = np.pad(predicted, ((0, 0), (0, 50 - predicted.shape[1])), constant_values=-1)
    active = (
        None
        if patch is None
        else active_targets(dataset["target"], codes, patch.branches.cpu().numpy(), vocab_size)
    )
    result = {
        "metrics": aggregate_metrics(predicted, dataset["target"], dataset["group"], active),
        "inference_seconds": elapsed,
        "requests_per_second": len(inputs) / elapsed,
        "catalog_items": int(np.sum(groups >= 0)),
        "beam_size": beam_size,
        "oracle_depth": oracle_depth,
        "peak_cuda_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0,
        "includes_target_scoring": True,
    }
    trace = {
        "predictions": predicted,
        "beam_scores": np.concatenate(score_batches),
        "survival": np.concatenate(survival),
        "target_log_probs": np.concatenate(target_scores),
        "target": dataset["target"],
        "group": dataset["group"],
        "user": dataset["user"],
    }
    if active is not None:
        trace["active"] = active
    return result, trace


@torch.no_grad()
def content_predictions(histories, vectors, groups, device, cutoff=50, mode="mean", bias=0.0):
    embeddings = torch.as_tensor(vectors, device=device)
    available = torch.as_tensor(groups >= 0, device=device)
    new_items = torch.as_tensor(groups == 2, device=device)
    predictions = []
    started = time.perf_counter()
    for offset in range(0, len(histories), 256):
        history = torch.as_tensor(histories[offset : offset + 256], device=device)
        present = history >= 0
        values = embeddings[history.clamp(min=0)]
        if mode == "last":
            last = present.sum(1).clamp(min=1) - 1
            query = values[torch.arange(len(history), device=device), last]
        elif mode == "mean":
            query = (values * present[:, :, None]).sum(1) / present.sum(1).clamp(min=1)[:, None]
        else:
            raise ValueError("Unknown content query mode")
        query = torch.nn.functional.normalize(query, dim=1)
        scores = query @ embeddings.T + bias * new_items[None, :]
        scores.masked_fill_(~available, -torch.inf)
        predictions.append(scores.topk(cutoff, dim=1).indices.cpu().numpy())
    return np.concatenate(predictions), time.perf_counter() - started


def popularity_predictions(processed: Path, data: Path, edit_at: str, groups, count: int):
    with duckdb.connect(config={"threads": 4}) as connection:
        connection.read_parquet(str(processed / "interactions.parquet")).create_view("events")
        connection.read_parquet(str(data / "catalog.parquet")).create_view("catalog")
        items = connection.execute(
            """
            SELECT item_id FROM events JOIN catalog USING (parent_asin)
            WHERE timestamp < ? GROUP BY item_id ORDER BY count(*) DESC, item_id LIMIT 50
        """,
            [timestamp_ms(edit_at)],
        ).fetchall()
    ranked = np.array([row[0] for row in items], dtype=np.int64)
    if not np.all(groups[ranked] >= 0):
        raise ValueError("Popular item outside current catalog")
    return np.broadcast_to(ranked, (count, len(ranked))).copy()


def save_evaluation(output: Path, result: dict, trace: dict):
    output.parent.mkdir(parents=True, exist_ok=True)
    output.with_name(output.name + ".json").write_text(json.dumps(result, indent=2) + "\n")
    np.savez_compressed(output.with_name(output.name + ".npz"), **trace)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a baseline or ScopeRec patch.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/experiment.json"))
    parser.add_argument("--data", type=Path, default=Path("data/experiment"))
    parser.add_argument("--sid", type=Path, default=Path("artifacts/sid"))
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--patch", type=Path)
    parser.add_argument("--device", default="cuda:3")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--oracle-depth", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    setup_torch(17, arguments.device)
    config = json.loads(arguments.config.read_text())
    model, checkpoint = load_base(arguments.checkpoint, arguments.device)
    codes = np.load(arguments.sid / "codes.npy")
    sid_sha = file_fingerprint(arguments.sid / "codes.npy")["sha256"]
    if sid_sha != checkpoint["sid_sha256"]:
        raise ValueError("Checkpoint and SID versions disagree")
    patch = (
        None
        if arguments.patch is None
        else load_patch(
            arguments.patch,
            file_fingerprint(arguments.checkpoint)["sha256"],
            sid_sha,
            arguments.device,
        )
    )
    result, trace = evaluate_model(
        model,
        load_requests(arguments.data, arguments.split, arguments.limit),
        codes,
        np.load(arguments.data / f"{arguments.split}_groups.npy"),
        checkpoint["specification"]["vocab_size"],
        config["beam_size"],
        patch,
        arguments.batch_size,
        arguments.oracle_depth,
    )
    result.update({"split": arguments.split, "seed": checkpoint["seed"]})
    save_evaluation(arguments.output, result, trace)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
