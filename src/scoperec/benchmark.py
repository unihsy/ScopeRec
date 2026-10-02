import argparse
import json
import time
from pathlib import Path

import faiss
import numpy as np
import pyarrow as pa
import pyarrow.parquet as parquet
import torch

from scoperec.adapt import cache_hidden, fit_patch, pseudo_requests
from scoperec.decoding import SIDTree
from scoperec.experiment import write_json
from scoperec.features import encode_static_text, item_text
from scoperec.model import history_tokens
from scoperec.patches import load_patch, save_patch
from scoperec.prepare import file_fingerprint
from scoperec.train import autocast_for, load_base, setup_torch, trim_padding


def synchronize(device):
    if str(device).startswith("cuda"):
        torch.cuda.synchronize(torch.device(device))


@torch.no_grad()
def serving_benchmark(model, patch, codes, groups, histories, device, repetitions=7):
    tree = SIDTree(
        codes[groups >= 0], np.flatnonzero(groups >= 0), model.specification["vocab_size"], device
    )
    tokens = trim_padding(torch.as_tensor(history_tokens(histories, codes), device=device))
    model.eval()
    records = {}
    for name, current in (("frozen", None), ("local", patch)):
        with autocast_for(device):
            model.generate(tokens, tree, patch=current)
        synchronize(device)
        if str(device).startswith("cuda"):
            torch.cuda.reset_peak_memory_stats(torch.device(device))
        durations = []
        for _ in range(repetitions):
            started = time.perf_counter()
            with autocast_for(device):
                model.generate(tokens, tree, patch=current)
            synchronize(device)
            durations.append(time.perf_counter() - started)
        records[name] = {
            "batch_size": len(tokens),
            "repetitions": repetitions,
            "batch_ms_median": float(np.median(durations) * 1000),
            "batch_ms_p95": float(np.quantile(durations, 0.95) * 1000),
            "requests_per_second": float(len(tokens) / np.median(durations)),
            "peak_cuda_bytes": torch.cuda.max_memory_allocated(torch.device(device))
            if str(device).startswith("cuda")
            else 0,
        }
    return records


def benchmark_update(device):
    setup_torch(17, device)
    synchronize(device)
    started = time.perf_counter()
    costs = {}
    stage_started = started
    config = json.loads(Path("configs/experiment.json").read_text())
    selection = json.loads(Path("reports/validation/selection.json").read_text())
    data = Path("data/experiment")
    vectors = np.load(data / "embeddings.npy")
    groups = np.load(data / "test_groups.npy")
    codes = np.load("artifacts/sid/codes.npy")
    new_items = np.flatnonzero(groups == 2)
    catalog = parquet.read_table(data / "catalog.parquet")
    texts = [item_text(row) for row in catalog.take(pa.array(new_items)).to_pylist()]
    costs["load_data_seconds"] = time.perf_counter() - stage_started
    stage_started = time.perf_counter()
    from sentence_transformers import SentenceTransformer

    encoder = SentenceTransformer("artifacts/text_encoder", local_files_only=True, device=device)
    encoder.max_seq_length = config["encoder"]["max_tokens"]
    new_vectors = encode_static_text(encoder, texts, 256)
    synchronize(device)
    costs["load_encoder_and_encode_new_seconds"] = time.perf_counter() - stage_started
    maximum_embedding_difference = float(np.max(np.abs(vectors[new_items] - new_vectors)))
    del encoder
    if str(device).startswith("cuda"):
        torch.cuda.empty_cache()
    stage_started = time.perf_counter()
    transform = faiss.read_VectorTransform("artifacts/sid/pca.faiss")
    quantizer = faiss.read_index("artifacts/sid/quantizer.faiss")
    semantic = faiss.unpack_bitstrings(
        quantizer.rq.compute_codes(transform.apply(new_vectors)),
        config["sid"]["levels"],
        config["sid"]["bits"],
    )
    offsets = 3 + np.arange(config["sid"]["levels"]) * 2 ** config["sid"]["bits"]
    semantic_matches = bool(np.array_equal(semantic + offsets, codes[new_items, :-1]))
    if not semantic_matches:
        raise RuntimeError("Recomputed SID disagrees with the frozen catalog; refuse update")
    costs["load_quantizer_and_encode_new_seconds"] = time.perf_counter() - stage_started
    stage_started = time.perf_counter()
    archive = np.load(data / "train.npz")
    training = {name: archive[name] for name in archive.files}
    pseudo = pseudo_requests(
        training,
        vectors,
        new_items,
        config["adaptation"]["neighbors"],
        config["adaptation"]["examples_per_item"],
        config["sampling_seed"],
    )
    rng = np.random.default_rng(config["sampling_seed"])
    replay_rows = rng.choice(
        len(training["target"]), config["adaptation"]["replay_requests"], replace=False
    )
    replay = {name: values[replay_rows] for name, values in training.items()}
    costs["neighbor_search_and_samples_seconds"] = time.perf_counter() - stage_started
    stage_started = time.perf_counter()
    checkpoint_path = Path("artifacts/baseline/seed_17/model.pt")
    model, checkpoint = load_base(checkpoint_path, device)
    model.eval().requires_grad_(False)
    synchronize(device)
    costs["load_base_seconds"] = time.perf_counter() - stage_started
    stage_started = time.perf_counter()
    caches = {"new": cache_hidden(model, pseudo, codes), "old": cache_hidden(model, replay, codes)}
    synchronize(device)
    costs["cache_generation_seconds"] = time.perf_counter() - stage_started
    stage_started = time.perf_counter()
    specification = selection["selected"]["local"]["specification"]
    patch, training_cost = fit_patch(
        model,
        codes,
        groups,
        caches,
        specification["depth"],
        specification["rank"],
        specification["replay_weight"],
        config["adaptation"],
        17,
    )
    synchronize(device)
    costs["patch_build_and_optimization_seconds"] = time.perf_counter() - stage_started
    stage_started = time.perf_counter()
    path = Path("artifacts/benchmark/local.pt")
    base_sha = file_fingerprint(checkpoint_path)["sha256"]
    save_patch(patch, path, base_sha, checkpoint["sid_sha256"])
    restored = load_patch(path, base_sha, checkpoint["sid_sha256"], device)
    synchronize(device)
    costs["save_load_and_verify_seconds"] = time.perf_counter() - stage_started
    total = time.perf_counter() - started
    del caches, patch
    if str(device).startswith("cuda"):
        torch.cuda.empty_cache()
    histories = np.load(data / "test.npz")["history"]
    serving = {
        "batch_1": serving_benchmark(model, restored, codes, groups, histories[:1], device),
        "batch_32": serving_benchmark(model, restored, codes, groups, histories[:32], device),
    }
    report = {
        "device": device,
        "new_items": len(new_items),
        "stages": costs,
        "total_update_seconds": total,
        "parameters": training_cost["parameters"],
        "cache_bytes": training_cost["cache_bytes"],
        "patch_bytes": path.stat().st_size,
        "max_embedding_recompute_difference": maximum_embedding_difference,
        "recomputed_semantic_codes_match": semantic_matches,
        "serving": serving,
        "timing_cuda_synchronized": True,
        "cold_scope": "From local data/model artifacts through encoding, edit, save and reload",
        "excluded": [
            "network downloads",
            "base training",
            "old-item vector/index preparation",
            "Python process import time",
            "raw-data extraction and temporal preparation",
        ],
        "serving_excludes": ["tokenization", "tree construction", "target/oracle scoring"],
    }
    write_json(Path("reports/cost_benchmark.json"), report)
    return report


def main():
    parser = argparse.ArgumentParser(description="Measure synchronized update and serving costs.")
    parser.add_argument("--device", default="cuda:3")
    arguments = parser.parse_args()
    print(json.dumps(benchmark_update(arguments.device), indent=2))


if __name__ == "__main__":
    main()
