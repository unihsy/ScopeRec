import argparse
import json
import time
from collections import Counter
from pathlib import Path

import faiss
import numpy as np
import pyarrow.parquet as parquet

from scoperec.protocol import timestamp_ms


def unique_codes(semantic: np.ndarray, codebook_size: int, capacity: int) -> np.ndarray:
    if semantic.ndim != 2 or np.any(semantic < 0) or np.any(semantic >= codebook_size):
        raise ValueError("Invalid semantic code array")
    counts = Counter()
    suffixes = np.empty((len(semantic), 1), dtype=np.int64)
    for item_index, code in enumerate(semantic):
        key = tuple(code)
        suffixes[item_index, 0] = counts[key]
        counts[key] += 1
        if counts[key] > capacity:
            raise ValueError("Fixed collision capacity exhausted; do not silently drop items")
    offsets = 3 + np.arange(semantic.shape[1], dtype=np.int64) * codebook_size
    return np.concatenate(
        [semantic.astype(np.int64) + offsets, suffixes + 3 + semantic.shape[1] * codebook_size],
        axis=1,
    )


def train_quantizer(vectors: np.ndarray, train_mask: np.ndarray, config: dict, seed: int):
    faiss.omp_set_num_threads(4)
    vectors = np.ascontiguousarray(vectors, dtype=np.float32)
    training = np.ascontiguousarray(vectors[train_mask])
    if len(training) < 2 ** config["bits"]:
        raise ValueError("Too few training items for the codebook")
    transform = faiss.PCAMatrix(vectors.shape[1], config["dimensions"])
    transform.train(training)
    projected = transform.apply(vectors)
    index = faiss.IndexResidualQuantizer(config["dimensions"], config["levels"], config["bits"])
    index.rq.cp.seed = seed
    index.rq.cp.niter = 20
    index.rq.max_beam_size = 4
    index.train(np.ascontiguousarray(projected[train_mask]))
    encoded = index.rq.compute_codes(projected)
    semantic = faiss.unpack_bitstrings(encoded, config["levels"], config["bits"])
    reconstruction = index.rq.decode(encoded)
    errors = np.mean((projected - reconstruction) ** 2, axis=1)
    return transform, index, semantic, errors


def build_ids(data: Path, output: Path, config: dict) -> dict:
    started = time.perf_counter()
    output.mkdir(parents=True, exist_ok=True)
    catalog = parquet.read_table(data / "catalog.parquet")
    first_seen = catalog["first_seen"].to_numpy()
    if np.any(first_seen[1:] < first_seen[:-1]):
        raise ValueError("Catalog must be in arrival order to preserve existing collision IDs")
    train_mask = first_seen < timestamp_ms(config["base_cutoff"])
    vectors = np.load(data / "embeddings.npy")
    if len(vectors) != len(catalog):
        raise ValueError("Catalog and embeddings are misaligned")
    transform, index, semantic, errors = train_quantizer(
        vectors, train_mask, config["sid"], config["sampling_seed"]
    )
    size = 2 ** config["sid"]["bits"]
    codes = unique_codes(semantic, size, config["sid"]["collision_capacity"])
    faiss.write_VectorTransform(transform, str(output / "pca.faiss"))
    faiss.write_index(index, str(output / "quantizer.faiss"))
    np.save(output / "codes.npy", codes)
    levels = config["sid"]["levels"]
    report = {
        "algorithm": "Faiss PCA + beam residual quantization + append-only collision token",
        "configuration": config["sid"], "training_cutoff": config["base_cutoff"],
        "training_items": int(train_mask.sum()), "encoded_items": len(codes),
        "vocab_size": 3 + levels * size + config["sid"]["collision_capacity"],
        "sid_length": int(codes.shape[1]),
        "semantic_unique": int(len(np.unique(semantic, axis=0))),
        "complete_unique": int(len(np.unique(codes, axis=0))),
        "max_collision_bucket": int(codes[:, -1].max() - 3 - levels * size + 1),
        "train_reconstruction_mse": float(errors[train_mask].mean()),
        "future_reconstruction_mse": float(errors[~train_mask].mean()),
        "seconds": time.perf_counter() - started,
        "prefix_counts": {
            str(depth): int(len(np.unique(codes[:, :depth], axis=0)))
            for depth in range(1, codes.shape[1])
        },
    }
    (output / "sid.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit SID quantizers on pre-cutoff items only.")
    parser.add_argument("--config", type=Path, default=Path("configs/experiment.json"))
    parser.add_argument("--data", type=Path, default=Path("data/experiment"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/sid"))
    arguments = parser.parse_args()
    print(json.dumps(build_ids(
        arguments.data, arguments.output, json.loads(arguments.config.read_text())
    ), indent=2))


if __name__ == "__main__":
    main()