import argparse
import json
import os
import time
from importlib.metadata import version
from pathlib import Path

import faiss
import numpy as np
import pyarrow.parquet as parquet
from sklearn.decomposition import PCA

from scoperec.experiment import write_json
from scoperec.official_tokens import released_codes, token_layout
from scoperec.prepare import file_fingerprint


def fit_released_quantizer(vectors, train_mask, dimensions=128, levels=3, bits=8, seed=2024):
    np.random.seed(seed)
    faiss.omp_set_num_threads(32)
    pca = PCA(n_components=dimensions, whiten=True)
    projected = np.ascontiguousarray(pca.fit_transform(vectors), dtype=np.float32)
    index = faiss.IndexResidualQuantizer(dimensions, levels, bits, faiss.METRIC_INNER_PRODUCT)
    index.train(projected[train_mask])
    index.add(projected)
    encoded = index.rq.compute_codes(projected)
    semantic = faiss.unpack_bitstrings(encoded, levels, bits)
    return pca, projected, index, semantic


def prepare_features(download=False, device="cuda:3"):
    config = json.loads(Path("configs/official_software.json").read_text())
    data = Path("data/official_software")
    output = Path("artifacts/official_benchmark/sid")
    model_path = Path("artifacts/official_benchmark/sentence_t5")
    if download:
        os.environ.setdefault("HF_HOME", str(Path(".cache/huggingface").resolve()))
        os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
        from huggingface_hub import snapshot_download

        snapshot_download(
            config["encoder"]["model_id"],
            revision=config["encoder"]["revision"],
            endpoint=config["encoder"]["endpoint"],
            local_dir=model_path,
            allow_patterns=[
                "*.json",
                "model.safetensors",
                "pytorch_model.bin",
                "spiece.model",
                "vocab.txt",
                "1_Pooling/*",
                "2_Dense/*",
                "3_Normalize/*",
                "README.md",
            ],
            max_workers=3,
        )
    catalog = parquet.read_table(data / "catalog.parquet")
    if not (data / "sentence_t5.npy").exists():
        import torch
        from sentence_transformers import SentenceTransformer

        started = time.perf_counter()
        torch.set_num_threads(4)
        if device.startswith("cuda"):
            torch.cuda.set_device(torch.device(device))
        torch.set_float32_matmul_precision("highest")
        encoder = SentenceTransformer(str(model_path), device=device, local_files_only=True)
        vectors = encoder.encode(
            catalog["text"].to_pylist(),
            convert_to_numpy=True,
            show_progress_bar=True,
            batch_size=config["encoder"]["batch_size"],
            device=device,
        ).astype(np.float32)
        if vectors.shape != (len(catalog), 768) or not np.isfinite(vectors).all():
            raise ValueError("Unexpected Sentence-T5 embeddings")
        np.save(data / "sentence_t5.npy", vectors)
        write_json(
            data / "sentence_t5.json",
            {
                "encoder": config["encoder"],
                "shape": list(vectors.shape),
                "seconds": time.perf_counter() - started,
                "float32_matmul_precision": "highest",
                "max_seq_length": encoder.max_seq_length,
                "model_files": [
                    file_fingerprint(path)
                    for path in sorted(model_path.rglob("*"))
                    if path.is_file() and ".cache" not in path.parts
                ],
                "embeddings": file_fingerprint(data / "sentence_t5.npy"),
            },
        )
        del encoder
    else:
        vectors = np.load(data / "sentence_t5.npy")
    if (output / "sid.json").exists():
        return json.loads((output / "sid.json").read_text())
    started = time.perf_counter()
    train_mask = np.load(data / "quantizer_train_mask.npy")
    pca, projected, index, semantic = fit_released_quantizer(
        vectors,
        train_mask,
        config["sid"]["dimensions"],
        config["sid"]["levels"],
        config["sid"]["bits"],
        config["seeds"][0],
    )
    codes = released_codes(semantic, 2 ** config["sid"]["bits"])
    layout = token_layout(2 ** config["sid"]["bits"], config["sid"]["levels"])
    output.mkdir(parents=True, exist_ok=True)
    np.save(output / "codes.npy", codes)
    np.save(output / "projected.npy", projected)
    np.savez_compressed(
        output / "pca.npz",
        mean=pca.mean_,
        components=pca.components_,
        explained_variance=pca.explained_variance_,
    )
    faiss.write_index(index, str(output / "quantizer.faiss"))
    report = {
        "configuration": config["sid"],
        "layout": layout,
        "items": len(codes),
        "semantic_unique": len(np.unique(semantic, axis=0)),
        "complete_unique": len(np.unique(codes, axis=0)),
        "max_collision_bucket": int(codes[:, -1].max() - 769),
        "rq_training_items": int(train_mask.sum()),
        "pca_training_items": len(vectors),
        "faiss_version": version("faiss-cpu"),
        "sklearn_version": version("scikit-learn"),
        "pca_solver": pca._fit_svd_solver,
        "rq_max_beam_size": index.rq.max_beam_size,
        "rq_iterations": index.rq.cp.niter,
        "rq_seed": index.rq.cp.seed,
        "seconds": time.perf_counter() - started,
        "codes": file_fingerprint(output / "codes.npy"),
        "catalog": file_fingerprint(data / "catalog.parquet"),
        "embedding_file": file_fingerprint(data / "sentence_t5.npy"),
        "training_mask": file_fingerprint(data / "quantizer_train_mask.npy"),
        "semantics": "PCA sees all catalog items; RQ sees filtered training sequences only",
    }
    write_json(output / "sid.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(
        description="Prepare released Sentence-T5/PCA/Faiss SID features."
    )
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--device", default="cuda:3")
    arguments = parser.parse_args()
    print(json.dumps(prepare_features(arguments.download, arguments.device), indent=2))


if __name__ == "__main__":
    main()
