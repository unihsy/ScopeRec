import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as parquet

from scoperec.prepare import file_fingerprint


def item_text(record: dict) -> str:
    return "\n".join(
        str(record.get(field) or "").strip() for field in ("title", "features", "description")
    ).strip()[:10000]


def encode_static_text(model, texts, batch_size, progress=False):
    import torch

    previous_precision = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("highest")
    try:
        return model.encode(
            texts, batch_size=batch_size, show_progress_bar=progress,
            convert_to_numpy=True, normalize_embeddings=True,
        ).astype(np.float32)
    finally:
        torch.set_float32_matmul_precision(previous_precision)


def encode_items(config: dict, data: Path, model_path: Path, device: str, batch_size: int) -> dict:
    from sentence_transformers import SentenceTransformer

    started = time.perf_counter()
    table = parquet.read_table(data / "catalog.parquet")
    texts = [item_text(record) for record in table.to_pylist()]
    if any(not text for text in texts):
        raise ValueError("Empty item text needs an explicit missing-text policy")
    model = SentenceTransformer(str(model_path), device=device, local_files_only=True)
    model.max_seq_length = config["encoder"]["max_tokens"]
    vectors = encode_static_text(model, texts, batch_size, progress=True)
    if not np.isfinite(vectors).all():
        raise ValueError("Text encoder produced non-finite embeddings")
    np.save(data / "embeddings.npy", vectors)
    report = {
        **config["encoder"], "seconds": time.perf_counter() - started,
        "shape": list(vectors.shape), "device": device, "normalized": True,
        "float32_matmul_precision": "highest",
        "metadata_text_fields": ["title", "features", "description"],
        "character_limit_before_tokenization": 10000,
        "model_files": [
            file_fingerprint(model_path / name)
            for name in ("config.json", "model.safetensors", "tokenizer.json")
        ],
        "output": file_fingerprint(data / "embeddings.npy"),
        "caveat": "Pretrained encoder and metadata are not historically versioned at 2018.",
    }
    (data / "embeddings.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Encode item text with a pinned local model.")
    parser.add_argument("--config", type=Path, default=Path("configs/experiment.json"))
    parser.add_argument("--data", type=Path, default=Path("data/experiment"))
    parser.add_argument("--model", type=Path, default=Path("artifacts/text_encoder"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--download", action="store_true")
    arguments = parser.parse_args()
    config = json.loads(arguments.config.read_text())
    if arguments.download:
        os.environ.setdefault("HF_HOME", str(Path(".cache/huggingface").resolve()))
        os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
        from huggingface_hub import snapshot_download

        snapshot_download(
            config["encoder"]["model_id"], revision=config["encoder"]["revision"],
            endpoint=config["encoder"]["endpoint"], local_dir=arguments.model,
            allow_patterns=[
                "*.json", "model.safetensors", "vocab.txt", "1_Pooling/config.json", "README.md"
            ], max_workers=4,
        )
    print(json.dumps(encode_items(
        config, arguments.data, arguments.model, arguments.device, arguments.batch_size
    ), indent=2))


if __name__ == "__main__":
    main()