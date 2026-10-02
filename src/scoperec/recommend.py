import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as parquet
import torch

from scoperec.decoding import SIDTree
from scoperec.model import history_tokens
from scoperec.patches import load_patch
from scoperec.prepare import file_fingerprint
from scoperec.train import autocast_for, load_base, setup_torch, trim_padding


def resolve_history(identifiers, catalog, limit):
    lookup = {record["parent_asin"]: index for index, record in enumerate(catalog)}
    unknown = [identifier for identifier in identifiers if identifier not in lookup]
    if unknown:
        raise ValueError(f"Unknown product identifiers: {unknown}")
    if not identifiers:
        raise ValueError("At least one historical product is required")
    history = np.full((1, limit), -1, dtype=np.int32)
    selected = identifiers[-limit:]
    history[0, : len(selected)] = [lookup[identifier] for identifier in selected]
    return history


@torch.no_grad()
def recommend(checkpoint_path, patch_path, request_index, identifiers, device, cutoff, disabled):
    setup_torch(17, device)
    data = Path("data/experiment")
    catalog = parquet.read_table(data / "catalog.parquet").to_pylist()
    codes = np.load("artifacts/sid/codes.npy")
    groups = np.load(data / "test_groups.npy")
    model, checkpoint = load_base(checkpoint_path, device)
    expected_sid = file_fingerprint(Path("artifacts/sid/codes.npy"))["sha256"]
    if checkpoint["sid_sha256"] != expected_sid:
        raise ValueError("Checkpoint and SID version mismatch")
    if identifiers is not None:
        histories = resolve_history(
            identifiers, catalog, checkpoint["configuration"]["history_items"]
        )
    else:
        requests = np.load(data / "test.npz")
        if not 0 <= request_index < len(requests["history"]):
            raise ValueError("Request index outside the fixed test snapshot")
        histories = requests["history"][request_index : request_index + 1]
    if cutoff < 1 or cutoff > 50:
        raise ValueError("top-k must be between 1 and 50")
    patch = (
        None
        if patch_path is None
        else load_patch(
            patch_path, file_fingerprint(checkpoint_path)["sha256"], expected_sid, device
        )
    )
    if patch is not None:
        patch.enabled = not disabled
    tokens = trim_padding(torch.as_tensor(history_tokens(histories, codes), device=device))
    tree = SIDTree(
        codes[groups >= 0],
        np.flatnonzero(groups >= 0),
        checkpoint["specification"]["vocab_size"],
        device,
    )
    with autocast_for(device):
        generated = model.generate(tokens, tree, beam_size=50, patch=patch)
    recommendations = []
    for rank, item in enumerate(generated["items"][0, :cutoff].cpu().tolist(), start=1):
        if item < 0:
            continue
        recommendations.append(
            {
                "rank": rank,
                "parent_asin": catalog[item]["parent_asin"],
                "title": catalog[item]["title"],
                "is_new": bool(groups[item] == 2),
                "sid": codes[item].tolist(),
                "log_probability": float(generated["scores"][0, rank - 1]),
            }
        )
    return {
        "catalog_snapshot": "2019-07-01",
        "base_seed": checkpoint["seed"],
        "patch_enabled": patch is not None and patch.enabled,
        "history": [
            {"parent_asin": catalog[item]["parent_asin"], "title": catalog[item]["title"]}
            for item in histories[0]
            if item >= 0
        ],
        "recommendations": recommendations,
        "experimental_not_production": True,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Recommend products from a fixed ScopeRec snapshot."
    )
    parser.add_argument(
        "--checkpoint", type=Path, default=Path("artifacts/baseline/seed_17/model.pt")
    )
    parser.add_argument("--patch", type=Path)
    parser.add_argument("--request", type=int, default=0)
    parser.add_argument("--history", nargs="+")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--disable-patch", action="store_true")
    arguments = parser.parse_args()
    print(
        json.dumps(
            recommend(
                arguments.checkpoint,
                arguments.patch,
                arguments.request,
                arguments.history,
                arguments.device,
                arguments.top_k,
                arguments.disable_patch,
            ),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
