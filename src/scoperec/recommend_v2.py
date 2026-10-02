import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as parquet
import torch

from scoperec.compare_v2 import setup_comparison
from scoperec.content_scope import load_content_artifact
from scoperec.decoding import SIDTree
from scoperec.genrecedit import load_editor
from scoperec.model import history_tokens
from scoperec.patches import load_patch
from scoperec.prepare import file_fingerprint
from scoperec.recommend import resolve_history
from scoperec.train import autocast_for, trim_padding

SERVING_METHODS = (
    "tiger",
    "genrecedit",
    "genrecedit_balanced",
    "scoperec_v1",
    "scoperec_content",
    "scoperec_content_balanced",
)


def load_serving_model(method, seed, device):
    if method not in SERVING_METHODS:
        raise ValueError(f"Unknown v2 method: {method}")
    config, base, codes, provenance = setup_comparison(seed, device)
    torch.set_float32_matmul_precision("highest")
    data = Path("data/experiment_v2")
    groups = np.load(data / "test_groups.npy")
    patch = None
    model = base
    if method.startswith("genrecedit"):
        model = load_editor(
            base, Path(f"reports/v2/test/seed_{seed}/{method}.pt"), provenance, device
        )
    elif method.startswith("scoperec_content"):
        vectors = np.load(data / "embeddings.npy")
        provenance.update(
            {
                "embeddings_sha256": file_fingerprint(data / "embeddings.npy")["sha256"],
                "catalog_groups_sha256": file_fingerprint(data / "test_groups.npy")["sha256"],
            }
        )
        model = load_content_artifact(
            base,
            Path(f"artifacts/v2/content/seed_{seed}/{method}.json"),
            codes,
            groups,
            vectors,
            provenance,
        )
    elif method == "scoperec_v1":
        patch = load_patch(
            Path(f"reports/v2/test/seed_{seed}/scoperec_v1.pt"),
            provenance["base_sha256"],
            provenance["sid_sha256"],
            device,
        )
    return config, model, patch, codes, groups


@torch.no_grad()
def recommend(method, seed, device, request_index, identifiers, cutoff, disabled):
    if cutoff < 1 or cutoff > 50:
        raise ValueError("top-k must be between 1 and 50")
    config, model, patch, codes, groups = load_serving_model(method, seed, device)
    data = Path("data/experiment_v2")
    catalog = parquet.read_table(data / "catalog.parquet").to_pylist()
    if identifiers is not None:
        history = resolve_history(identifiers, catalog, config["history_items"])
    else:
        requests = np.load(data / "test.npz")
        if not 0 <= request_index < len(requests["history"]):
            raise ValueError("Request index outside the v2 confirmation set")
        history = requests["history"][request_index : request_index + 1]
    if disabled:
        if patch is not None:
            patch.enabled = False
        if hasattr(model, "enabled"):
            model.enabled = False
    tree = SIDTree(
        codes[groups >= 0], np.flatnonzero(groups >= 0), model.specification["vocab_size"], device
    )
    inputs = trim_padding(torch.as_tensor(history_tokens(history, codes), device=device))
    with autocast_for(device):
        generated = model.generate(inputs, tree, beam_size=50, patch=patch)
    output = []
    for rank, item in enumerate(generated["items"][0, :cutoff].cpu().tolist(), start=1):
        if item < 0:
            continue
        output.append(
            {
                "rank": rank,
                "parent_asin": catalog[item]["parent_asin"],
                "title": catalog[item]["title"],
                "new_in_current_batch": bool(groups[item] == 2),
                "sid": codes[item].tolist(),
                "log_probability": float(generated["scores"][0, rank - 1]),
            }
        )
    return {
        "method": method,
        "seed": seed,
        "editing_enabled": method != "tiger" and not disabled,
        "catalog_snapshot": config["windows"]["test"]["edit_at"],
        "history": [
            {"parent_asin": catalog[item]["parent_asin"], "title": catalog[item]["title"]}
            for item in history[0]
            if item >= 0
        ],
        "recommendations": output,
        "scope": "Offline fixed snapshot, not a live store",
    }


def main():
    parser = argparse.ArgumentParser(
        description="Compare v2 recommenders on the same fixed snapshot."
    )
    parser.add_argument("--method", choices=SERVING_METHODS, default="scoperec_content_balanced")
    parser.add_argument("--seed", type=int, choices=(17, 29, 43), default=17)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--request", type=int, default=0)
    parser.add_argument("--history", nargs="+")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--disable-edit", action="store_true")
    arguments = parser.parse_args()
    result = recommend(
        arguments.method,
        arguments.seed,
        arguments.device,
        arguments.request,
        arguments.history,
        arguments.top_k,
        arguments.disable_edit,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
