import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as parquet
import torch

from scoperec.experiment import experiment_signature
from scoperec.official_content import ReleasedContentScope
from scoperec.official_experiment import ROOT, setup
from scoperec.official_model import ReleasedGenRecEdit
from scoperec.official_tokens import encode_requests, resolve_items
from scoperec.official_train import read_requests
from scoperec.prepare import file_fingerprint
from scoperec.recommend import resolve_history
from scoperec.train import trim_padding

SERVING_METHODS = ("scoperec_c", "tiger", "genrecedit_3000", "genrecedit_1000")


@torch.no_grad()
def recommend(
    method, seed, device, request_index=0, identifiers=None, top_k=10, disable_edit=False
):
    if method not in SERVING_METHODS:
        raise ValueError("Unknown released-benchmark method")
    if not 1 <= top_k <= 50:
        raise ValueError("top-k must be between 1 and 50")
    config, base, codes, provenance = setup(seed, device)
    selection_path = ROOT / "selection.json"
    selection = json.loads(selection_path.read_text())
    if not selection["locked"] or selection["configuration_sha256"] != experiment_signature(config):
        raise ValueError("Model settings do not match the locked benchmark configuration")
    directory = ROOT / f"test/seed_{seed}"
    reference = json.loads((directory / f"{method}.json").read_text())
    for name in ("base_sha256", "sid_sha256", "configuration_sha256"):
        if reference["provenance"][name] != provenance[name]:
            raise ValueError(f"Saved benchmark version mismatch: {name}")
    if reference["provenance"]["selection_sha256"] != file_fingerprint(selection_path)["sha256"]:
        raise ValueError("Model selection changed after evaluation")
    data = Path("data/official_software")
    catalog = parquet.read_table(data / "catalog.parquet").to_pylist()
    if identifiers is None:
        requests = read_requests(data / "test.npz")
        if not 0 <= request_index < len(requests["target"]):
            raise ValueError("Request index outside the saved benchmark")
        requests = {
            name: values[request_index : request_index + 1] for name, values in requests.items()
        }
    else:
        requests = {
            "history": resolve_history(identifiers, catalog, config["history_items"]),
            "user": np.array([0]),
            "target": np.array([0]),
        }
    model = base
    if method == "scoperec_c":
        for key, path in (
            ("embeddings_sha256", data / "sentence_t5.npy"),
            ("groups_sha256", data / "test_groups.npy"),
        ):
            if reference[key] != file_fingerprint(path)["sha256"]:
                raise ValueError(f"Content index version mismatch: {key}")
        model = ReleasedContentScope(
            base,
            codes,
            np.load(data / "test_groups.npy"),
            np.load(data / "sentence_t5.npy"),
            **selection["selected"]["settings"],
        ).eval()
    elif method.startswith("genrecedit"):
        payload = torch.load(directory / f"{method}.pt", map_location="cpu", weights_only=True)
        if payload["provenance"] != reference["provenance"]:
            raise ValueError("GenRecEdit weights do not match the evaluated version")
        model = ReleasedGenRecEdit(
            base, payload["position_layers"], list(payload["weight_deltas"].to(device))
        ).eval()
    if method != "tiger":
        model.enabled = not disable_edit
    tokens, _ = encode_requests(requests, codes, model.layout)
    generated = model.generate(trim_padding(torch.as_tensor(tokens, device=device)), beam_size=50)
    predicted_codes = generated["codes"].cpu().numpy()
    item_ids = resolve_items(predicted_codes, codes)
    recommendations = []
    for rank, item in enumerate(item_ids[0, :top_k], start=1):
        row = {
            "rank": rank,
            "valid_catalog_item": bool(item >= 0),
            "sid": predicted_codes[0, rank - 1].tolist(),
            "log_probability": float(generated["scores"][0, rank - 1]),
        }
        if item >= 0:
            row.update(
                {"parent_asin": catalog[item]["parent_asin"], "title": catalog[item]["title"]}
            )
        recommendations.append(row)
    return {
        "method": method,
        "seed": seed,
        "editing_enabled": method != "tiger" and not disable_edit,
        "protocol": "Released Software; full-vocabulary search, invalid paths retained",
        "history": [
            {"parent_asin": catalog[item]["parent_asin"], "title": catalog[item]["title"]}
            for item in requests["history"][0]
            if item >= 0
        ],
        "recommendations": recommendations,
        "scope": "Offline experiment, not a live service",
    }


def main():
    parser = argparse.ArgumentParser(
        description="Run the final released-benchmark recommendation demo."
    )
    parser.add_argument("--method", choices=SERVING_METHODS, default="scoperec_c")
    parser.add_argument("--seed", type=int, choices=(2024, 2025, 2026), default=2024)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--request", type=int, default=0)
    parser.add_argument("--history", nargs="+")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--disable-edit", action="store_true")
    arguments = parser.parse_args()
    print(
        json.dumps(
            recommend(
                arguments.method,
                arguments.seed,
                arguments.device,
                arguments.request,
                arguments.history,
                arguments.top_k,
                arguments.disable_edit,
            ),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
