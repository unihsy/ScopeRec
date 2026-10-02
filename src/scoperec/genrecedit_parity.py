import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from scoperec.genrecedit import optimize_deltas, solve_weight_delta
from scoperec.model import history_tokens
from scoperec.prepare import file_fingerprint
from scoperec.train import load_base, setup_torch, trim_padding
from scoperec.upstream import REVISION, load_reference


def check_reference(device):
    setup_torch(17, device)
    torch.set_float32_matmul_precision("highest")
    root = Path("artifacts/v2/upstream")
    reference_class, hyperparameter_class = load_reference(root)
    config = json.loads(Path("configs/experiment_v2.json").read_text())
    settings = config["genrecedit"]
    hparams = hyperparameter_class(
        pos2layer=settings["pos2layer"],
        v_lr=settings["v_lr"],
        v_num_grad_steps=settings["v_num_grad_steps"],
        v_weight_decay=settings["v_weight_decay"],
        z_vector_max=settings["z_vector_max"],
        covariance_cache_dir="artifacts/v2/parity_covariance",
        cov_lambda=1000,
    )
    official = reference_class(hparams)
    model, _ = load_base(Path("artifacts/v2/baseline/seed_17/model.pt"), device)
    model.eval().requires_grad_(False)
    codes = np.load("artifacts/sid/codes.npy")
    dataset = np.load("data/experiment_v2/train.npz")
    inputs = trim_padding(
        torch.as_tensor(history_tokens(dataset["history"][:8], codes), device=device)
    )
    targets = torch.as_tensor(codes[-8:], device=device)
    requests = [
        {"history": history, "target_sids": target}
        for history, target in zip(inputs.cpu().tolist(), targets.cpu().tolist(), strict=True)
    ]
    bundle = SimpleNamespace(model=SimpleNamespace(t5=model.t5, device=torch.device(device)))
    reports = []
    for position, layer in enumerate(settings["pos2layer"]):
        _, failed, deltas = official.genrecedit_optimize_z_vectors(
            bundle, requests, layer, position, batch_size=8
        )
        actual, success = optimize_deltas(model, inputs, targets, position, layer, settings)
        expected_success = torch.tensor(
            [index not in failed for index in range(len(requests))], device=device
        )
        if not torch.equal(success, expected_success):
            raise AssertionError(
                f"Official and port disagree on successful rows at position {position}"
            )
        valid = [delta for delta in deltas if delta is not None]
        difference = 0.0
        if valid:
            expected = torch.stack(valid)
            torch.testing.assert_close(actual[success], expected, atol=5e-4, rtol=5e-4)
            difference = float((actual[success] - expected).abs().max())
        reports.append(
            {
                "position": position,
                "layer": layer,
                "requests": len(requests),
                "successful": len(valid),
                "max_delta_abs_difference": difference,
                "success_mask_identical": True,
            }
        )
    torch.manual_seed(43)
    keys = torch.randn(24, 16, device=device)
    deltas = torch.randn(24, model.t5.config.d_model, device=device)
    old_keys = torch.randn(50, 16, device=device)
    covariance = (old_keys.double().T @ old_keys.double() / len(old_keys)).float()
    official.genrecedit_get_or_compute_cov = lambda **kwargs: covariance
    expected = official.genrecedit_solve_weight_delta(
        bundle, 0, list(deltas), list(keys), list(deltas), 0
    )["decoder.block.0.layer.2.DenseReluDense.wo.weight"]
    actual, residual = solve_weight_delta(keys, deltas, covariance, 1000)
    torch.testing.assert_close(actual, expected, atol=1e-7, rtol=1e-5)
    report = {
        "passed": True,
        "upstream_revision": REVISION,
        "source_manifest": file_fingerprint(root / "source_manifest.json"),
        "official_functions_executed": [
            "genrecedit_optimize_z_vectors",
            "genrecedit_solve_weight_delta",
        ],
        "position_optimization": reports,
        "solve_max_abs_difference": float((actual - expected).abs().max()),
        "solve_relative_residual": residual,
        "device": device,
        "float32_matmul_precision": "highest",
        "differences": [
            "Complete prefix recomputation instead of upstream incremental cache",
            "Vectorized identical Adam/cosine/regularizer/success-mask operations",
            "torch.linalg.solve instead of explicit inverse",
        ],
        "not_claimed": "Paper split, text encoder, codebook sizes, and numerical table replication",
    }
    output = Path("reports/v2/genrecedit_reference_parity.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(
        description="Compare the port to executed pinned official functions."
    )
    parser.add_argument("--device", default="cuda:3")
    arguments = parser.parse_args()
    print(json.dumps(check_reference(arguments.device), indent=2))


if __name__ == "__main__":
    main()
