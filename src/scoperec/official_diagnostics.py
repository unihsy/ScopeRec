import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from scoperec.experiment import write_json
from scoperec.genrecedit import optimize_deltas, solve_weight_delta
from scoperec.official_content import ReleasedContentScope
from scoperec.official_evaluate import evaluate_released
from scoperec.official_experiment import ROOT, setup
from scoperec.official_model import ReleasedGenRecEdit
from scoperec.official_tokens import encode_requests
from scoperec.official_train import read_requests
from scoperec.prepare import file_fingerprint
from scoperec.train import trim_padding
from scoperec.upstream import load_reference


def compare_official_editor(base, config, codes):
    official_class, params_class = load_reference(Path("artifacts/official_benchmark/upstream"))
    settings = config["genrecedit"]
    official = official_class(
        params_class(
            pos2layer=settings["pos2layer"],
            v_lr=settings["v_lr"],
            v_num_grad_steps=settings["v_num_grad_steps"],
            v_weight_decay=settings["v_weight_decay"],
            z_vector_max=settings["z_vector_max"],
            cov_lambda=3000,
            covariance_cache_dir="artifacts/official_benchmark/parity_covariance",
        )
    )
    pseudo = read_requests(Path("artifacts/official_benchmark/samples/valid/seed_2024/pseudo.npz"))
    pseudo = {name: values[:8] for name, values in pseudo.items()}
    tokens, labels = encode_requests(pseudo, codes, base.layout)
    inputs = trim_padding(torch.as_tensor(tokens, device=base.device))
    targets = torch.as_tensor(labels, device=base.device)
    requests = [
        {"history": history, "target_sids": target}
        for history, target in zip(inputs.cpu().tolist(), labels.tolist(), strict=True)
    ]
    bundle = SimpleNamespace(model=base)
    result = []
    for position, layer in enumerate(settings["pos2layer"]):
        _, failed, reference_deltas = official.genrecedit_optimize_z_vectors(
            bundle, requests, layer, position, batch_size=8
        )
        deltas, valid = optimize_deltas(base, inputs, targets, position, layer, settings)
        expected_valid = torch.as_tensor(
            [index not in failed for index in range(8)], device=base.device
        )
        if not torch.equal(valid, expected_valid):
            raise AssertionError("Official target optimizer success mask mismatch")
        valid_reference = [delta for delta in reference_deltas if delta is not None]
        difference = 0.0
        if valid_reference:
            expected = torch.stack(valid_reference)
            torch.testing.assert_close(deltas[valid], expected, atol=5e-4, rtol=5e-4)
            difference = float((deltas[valid] - expected).abs().max())
        result.append(
            {
                "position": position,
                "successful": len(valid_reference),
                "requests": 8,
                "success_mask_identical": True,
                "max_delta_abs_difference": difference,
            }
        )
    torch.manual_seed(2024)
    keys = torch.randn(64, 16, device=base.device)
    deltas = torch.randn(64, 128, device=base.device)
    old = torch.randn(128, 16, device=base.device)
    covariance = (old.double().T @ old.double() / len(old)).float()
    official.genrecedit_get_or_compute_cov = lambda **kwargs: covariance
    reference = official.genrecedit_solve_weight_delta(
        bundle, 0, list(deltas), list(keys), list(deltas), 0
    )
    expected = reference["decoder.block.0.layer.2.DenseReluDense.wo.weight"]
    actual, residual = solve_weight_delta(keys, deltas, covariance, 3000)
    torch.testing.assert_close(actual, expected, atol=1e-7, rtol=1e-5)
    return {
        "passed": True,
        "position_checks": result,
        "solve_max_abs_difference": float((actual - expected).abs().max()),
        "solve_relative_residual": residual,
        "reference_manifest": file_fingerprint(
            Path("artifacts/official_benchmark/upstream/source_manifest.json")
        ),
    }


def run(seed, device):
    config, base, codes, provenance = setup(seed, device)
    base.eval().requires_grad_(False)
    report = {"seed": seed, "provenance": provenance, "methods": {}}
    if seed == 2024:
        report["official_editor_parity"] = compare_official_editor(base, config, codes)
    data = Path("data/official_software")
    requests = {name: values[:128] for name, values in read_requests(data / "test.npz").items()}
    inputs, labels = encode_requests(requests, codes, base.layout)
    inputs = torch.as_tensor(inputs, device=base.device)
    labels = torch.as_tensor(labels, device=base.device)
    _, reference = evaluate_released(base, requests, codes)
    with torch.no_grad():
        original_scores = base.path_scores(inputs, labels)
    original_weights = {name: value.cpu().clone() for name, value in base.state_dict().items()}
    directory = ROOT / f"test/seed_{seed}"
    selection = json.loads((ROOT / "selection.json").read_text())
    for name in ("genrecedit_3000", "genrecedit_1000", "scoperec_c"):
        if name.startswith("genrecedit"):
            artifact = torch.load(directory / f"{name}.pt", weights_only=True, map_location="cpu")
            for key in ("base_sha256", "sid_sha256"):
                if artifact["provenance"][key] != provenance[key]:
                    raise ValueError("Saved editor checkpoint mismatch")
            model = ReleasedGenRecEdit(
                base, artifact["position_layers"], list(artifact["weight_deltas"].to(device))
            ).eval()
        else:
            model = ReleasedContentScope(
                base,
                codes,
                np.load(data / "test_groups.npy"),
                np.load(data / "sentence_t5.npy"),
                **selection["selected"]["settings"],
            ).eval()
        _, trace = evaluate_released(model, requests, codes)
        with np.load(directory / f"{name}.npz") as stored:
            np.testing.assert_array_equal(trace["sequences"], stored["sequences"][:128])
        model.enabled = False
        _, restored = evaluate_released(model, requests, codes)
        np.testing.assert_array_equal(restored["sequences"], reference["sequences"])
        with torch.no_grad():
            restored_scores = model.path_scores(inputs, labels)
        torch.testing.assert_close(restored_scores, original_scores, atol=0, rtol=0)
        for key, value in base.state_dict().items():
            if not torch.equal(value.cpu(), original_weights[key]):
                raise AssertionError("Editor mutated shared base weights")
        report["methods"][name] = {
            "predictions_reloaded_identically": True,
            "disabled_predictions_match": True,
            "path_score_rollback_error": float((restored_scores - original_scores).abs().max()),
            "backbone_unchanged": True,
            "checked_requests": len(requests["target"]),
        }
        del model
    output = ROOT / f"diagnostics/seed_{seed}.json"
    write_json(output, report)
    return report


def main():
    parser = argparse.ArgumentParser(
        description="Check released benchmark optimizer parity and rollback."
    )
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--device", default="cuda:3")
    arguments = parser.parse_args()
    print(json.dumps(run(arguments.seed, arguments.device), indent=2))


if __name__ == "__main__":
    main()
