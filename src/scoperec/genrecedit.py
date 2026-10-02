import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as functional

from scoperec.adapt import prepare_edits
from scoperec.evaluate import evaluate_model, load_requests, save_evaluation
from scoperec.model import GenerativeRecommender, history_tokens
from scoperec.prepare import file_fingerprint
from scoperec.train import load_base, setup_torch, trim_padding


def edit_module(model, layer):
    return model.t5.decoder.block[layer].layer[2].DenseReluDense.wo


def solve_weight_delta(keys, deltas, covariance, weight):
    keys = keys.double()
    deltas = deltas.double()
    matrix = keys.T @ keys + int(weight) * covariance.double()
    right = deltas.T @ keys
    solution = torch.linalg.solve(matrix.T, right.T).T
    if not torch.isfinite(solution).all():
        raise RuntimeError("Non-finite GenRecEdit linear solve")
    residual = torch.linalg.norm(solution @ matrix - right) / torch.linalg.norm(right).clamp_min(
        1e-30
    )
    return solution.float(), float(residual)


@torch.no_grad()
def collect_keys(model, requests, codes, position, layer, batch_size=256):
    device = next(model.parameters()).device
    input_tokens = history_tokens(requests["history"], codes)
    outputs = []
    captured = {}

    def capture(module, arguments, output):
        captured["keys"] = arguments[0][:, -1].detach().float()

    handle = edit_module(model, layer).register_forward_hook(capture)
    try:
        for offset in range(0, len(input_tokens), batch_size):
            inputs = trim_padding(
                torch.as_tensor(input_tokens[offset : offset + batch_size], device=device)
            )
            prefix = torch.as_tensor(
                codes[requests["target"][offset : offset + batch_size], :position], device=device
            )
            encoder = model.t5.encoder(
                input_ids=inputs, attention_mask=inputs.ne(0)
            ).last_hidden_state
            model.decode_logits(encoder, inputs.ne(0), prefix)
            outputs.append(captured["keys"].cpu())
    finally:
        handle.remove()
    return torch.cat(outputs)


def optimize_deltas(model, inputs, targets, position, layer, settings):
    model.eval().requires_grad_(False)
    device = inputs.device
    with torch.no_grad():
        encoder = model.t5.encoder(input_ids=inputs, attention_mask=inputs.ne(0)).last_hidden_state
    initial = {}
    deltas = torch.zeros(len(inputs), model.t5.config.d_model, device=device, requires_grad=True)
    active = torch.ones(len(inputs), dtype=torch.bool, device=device)
    accepted = torch.zeros_like(deltas)
    success = torch.zeros_like(active)

    def intervene(module, arguments, output):
        if "value" not in initial:
            initial["value"] = output[:, -1].detach().clone()
        changed = output.clone()
        changed[:, -1] = torch.where(active[:, None], initial["value"] + deltas, output[:, -1])
        return changed

    optimizer = torch.optim.Adam([deltas], lr=settings["v_lr"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=settings["v_num_grad_steps"], eta_min=0.01
    )
    handle = edit_module(model, layer).register_forward_hook(intervene)
    try:
        for step in range(settings["v_num_grad_steps"]):
            if not active.any():
                break
            logits, _ = model.decode_logits(encoder, inputs.ne(0), targets[:, :position])
            probabilities = logits.softmax(dim=-1)
            nll = (
                -probabilities.gather(1, targets[:, position : position + 1]).clamp_min(1e-12).log()
            )
            regularizer = settings["v_weight_decay"] * (
                deltas.norm(dim=1) / (initial["value"].norm(dim=1) + 1e-8)
            )
            loss = (nll.squeeze(1) + regularizer)[active].sum()
            optimizer.zero_grad(set_to_none=True)
            if loss.item() > 0:
                loss.backward()
                optimizer.step()
                scheduler.step()
            with torch.no_grad():
                scale = (settings["z_vector_max"] / deltas.norm(dim=1).clamp_min(1e-12)).clamp(
                    max=1
                )
                deltas.mul_(scale[:, None])
                if step > 0 and (step % 10 == 0 or step > settings["v_num_grad_steps"] - 10):
                    finished = active & (logits.argmax(dim=1) == targets[:, position])
                    accepted[finished] = deltas[finished]
                    success |= finished
                    active = active & ~finished
    finally:
        handle.remove()
    return accepted.detach(), success.detach()


def prepare_edit_statistics(model, samples, codes, settings, destination: Path, provenance):
    destination.mkdir(parents=True, exist_ok=True)
    model.eval().requires_grad_(False)
    device = next(model.parameters()).device
    positions = []
    started = time.perf_counter()
    for position, layer in enumerate(settings["pos2layer"]):
        cached_path = destination / f"position_{position}.pt"
        if cached_path.is_file():
            payload = torch.load(cached_path, weights_only=True, map_location="cpu")
            if payload["provenance"] != provenance or payload["settings"] != settings:
                raise ValueError("Stale GenRecEdit statistics; use a new experiment directory")
            positions.append(payload["statistics"])
            continue
        position_started = time.perf_counter()
        keys = collect_keys(model, samples["new"], codes, position, layer, settings["batch_size"])
        old_keys = collect_keys(
            model, samples["old"], codes, position, layer, settings["batch_size"]
        )
        covariance = (old_keys.double().T @ old_keys.double() / len(old_keys)).float()
        tokens = history_tokens(samples["new"]["history"], codes)
        valid_keys = []
        valid_deltas = []
        for offset in range(0, len(tokens), settings["batch_size"]):
            inputs = trim_padding(
                torch.as_tensor(tokens[offset : offset + settings["batch_size"]], device=device)
            )
            targets = torch.as_tensor(
                codes[samples["new"]["target"][offset : offset + settings["batch_size"]]],
                device=device,
            )
            deltas, success = optimize_deltas(model, inputs, targets, position, layer, settings)
            valid_keys.append(keys[offset : offset + len(inputs)][success.cpu()])
            valid_deltas.append(deltas[success].cpu())
            if offset % (settings["batch_size"] * 20) == 0:
                print(
                    f"GenRecEdit pos={position} optimized={offset + len(inputs)}/{len(tokens)}",
                    flush=True,
                )
        statistics = {
            "position": position,
            "layer": layer,
            "keys": torch.cat(valid_keys),
            "deltas": torch.cat(valid_deltas),
            "covariance": covariance,
            "requests": len(tokens),
            "seconds": time.perf_counter() - position_started,
        }
        torch.save(
            {"provenance": provenance, "settings": settings, "statistics": statistics}, cached_path
        )
        print(
            json.dumps(
                {
                    "position": position,
                    "successful_requests": len(statistics["keys"]),
                    "seconds": statistics["seconds"],
                }
            ),
            flush=True,
        )
        positions.append(statistics)
    return positions, time.perf_counter() - started


class GenRecEditModel(GenerativeRecommender):
    def __init__(self, base, position_layers, deltas):
        nn.Module.__init__(self)
        self.t5 = base.t5
        self.specification = base.specification.copy()
        self.position_layers = list(position_layers)
        self.register_buffer("weight_deltas", torch.stack(deltas))
        self.enabled = True

    def decode_logits(self, encoder, attention_mask, prefix):
        position = prefix.shape[1]
        if not self.enabled or position >= len(self.position_layers):
            return super().decode_logits(encoder, attention_mask, prefix)

        def one_one(module, arguments, output):
            changed = output.clone()
            changed[:, -1] += functional.linear(
                arguments[0][:, -1], self.weight_deltas[position].to(arguments[0].dtype)
            )
            return changed

        handle = edit_module(self, self.position_layers[position]).register_forward_hook(one_one)
        try:
            return super().decode_logits(encoder, attention_mask, prefix)
        finally:
            handle.remove()

    def teacher(self, inputs, targets, patch=None):
        if patch is not None:
            raise ValueError("Do not silently combine GenRecEdit and output patches")
        if not self.enabled:
            return super().teacher(inputs, targets)
        encoder = self.t5.encoder(input_ids=inputs, attention_mask=inputs.ne(0)).last_hidden_state
        steps = [
            self.decode_logits(encoder, inputs.ne(0), targets[:, :position])
            for position in range(targets.shape[1])
        ]
        return torch.stack([step[0] for step in steps], dim=1), torch.stack(
            [step[1] for step in steps], dim=1
        )


def construct_editor(base, positions, weight, device):
    deltas = []
    diagnostics = []
    for position in positions:
        if not len(position["keys"]):
            delta = torch.zeros_like(edit_module(base, position["layer"]).weight)
            residual = 0.0
        else:
            delta, residual = solve_weight_delta(
                position["keys"].to(device),
                position["deltas"].to(device),
                position["covariance"].to(device),
                weight,
            )
        deltas.append(delta.to(device))
        diagnostics.append(
            {
                "position": position["position"],
                "layer": position["layer"],
                "successful": len(position["keys"]),
                "requested": position["requests"],
                "solve_relative_residual": residual,
                "delta_frobenius_norm": float(delta.norm()),
            }
        )
    return GenRecEditModel(base, [position["layer"] for position in positions], deltas), diagnostics


def load_editor(base, path: Path, provenance, device):
    artifact = torch.load(path, map_location="cpu", weights_only=True)
    for key in ("base_sha256", "sid_sha256"):
        if artifact["provenance"][key] != provenance[key]:
            raise ValueError("GenRecEdit artifact does not match the base/SID version")
    return (
        GenRecEditModel(
            base, artifact["position_layers"], list(artifact["weight_deltas"].to(device))
        )
        .to(device)
        .eval()
    )


def main():
    parser = argparse.ArgumentParser(
        description="Fit the official-equivalent GenRecEdit FFN algorithm."
    )
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--device", default="cuda:3")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--cov-lambda", type=int, action="append")
    arguments = parser.parse_args()
    setup_torch(arguments.seed, arguments.device)
    torch.set_float32_matmul_precision("highest")
    config = json.loads(Path("configs/experiment_v2.json").read_text())
    data = Path("data/experiment_v2")
    codes = np.load("artifacts/sid/codes.npy")
    checkpoint_path = Path(f"artifacts/v2/baseline/seed_{arguments.seed}/model.pt")
    model, checkpoint = load_base(checkpoint_path, arguments.device)
    model.eval().requires_grad_(False)
    provenance = {
        "base_sha256": file_fingerprint(checkpoint_path)["sha256"],
        "sid_sha256": checkpoint["sid_sha256"],
        "split": arguments.split,
    }
    groups = np.load(data / f"{arguments.split}_groups.npy")
    samples, preparation = prepare_edits(
        data,
        Path(f"artifacts/v2/samples/{arguments.split}/seed_{arguments.seed}"),
        groups,
        config,
        arguments.split,
    )
    positions, cost = prepare_edit_statistics(
        model,
        samples,
        codes,
        config["genrecedit"],
        Path(f"artifacts/v2/genrecedit/{arguments.split}/seed_{arguments.seed}"),
        provenance,
    )
    dataset = load_requests(data, arguments.split, arguments.limit)
    output = Path(f"reports/v2/{arguments.split}/seed_{arguments.seed}")
    output.mkdir(parents=True, exist_ok=True)
    for weight in arguments.cov_lambda or config["genrecedit"]["cov_lambdas"]:
        editor, details = construct_editor(model, positions, weight, arguments.device)
        result, trace = evaluate_model(
            editor, dataset, codes, groups, model.specification["vocab_size"]
        )
        result.update(
            {
                "method": "GenRecEdit FFN (official-equivalent port)",
                "cov_lambda": weight,
                "provenance": provenance,
                "position_diagnostics": details,
                "preparation": preparation,
                "edit_statistics_seconds": sum(position["seconds"] for position in positions),
                "statistics_load_or_compute_seconds": cost,
                "upstream_revision": config["genrecedit"]["upstream_revision"],
                "differences": [
                    "shared temporal samples and fixed SID",
                    "linear solve instead of explicit inverse",
                    "vectorized optimizer, checked against official implementation",
                    "common legal-mask-before-softmax evaluation",
                ],
                "edit_parameters": int(editor.weight_deltas.numel()),
            }
        )
        save_evaluation(output / f"genrecedit_cov{weight}", result, trace)
        torch.save(
            {
                "position_layers": editor.position_layers,
                "weight_deltas": editor.weight_deltas.cpu(),
                "provenance": provenance,
                "cov_lambda": weight,
            },
            output / f"genrecedit_cov{weight}.pt",
        )
        print(json.dumps({"cov_lambda": weight, "metrics": result["metrics"]}), flush=True)


if __name__ == "__main__":
    main()
