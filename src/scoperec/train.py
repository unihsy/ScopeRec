import argparse
import json
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as parquet
import torch

from scoperec.decoding import SIDTree
from scoperec.model import GenerativeRecommender, history_tokens
from scoperec.prepare import file_fingerprint
from scoperec.protocol import timestamp_ms


def setup_torch(seed: int, device: str):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("high")
    if device.startswith("cuda"):
        torch.cuda.set_device(torch.device(device))
        torch.cuda.reset_peak_memory_stats()


def trim_padding(inputs: torch.Tensor):
    length = max(1, int(inputs.ne(0).sum(dim=1).max().item()))
    return inputs[:, :length]


def autocast_for(device):
    device = torch.device(device)
    return torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda")


def load_base(path: Path, device="cpu"):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    model = GenerativeRecommender(**checkpoint["specification"])
    model.load_state_dict(checkpoint["state"])
    return model.to(device), checkpoint


@torch.no_grad()
def heldout_loss(model, inputs, targets, tree, batch_size=256):
    model.eval()
    total = 0.0
    for offset in range(0, len(inputs), batch_size):
        selected_inputs = trim_padding(inputs[offset : offset + batch_size])
        selected_targets = targets[offset : offset + batch_size]
        with autocast_for(inputs.device):
            losses = -model.path_scores(selected_inputs, selected_targets, tree).sum(dim=1)
        total += losses.sum().item()
    return total / len(inputs)


def train_baseline(
    config: dict,
    data: Path,
    sid: Path,
    output: Path,
    seed: int,
    device: str,
    max_steps: int | None = None,
):
    setup_torch(seed, device)
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    codes = np.load(sid / "codes.npy")
    sid_info = json.loads((sid / "sid.json").read_text())
    first_seen = parquet.read_table(data / "catalog.parquet", columns=["first_seen"])[
        "first_seen"
    ].to_numpy()
    warm = np.flatnonzero(first_seen < timestamp_ms(config["base_cutoff"]))
    tree = SIDTree(codes[warm], warm, sid_info["vocab_size"], device)
    arrays = {}
    for split in ("train", "holdout"):
        dataset = np.load(data / f"{split}.npz")
        arrays[split] = (
            torch.as_tensor(history_tokens(dataset["history"], codes), device=device),
            torch.as_tensor(codes[dataset["target"]], device=device),
        )
    inputs, targets = arrays["train"]
    model = GenerativeRecommender(sid_info["vocab_size"], **config["model"]).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config["training"]["learning_rate"], weight_decay=0.01
    )
    batch_size = config["training"]["batch_size"]
    total_steps = int(np.ceil(len(inputs) / batch_size)) * config["training"]["epochs"]
    history = []
    best = float("inf")
    step = 0
    for epoch in range(config["training"]["epochs"]):
        model.train()
        order = torch.randperm(len(inputs), device=device)
        epoch_loss = 0.0
        examples = 0
        for offset in range(0, len(inputs), batch_size):
            selected = order[offset : offset + batch_size]
            optimizer.zero_grad(set_to_none=True)
            with autocast_for(device):
                scores = model.path_scores(trim_padding(inputs[selected]), targets[selected], tree)
                loss = -scores.sum(dim=1).mean()
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite baseline loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            step += 1
            warmup = min(1.0, step / 100)
            decay = max(0.1, 1 - step / max(total_steps, 1))
            for parameter_group in optimizer.param_groups:
                parameter_group["lr"] = config["training"]["learning_rate"] * warmup * decay
            optimizer.step()
            epoch_loss += loss.item() * len(selected)
            examples += len(selected)
            if step % 250 == 0:
                print(
                    f"seed={seed} epoch={epoch + 1} step={step} nll={loss.item():.4f}", flush=True
                )
            if max_steps is not None and step >= max_steps:
                break
        validation = heldout_loss(model, *arrays["holdout"], tree)
        entry = {
            "epoch": epoch + 1,
            "step": step,
            "train_nll": epoch_loss / examples,
            "holdout_nll": validation,
            "elapsed_seconds": time.perf_counter() - started,
        }
        history.append(entry)
        print(json.dumps({"seed": seed, **entry}), flush=True)
        if validation < best:
            best = validation
            torch.save(
                {
                    "specification": model.specification,
                    "state": {name: value.cpu() for name, value in model.state_dict().items()},
                    "configuration": config,
                    "seed": seed,
                    "best_epoch": epoch + 1,
                    "sid_sha256": file_fingerprint(sid / "codes.npy")["sha256"],
                },
                output / "model.pt",
            )
        (output / "training.json").write_text(json.dumps(history, indent=2) + "\n")
        if max_steps is not None and step >= max_steps:
            break
    report = {
        "seed": seed,
        "device": device,
        "best_holdout_nll": best,
        "training_seconds": time.perf_counter() - started,
        "steps": step,
        "trainable_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "peak_cuda_bytes": torch.cuda.max_memory_allocated() if device.startswith("cuda") else 0,
        "checkpoint": file_fingerprint(output / "model.pt"),
        "smoke_only": max_steps is not None,
        "configuration": config,
    }
    (output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a compact TIGER-style temporal baseline.")
    parser.add_argument("--config", type=Path, default=Path("configs/experiment.json"))
    parser.add_argument("--data", type=Path, default=Path("data/experiment"))
    parser.add_argument("--sid", type=Path, default=Path("artifacts/sid"))
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--max-steps", type=int)
    arguments = parser.parse_args()
    output = arguments.output or Path(f"artifacts/baseline/seed_{arguments.seed}")
    report = train_baseline(
        json.loads(arguments.config.read_text()),
        arguments.data,
        arguments.sid,
        output,
        arguments.seed,
        arguments.device,
        arguments.max_steps,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
