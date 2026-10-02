import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from transformers import get_scheduler

from scoperec.evaluate import save_evaluation
from scoperec.experiment import write_json
from scoperec.official_evaluate import evaluate_released
from scoperec.official_model import ReleasedTIGER
from scoperec.official_tokens import encode_requests
from scoperec.prepare import file_fingerprint
from scoperec.train import setup_torch, trim_padding


def read_requests(path):
    with np.load(path) as archive:
        return {name: archive[name] for name in archive.files}


def load_released_checkpoint(path, device="cpu"):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    model = ReleasedTIGER(**payload["specification"])
    model.load_state_dict(payload["state"])
    return model.to(device), payload


def train(seed, device, max_steps=None):
    config = json.loads(Path("configs/official_software.json").read_text())
    setup_torch(seed, device)
    torch.set_float32_matmul_precision("highest")
    data = Path("data/official_software")
    sid = Path("artifacts/official_benchmark/sid")
    output = Path(f"artifacts/official_benchmark/baseline/seed_{seed}")
    if max_steps is not None:
        output = Path("artifacts/official_benchmark/smoke")
    if (output / "completed.json").exists():
        raise FileExistsError("Completed official baseline exists; do not overwrite it")
    output.mkdir(parents=True, exist_ok=True)
    codes = np.load(sid / "codes.npy")
    sid_info = json.loads((sid / "sid.json").read_text())
    inputs, labels = encode_requests(read_requests(data / "train.npz"), codes, sid_info["layout"])
    validation = read_requests(data / "valid.npz")
    model = ReleasedTIGER(sid_info["layout"], **config["model"]).to(device)
    settings = config["training"]
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=settings["learning_rate"], weight_decay=settings["weight_decay"]
    )
    steps_per_epoch = math.ceil(len(inputs) / settings["batch_size"])
    total_steps = steps_per_epoch * settings["epochs"]
    scheduler = get_scheduler(
        "cosine",
        optimizer=optimizer,
        num_warmup_steps=settings["warmup_steps"],
        num_training_steps=total_steps,
    )
    history = []
    started = time.perf_counter()
    step = 0
    for epoch in range(settings["epochs"]):
        model.train()
        order = torch.randperm(len(inputs)).numpy()
        summed_loss = 0.0
        examples = 0
        for offset in range(0, len(inputs), settings["batch_size"]):
            selected = order[offset : offset + settings["batch_size"]]
            optimizer.zero_grad(set_to_none=True)
            batch_loss = 0.0
            for micro_offset in range(0, len(selected), settings["micro_batch_size"]):
                rows = selected[micro_offset : micro_offset + settings["micro_batch_size"]]
                batch_inputs = trim_padding(torch.as_tensor(inputs[rows], device=device))
                batch_labels = torch.as_tensor(labels[rows], device=device)
                loss = model(batch_inputs, batch_labels).loss
                if not torch.isfinite(loss):
                    raise RuntimeError("Official baseline loss became non-finite")
                (loss * (len(rows) / len(selected))).backward()
                batch_loss += loss.item() * len(rows)
            torch.nn.utils.clip_grad_norm_(model.parameters(), settings["max_grad_norm"])
            optimizer.step()
            scheduler.step()
            step += 1
            summed_loss += batch_loss
            examples += len(selected)
            if step % 100 == 0:
                print(
                    json.dumps(
                        {
                            "seed": seed,
                            "epoch": epoch + 1,
                            "step": step,
                            "token_nll": batch_loss / len(selected),
                            "lr": scheduler.get_last_lr()[0],
                        }
                    ),
                    flush=True,
                )
            if max_steps is not None and step >= max_steps:
                break
        entry = {
            "epoch": epoch + 1,
            "step": step,
            "token_nll": summed_loss / examples,
            "elapsed_seconds": time.perf_counter() - started,
            "learning_rate": scheduler.get_last_lr()[0],
        }
        if (epoch + 1) % settings["eval_interval"] == 0 or max_steps is not None:
            evaluation_requests = (
                validation
                if max_steps is None
                else {name: values[:64] for name, values in validation.items()}
            )
            result, trace = evaluate_released(model, evaluation_requests, codes)
            entry["validation"] = result["metrics"]
            save_evaluation(output / f"validation_epoch{epoch + 1}", result, trace)
            torch.save(
                {
                    "specification": model.specification,
                    "state": {
                        name: value.detach().cpu() for name, value in model.state_dict().items()
                    },
                    "config": config,
                    "seed": seed,
                    "epoch": epoch + 1,
                    "sid_sha256": sid_info["codes"]["sha256"],
                    "data_sha256": file_fingerprint(data / "train.npz")["sha256"],
                },
                output / "model.pt",
            )
        history.append(entry)
        write_json(output / "training.json", {"epochs": history, "configuration": config})
        print(json.dumps({"seed": seed, **entry}), flush=True)
        if max_steps is not None and step >= max_steps:
            break
    summary = {
        "seed": seed,
        "steps": step,
        "full_steps": total_steps,
        "duration_seconds": time.perf_counter() - started,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "precision": "float32; highest matmul precision",
        "effective_batch_size": settings["batch_size"],
        "micro_batch_size": settings["micro_batch_size"],
        "gradient_accumulation": True,
        "dropout_note": "Microbatching changes dropout RNG consumption, not the loss objective",
        "warmup_completed": step >= settings["warmup_steps"],
        "checkpoint_rule": settings["checkpoint_rule"],
        "smoke_only": max_steps is not None,
        "checkpoint": file_fingerprint(output / "model.pt"),
        "runtime_fixes": [
            "eval_interval missing key",
            "ealuate typo",
            "acceleratorv typo",
            "leval_intervalog typo",
            "microbatch effective batch on 24GB GPU",
        ],
    }
    write_json(output / ("completed.json" if max_steps is None else "smoke.json"), summary)
    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Train released Software baseline with explicit runtime fixes."
    )
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-steps", type=int)
    arguments = parser.parse_args()
    print(json.dumps(train(arguments.seed, arguments.device, arguments.max_steps), indent=2))


if __name__ == "__main__":
    main()
