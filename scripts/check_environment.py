import argparse
import importlib.metadata
import json
import os
import platform
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def check_ml_runtime() -> dict:
    import torch
    from sentence_transformers import SentenceTransformer
    from transformers import T5Config, T5ForConditionalGeneration

    torch.manual_seed(0)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    configuration = T5Config(
        vocab_size=128,
        d_model=64,
        d_kv=16,
        d_ff=128,
        num_layers=1,
        num_decoder_layers=1,
        num_heads=4,
        decoder_start_token_id=0,
        pad_token_id=0,
        eos_token_id=1,
    )
    model = T5ForConditionalGeneration(configuration).to(device).train()
    inputs = torch.randint(2, 128, (2, 12), device=device)
    labels = torch.randint(2, 128, (2, 4), device=device)
    use_bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16):
        loss = model(input_ids=inputs, labels=labels).loss
    if not torch.isfinite(loss).item():
        raise RuntimeError("T5 forward pass produced a non-finite loss")
    loss.backward()
    gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    if not gradients or not all(torch.isfinite(gradient).all().item() for gradient in gradients):
        raise RuntimeError("T5 backward pass produced missing or non-finite gradients")
    previous_weights = model.shared.weight.detach().clone()
    optimizer.step()
    if torch.equal(previous_weights, model.shared.weight):
        raise RuntimeError("T5 optimizer did not update the model")
    return {
        "device": str(device),
        "bf16_autocast": use_bf16,
        "random_t5_training_step_passed": True,
        "loss": loss.item(),
        "sentence_transformer_class": SentenceTransformer.__name__,
        "weights_downloaded": False,
    }


def check_environment(require_gpus: int) -> dict:
    import torch

    if torch.cuda.device_count() < require_gpus:
        raise RuntimeError(
            f"Expected at least {require_gpus} CUDA devices; "
            f"found {torch.cuda.device_count()}"
        )

    devices = []
    for device_index in range(torch.cuda.device_count()):
        with torch.cuda.device(device_index):
            properties = torch.cuda.get_device_properties(device_index)
            supports_bf16 = torch.cuda.is_bf16_supported()
            dtype = torch.bfloat16 if supports_bf16 else torch.float32
            inputs = torch.ones((256, 256), device=device_index, dtype=dtype)
            result = inputs @ inputs
            torch.cuda.synchronize(device_index)
            if not torch.all(result == 256).item():
                raise RuntimeError(f"Matrix multiplication failed on GPU {device_index}")
            free_bytes, total_bytes = torch.cuda.mem_get_info(device_index)
            devices.append(
                {
                    "index": device_index,
                    "name": properties.name,
                    "compute_capability": list(torch.cuda.get_device_capability(device_index)),
                    "total_bytes": total_bytes,
                    "free_bytes_after_probe": free_bytes,
                    "bf16_supported": supports_bf16,
                    "matmul_dtype": str(dtype),
                    "matmul_passed": True,
                }
            )
            del inputs, result
            torch.cuda.empty_cache()

    packages = {}
    for package_name in (
        "torch", "numpy", "pyarrow", "duckdb", "transformers", "sentence-transformers",
        "accelerate", "huggingface-hub", "requests", "pytest", "ruff",
    ):
        try:
            packages[package_name] = importlib.metadata.version(package_name)
        except importlib.metadata.PackageNotFoundError:
            packages[package_name] = None

    disk = shutil.disk_usage(PROJECT_ROOT)
    return {
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "torch_cuda_version": torch.version.cuda,
        "packages": packages,
        "gpus": devices,
        "project_disk": {"total_bytes": disk.total, "free_bytes": disk.free},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe ScopeRec's Python and CUDA environment.")
    parser.add_argument("--require-gpus", type=int, default=1)
    parser.add_argument("--check-ml", action="store_true")
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    if arguments.require_gpus < 0:
        parser.error("--require-gpus must be nonnegative")
    report = check_environment(arguments.require_gpus)
    if arguments.check_ml:
        report["ml_runtime"] = check_ml_runtime()
    payload = json.dumps(report, indent=2) + "\n"
    if arguments.output is not None:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(payload, encoding="utf-8")
    print(payload, end="")


if __name__ == "__main__":
    main()