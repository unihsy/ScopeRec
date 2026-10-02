import time

import numpy as np
import torch

from scoperec.official_tokens import encode_requests, metrics, resolve_items
from scoperec.train import trim_padding


@torch.no_grad()
def evaluate_released(model, requests, codes, batch_size=32, beam_size=50):
    model.eval()
    inputs, labels = encode_requests(requests, codes, model.layout)
    predictions = []
    sequences = []
    scores = []
    started = time.perf_counter()
    for offset in range(0, len(inputs), batch_size):
        batch = trim_padding(
            torch.as_tensor(inputs[offset : offset + batch_size], device=model.device)
        )
        generated = model.generate(batch, beam_size=beam_size)
        predictions.append(generated["codes"].cpu().numpy())
        sequences.append(generated["sequences"].cpu().numpy())
        scores.append(generated["scores"].cpu().numpy())
    elapsed = time.perf_counter() - started
    predicted = np.concatenate(predictions)
    item_ids = resolve_items(predicted, codes)
    result = {
        "metrics": metrics(predicted, labels[:, :4], requests["group"]),
        "inference_seconds": elapsed,
        "beam_size": beam_size,
        "invalid_full_sid_fraction": float(np.mean(item_ids < 0)),
        "eos_at_fifth_fraction": float(
            np.mean(np.concatenate(sequences)[:, :, 4] == model.layout["eos"])
        ),
        "unmasked_full_vocabulary": True,
        "metric_note": "Released metric uses first three SID tokens; exact metric uses all four",
    }
    trace = {
        "codes": predicted,
        "sequences": np.concatenate(sequences),
        "scores": np.concatenate(scores),
        "predictions": item_ids,
        "target": requests["target"],
        "group": requests["group"],
        "user": requests["user"],
        "source_row": requests["source_row"],
    }
    return result, trace
