from collections import Counter

import numpy as np


def token_layout(codebook_size=256, semantic_levels=3, user_tokens=1):
    base_user = codebook_size * (semantic_levels + 1) + 1
    eos = base_user + user_tokens
    return {
        "padding": 0,
        "decoder_start": 0,
        "user_base": base_user,
        "user_tokens": user_tokens,
        "eos": eos,
        "vocab_size": eos + 1,
        "sid_length": semantic_levels + 1,
    }


def released_codes(semantic, codebook_size=256):
    semantic = np.asarray(semantic, dtype=np.int64)
    if semantic.ndim != 2 or np.any(semantic < 0) or np.any(semantic >= codebook_size):
        raise ValueError("Invalid semantic code array")
    counts = Counter()
    collision = []
    for code in semantic:
        key = tuple(code)
        counts[key] += 1
        if counts[key] > codebook_size:
            raise ValueError("Released collision capacity exceeded")
        collision.append(counts[key])
    raw = np.column_stack([semantic, collision])
    return raw + np.arange(raw.shape[1], dtype=np.int64) * codebook_size + 1


def encode_requests(requests, codes, layout):
    histories = np.asarray(requests["history"])
    present = histories >= 0
    lengths = present.sum(axis=1)
    if np.any(present != (np.arange(histories.shape[1])[None, :] < lengths[:, None])):
        raise ValueError("History padding must be on the right")
    if np.any(histories[present] >= len(codes)):
        raise ValueError("History item outside code table")
    sid_length = codes.shape[1]
    tokens = np.zeros((len(histories), histories.shape[1] * sid_length + 2), dtype=np.int64)
    tokens[:, 0] = layout["user_base"] + requests["user"] % layout["user_tokens"]
    encoded = codes[np.maximum(histories, 0)] * present[:, :, None]
    tokens[:, 1:-1] = encoded.reshape(len(histories), -1)
    tokens[np.arange(len(histories)), 1 + lengths * sid_length] = layout["eos"]
    labels = np.column_stack([codes[requests["target"]], np.full(len(histories), layout["eos"])])
    return tokens, labels.astype(np.int64)


def metric_rows(predicted_codes, target_codes, depth, cutoffs=(10, 20, 50)):
    if predicted_codes.ndim != 3 or len(predicted_codes) != len(target_codes):
        raise ValueError("Predictions must be request x rank x SID")
    if not 1 <= depth <= predicted_codes.shape[-1] or max(cutoffs) > predicted_codes.shape[1]:
        raise ValueError("Invalid metric depth or cutoff")
    hits = np.all(predicted_codes[:, :, :depth] == target_codes[:, None, :depth], axis=-1)
    rank = np.where(hits.any(axis=1), hits.argmax(axis=1) + 1, np.inf)
    return {
        **{f"recall@{cutoff}": (rank <= cutoff).astype(float) for cutoff in cutoffs},
        **{
            f"ndcg@{cutoff}": np.where(rank <= cutoff, 1 / np.log2(rank + 1), 0)
            for cutoff in cutoffs
        },
    }


def metrics(predicted_codes, target_codes, groups, cutoffs=(10, 20, 50)):
    result = {}
    for label, depth in (("prefix", 3), ("exact", 4)):
        values = metric_rows(predicted_codes, target_codes, depth, cutoffs)
        result[label] = {}
        for group, mask in (
            ("overall", np.ones(len(groups), dtype=bool)),
            ("cold", groups == 2),
            ("warm", groups != 2),
        ):
            result[label][group] = {
                "requests": int(mask.sum()),
                **{
                    metric: float(scores[mask].mean()) if mask.any() else None
                    for metric, scores in values.items()
                },
            }
    return result


def resolve_items(predictions, codes):
    lookup = {tuple(code): item for item, code in enumerate(codes)}
    return np.asarray(
        [lookup.get(tuple(code), -1) for code in predictions.reshape(-1, codes.shape[1])],
        dtype=np.int64,
    ).reshape(predictions.shape[:2])
