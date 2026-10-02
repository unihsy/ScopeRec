import numpy as np


def per_request_metrics(predictions: np.ndarray, targets: np.ndarray, cutoffs=(10, 20, 50)):
    if predictions.ndim != 2 or len(predictions) != len(targets):
        raise ValueError("Predictions must be an aligned request-by-candidate matrix")
    hits = predictions == targets[:, None]
    ranks = np.where(hits.any(axis=1), hits.argmax(axis=1) + 1, np.inf)
    metrics = {}
    for cutoff in cutoffs:
        if cutoff > predictions.shape[1]:
            raise ValueError("Not enough predictions for the requested cutoff")
        metrics[f"recall@{cutoff}"] = (ranks <= cutoff).astype(np.float64)
        metrics[f"ndcg@{cutoff}"] = np.where(ranks <= cutoff, 1 / np.log2(ranks + 1), 0)
    return metrics


def aggregate_metrics(predictions, targets, groups, active=None, cutoffs=(10, 20, 50)):
    values = per_request_metrics(predictions, targets, cutoffs)
    masks = {
        "overall": np.ones(len(targets), dtype=bool),
        "new": groups == 2,
        "old": groups != 2,
        "base_old": groups == 0,
        "prior_new": groups == 1,
    }
    if active is not None:
        masks["old_inside"] = (groups != 2) & active
        masks["old_outside"] = (groups != 2) & ~active
    return {
        name: {
            "requests": int(mask.sum()),
            **{
                metric: float(scores[mask].mean()) if mask.any() else None
                for metric, scores in values.items()
            },
        }
        for name, mask in masks.items()
    }


def paired_user_bootstrap(differences, users, seed=17, repetitions=1000):
    differences = np.asarray(differences, dtype=np.float64)
    if not len(differences):
        return {"difference": None, "low": None, "high": None, "users": 0}
    unique, inverse = np.unique(users, return_inverse=True)
    totals = np.bincount(inverse, weights=differences)
    counts = np.bincount(inverse)
    rng = np.random.default_rng(seed)
    estimates = np.empty(repetitions)
    for iteration in range(repetitions):
        selected = rng.integers(len(unique), size=len(unique))
        estimates[iteration] = totals[selected].sum() / counts[selected].sum()
    low, high = np.quantile(estimates, [0.025, 0.975])
    return {
        "difference": float(differences.mean()),
        "low": float(low),
        "high": float(high),
        "users": len(unique),
        "repetitions": repetitions,
        "unit": "paired user clusters",
    }


def validation_utility(metrics: dict, frozen_metrics: dict) -> float:
    new_recall = metrics["new"]["recall@20"]
    old_ndcg = metrics["old"]["ndcg@20"]
    minimum_old = 0.9 * frozen_metrics["old"]["ndcg@20"]
    if new_recall is None or old_ndcg is None or old_ndcg < minimum_old:
        return -1.0
    return float(new_recall * old_ndcg)
