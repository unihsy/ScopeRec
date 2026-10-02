import argparse
import json
from pathlib import Path

import numpy as np

from scoperec.compare_v2 import setup_comparison, summarize_validation
from scoperec.content_scope import ContentScopeModel
from scoperec.evaluate import evaluate_model, load_requests, save_evaluation


def run(device):
    config, base, codes, provenance = setup_comparison(17, device)
    data = Path("data/experiment_v2")
    groups = np.load(data / "validation_groups.npy")
    vectors = np.load(data / "embeddings.npy")
    requests = load_requests(data, "validation", config["selection"]["tuning_requests"])
    directory = Path("reports/v2/validation/seed_17")
    for temperature in (0.025, 0.05):
        for threshold in (0.3, 0.5, 0.7):
            for alpha in (0.05, 0.1, 0.2):
                settings = {
                    "depth": 1,
                    "alpha": alpha,
                    "temperature": temperature,
                    "query_mode": "last",
                    "confidence_threshold": threshold,
                    "confidence_width": 0.05,
                }
                name = f"content_scope_gate_t{temperature:g}_c{threshold:g}_a{alpha:g}".replace(
                    ".", "p"
                )
                if (directory / f"{name}.json").is_file():
                    continue
                adapted = ContentScopeModel(base, codes, groups, vectors, **settings).to(device)
                result, trace = evaluate_model(
                    adapted, requests, codes, groups, base.specification["vocab_size"]
                )
                result.update(
                    {
                        "settings": settings,
                        "provenance": provenance,
                        "method": "Confidence-gated scoped content mass mixture",
                        "trainable_parameters": 0,
                    }
                )
                save_evaluation(directory / name, result, trace)
                print(
                    json.dumps(
                        {
                            "name": name,
                            "new_r20": result["metrics"]["new"]["recall@20"],
                            "old_n20": result["metrics"]["old"]["ndcg@20"],
                            "overall_n20": result["metrics"]["overall"]["ndcg@20"],
                        }
                    ),
                    flush=True,
                )
                del adapted
    frozen = json.loads((directory / "tiger.json").read_text())
    summarize_validation(config, directory, provenance, frozen)


def main():
    parser = argparse.ArgumentParser(description="Validation-only confidence mixture ablation.")
    parser.add_argument("--device", default="cuda:3")
    arguments = parser.parse_args()
    run(arguments.device)


if __name__ == "__main__":
    main()
