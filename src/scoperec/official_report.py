import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scoperec.experiment import write_json
from scoperec.metrics import paired_user_bootstrap
from scoperec.official_experiment import METHODS, ROOT
from scoperec.official_tokens import metric_rows
from scoperec.official_train import read_requests

LABELS = {
    "tiger": "TIGER",
    "genrecedit_3000": "GenRecEdit, lambda 3000",
    "genrecedit_1000": "GenRecEdit, lambda 1000",
    "scoperec_c": "ScopeRec-C",
    "uniform_ablation": "Uniform suffix ablation",
}


def read_json(path):
    return json.loads(path.read_text())


def describe(values):
    values = np.asarray(values, dtype=float)
    return {
        "mean": float(values.mean()),
        "std": float(values.std(ddof=1)),
        "per_seed": values.tolist(),
    }


def summarize():
    config = read_json(Path("configs/official_software.json"))
    data = Path("data/official_software")
    requests = read_requests(data / "test.npz")
    codes = np.load("artifacts/official_benchmark/sid/codes.npy")
    train_mask = np.load(data / "quantizer_train_mask.npy")
    summary = {
        "configuration": config,
        "protocol": read_json(data / "protocol.json"),
        "selection": read_json(ROOT / "selection.json"),
        "paper_reference": read_json(Path("configs/paper_software_reference.json")),
        "sid": read_json(Path("artifacts/official_benchmark/sid/sid.json")),
        "methods": {},
        "paired_intervals": {},
    }
    per_request = {}
    for method in METHODS:
        reports = [read_json(ROOT / f"test/seed_{seed}/{method}.json") for seed in config["seeds"]]
        summary["methods"][method] = {
            definition: {
                group: {
                    metric: describe(
                        [report["metrics"][definition][group][metric] for report in reports]
                    )
                    for metric in reports[0]["metrics"][definition][group]
                    if metric != "requests"
                }
                for group in ("overall", "cold", "warm")
            }
            for definition in ("prefix", "exact")
        }
        for field in ("invalid_full_sid_fraction", "eos_at_fifth_fraction", "inference_seconds"):
            summary["methods"][method][field] = describe([report[field] for report in reports])
        per_request[method] = {}
        for definition, depth in (("prefix", 3), ("exact", 4)):
            rows = []
            for seed in config["seeds"]:
                with np.load(ROOT / f"test/seed_{seed}/{method}.npz") as trace:
                    rows.append(metric_rows(trace["codes"], codes[requests["target"]], depth))
            per_request[method][definition] = {
                name: np.mean([row[name] for row in rows], axis=0) for name in rows[0]
            }
        unseen = (requests["group"] == 2) & ~train_mask[requests["target"]]
        summary["methods"][method]["strict_training_history_unseen"] = {
            "requests": int(unseen.sum()),
            **{
                metric: float(values[unseen].mean())
                for metric, values in per_request[method]["exact"].items()
            },
        }
    for comparator in ("tiger", "genrecedit_3000", "genrecedit_1000", "uniform_ablation"):
        comparison = {}
        for definition, group, metric in (
            ("prefix", "overall", "ndcg@10"),
            ("prefix", "cold", "recall@20"),
            ("exact", "cold", "recall@20"),
            ("exact", "overall", "ndcg@20"),
            ("exact", "warm", "ndcg@20"),
        ):
            selected = (
                np.ones(len(requests["target"]), bool)
                if group == "overall"
                else requests["group"] == 2
                if group == "cold"
                else requests["group"] != 2
            )
            differences = (
                per_request["scoperec_c"][definition][metric]
                - per_request[comparator][definition][metric]
            )
            comparison[f"{definition}/{group}/{metric}"] = paired_user_bootstrap(
                differences[selected], requests["user"][selected], seed=2024, repetitions=2000
            )
        summary["paired_intervals"][f"scoperec_c_minus_{comparator}"] = comparison
    summary["uncertainty"] = (
        "2000 paired user-cluster resamples of request differences averaged over the three fixed "
        "trained models. Seed standard deviation is separate. Descriptive intervals are not "
        "multiple-comparison-corrected or a population interval over training randomness."
    )
    summary["baseline_training"] = {
        str(seed): read_json(
            Path(f"artifacts/official_benchmark/baseline/seed_{seed}/completed.json")
        )
        for seed in config["seeds"]
    }
    return summary


def plot_results(summary):
    directory = ROOT / "figures"
    directory.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    colors = ["#777777", "#b44454", "#cd8490", "#087e68", "#a28b38"]
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.4), constrained_layout=True)
    for axis, definition, group, metric, title in (
        (axes[0], "prefix", "overall", "ndcg@10", "Released 3-token Overall NDCG@10"),
        (axes[1], "exact", "cold", "recall@20", "Exact 4-token Cold Recall@20 (%)"),
    ):
        scale = 100 if definition == "exact" else 1
        means = [
            summary["methods"][method][definition][group][metric]["mean"] * scale
            for method in METHODS
        ]
        errors = [
            summary["methods"][method][definition][group][metric]["std"] * scale
            for method in METHODS
        ]
        axis.barh(range(len(METHODS)), means, xerr=errors, capsize=3, color=colors)
        axis.set_yticks(range(len(METHODS)), [LABELS[method] for method in METHODS])
        axis.invert_yaxis()
        axis.set_xlabel(title)
        axis.grid(axis="x", alpha=0.2)
    figure.suptitle("Released Software protocol: mean +/- seed standard deviation")
    figure.savefig(directory / "aligned_comparison.png", dpi=160)
    plt.close(figure)


def markdown(summary):
    method_results = summary["methods"]
    protocol = summary["protocol"]
    paper = summary["paper_reference"]
    lines = [
        "# GenRecEdit Released-Benchmark Alignment",
        "",
        "## Answer",
        "",
        "GenRecEdit evaluates Amazon Reviews 2023 **Video Games, Software, and Cell Phones "
        "and Accessories**, with time-based splits and Overall/Cold/Warm results at 10/20/50. "
        "This run aligns **Software** with its released implementation, rather than comparing "
        "our earlier custom 0-core time windows to its paper tables.",
        "",
        "The released-code experiment is complete: three trained backbones, two fixed "
        "GenRecEdit settings, ScopeRec-C and a uniform-suffix control. Tokenizer, search and "
        "metrics were checked by executing the pinned official functions. "
        "**The paper's GenRecEdit result has not been numerically reproduced.** "
        "Paper prose and released defaults conflict, and the measured cold fraction differs. "
        "This result is not evidence of beating the published paper.",
        "",
        "## Dataset and Protocol",
        "",
        "- Official Amazon 2023 `5core/timestamp_w_his` Software train/valid/test CSV archives.",
        "- Official absolute-time boundaries: 1628643414042 and 1658002729837 milliseconds.",
        "- Follow the released `max_rows=0.5` test-prefix user selection, including the first "
        "row exceeding the half boundary. Train/validation are filtered to that user set.",
        "- 3,653 selected users; 17,591 catalog items. Filtered training has 31,626 source rows; "
        "its repeated prefix expansion produces 464,196 training examples but only 31,626 "
        "unique user/history/target combinations. Duplicates are intentionally not removed.",
        "- Validation: 5,126 requests, 1,055 cold. Test: 9,386 requests, 2,226 cold (23.72%), "
        "versus 24.3% reported in the paper. These are the actual downloaded-file counts.",
        "- Released cold labels mean absent from filtered training **row targets**. "
        "Quantizer fitting uses all items occurring in training sequences (7,011 rather than "
        "6,621 target items). Therefore 37 cold test requests target items seen in training "
        "histories. Strict history-unseen diagnostic counts are separately included in JSON.",
        "- Test cold target identities define the adaptation candidate list, as in the release. "
        "This is label-dependent benchmark task construction, not label-free production arrival "
        "discovery. All methods use the same candidate identity information.",
        "- No extra rating threshold, deduplication, seen-item filtering or catalog masking.",
        "",
        "## Alignment Matrix",
        "",
        "| Component | Released code / this run | Paper text / unresolved difference |",
        "| --- | --- | --- |",
        "| Dataset | Official 5-core timestamp-with-history Software | "
        "Three categories in paper; only Software rerun here |",
        "| User filtering | Prefix-selected test users, `max_rows=0.5` | "
        "Prose does not fully specify this sampling |",
        "| Content | Sentence-T5-base; title/features/categories/description | "
        "Sentence-T5 also specified |",
        "| Quantization | All-catalog white PCA(128), Faiss residual quantizer, 3 x 256 | "
        "Prose specifies trained multi-head RQ-VAE, latent 32 |",
        "| Item code | Three semantic codes plus 1-based collision token | "
        "Section 5 says no appended token; other sections discuss four positions |",
        "| Input/output | User 1025, EOS 1026, decoder start/padding 0, vocabulary 1027 | "
        "Tokenizer specifics are implementation-level |",
        "| Backbone | T5 6 encoder/decoder layers, d=128, FFN=1024, 8 heads, d_kv=64 | "
        "Released architecture including gated-silu |",
        "| Training | AdamW lr=0.05, wd=0.05, batch=1024, 10 epochs, warmup=10000 | "
        "4540 total updates stays inside warmup |",
        "| Checkpoint | Last evaluated epoch, unconditional released save | "
        "Not validation-best checkpoint selection |",
        "| Search | Unmasked full vocabulary, beam=50, four SID tokens then fifth token/EOS | "
        "Invalid codes are possible and retained |",
        "| Main metric | First-three-token NDCG; Recall uses the same first hit | "
        "Paper describes Recall/NDCG without this implementation caveat |",
        "| GenRecEdit | FFN target optimization, uncentered old-key moments, One-One [0,1,2,3] | "
        "Classifier-based layer search in paper is not used by default scripts |",
        "| Preservation weight | Both 3000 and 1000 reported, not chosen on test | "
        "Paper Software=3000; released shell default=1000 |",
        "",
        "## Necessary Runtime Choices",
        "",
        "The released trainer contains a missing `eval_interval` dictionary key and misspelled "
        "`evaluate`/`accelerator`/logging calls. We repaired the execution path while retaining "
        "its optimizer, scheduling and unconditional last-evaluated checkpoint behavior. "
        "FP32 gradient accumulation (microbatch 128, effective batch 1024) fits the 24GB GPUs. "
        "It preserves the objective but not identical dropout random-number consumption.",
        "",
        "The original augmentation defaults to an unseeded RNG and Python-set item order. "
        "This run fixes seeds 2024/2025/2026 and item-ID ordering. The 400k covariance sample "
        "is shared across layer/position moments instead of separate per-layer draws; moments "
        "are collected together and compared with direct activation captures. The model's "
        "causal attention makes the full-prefix capture mathematically equivalent.",
        "",
        "Same Sentence-T5 and SID artifacts are shared across all three training seeds. "
        "Faiss 1.11.0 and scikit-learn 1.6.1 use the released default APIs; PCA selects "
        "`covariance_eigh` in this environment. Library versions, microbatching and random "
        "sample ordering are disclosed differences, not bitwise-reproduction claims.",
        "",
        "## Functional Checks",
        "",
        "The pinned reference is commit `e6878d9c7c6e57479e840ccb8c045b11a2bd69b5`.",
        "",
        "- Executed official collision extension and token mapping: all 17,591 item codes match.",
        "- Executed official tokenizer: user token, masks, right padding and five labels match.",
        "- Executed official unmasked Beam Search: five-token sequences match on a small model.",
        "- Executed released evaluator: first-three-token NDCG matches, including a deliberate "
        "case where the fourth token is wrong. Prefix and exact metrics are not interchangeable.",
        "- Executed official z optimizer on the actual trained model: success masks match in all "
        "four positions; maximum delta difference about 3.2e-5. The matrix-update difference "
        "is about 9.3e-10. The port uses vectorized Adam and `solve` instead of an inverse.",
        "- Saved edits reproduce predictions; disabling them restores exact base path scores. "
        "All three backbones are unchanged after inference/editing.",
        "",
        "## Against the Paper",
        "",
        "This is a reference comparison, **not matched-protocol superiority evidence**. "
        "Local values use released three-token matching and three-seed means; paper values "
        "are transcribed from Table 1. ScopeRec is not inserted into the paper's table.",
        "",
        "| Method | Paper Overall NDCG@10 | Local mean | Paper Cold R@20 | Local mean |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for local_name, paper_name in (("tiger", "tiger"), ("genrecedit_3000", "genrecedit")):
        local = method_results[local_name]["prefix"]
        lines.append(
            f"| {LABELS[local_name]} | {paper[paper_name]['overall']['ndcg@10']:.4f} | "
            f"{local['overall']['ndcg@10']['mean']:.5f} | "
            f"{paper[paper_name]['cold']['recall@20']:.4f} | "
            f"{local['cold']['recall@20']['mean']:.5f} |"
        )
    lines += [
        "",
        "The primary seed 2024 TIGER NDCG@10 is 0.03477, close to paper 0.0353. "
        "GenRecEdit's paper cold gains do not appear at the same scale. Reproducing the "
        "paper's trained tokenizer, checkpoints, exact user subset and selected edit layers "
        "is still necessary before treating its numeric table as reproduced.",
        "",
        "## Local Comparisons",
        "",
        "These are matched runs: identical backbone per seed, SID, requests, candidate identities "
        "and unmasked search. Values are mean +/- seed standard deviation. Recall is a percentage; "
        "NDCG remains on its original 0-1 scale.",
        "",
        "| Method | Prefix Overall N@10 | Prefix Cold R@20 (%) | "
        "Exact Cold R@20 (%) | Exact Overall N@20 |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for method in METHODS:
        values = []
        for definition, group, metric, scale in (
            ("prefix", "overall", "ndcg@10", 1),
            ("prefix", "cold", "recall@20", 100),
            ("exact", "cold", "recall@20", 100),
            ("exact", "overall", "ndcg@20", 1),
        ):
            entry = method_results[method][definition][group][metric]
            values.append(f"{entry['mean'] * scale:.5f} +/- {entry['std'] * scale:.5f}")
        lines.append(f"| {LABELS[method]} | " + " | ".join(values) + " |")
    lines += [
        "",
        "![Released benchmark comparison](figures/aligned_comparison.png)",
        "",
        "ScopeRec was selected by highest overall validation three-token NDCG@10, then cold "
        "Recall@20. Chosen: depth 1, alpha 0.05, temperature 0.05, confidence threshold 0.5, "
        "width 0.05. Only validation metrics selected this point. The uniform ablation retains "
        "the same confidence gate but uses a uniform new-item distribution inside each branch.",
        "",
        "The official-protocol ScopeRec mixture operates on the **full vocabulary**, including "
        "invalid SID strings and the fifth EOS position. Its prefix mass is preserved over "
        "that full path space; the valid-item catalog mass can change. It does not receive an "
        "undocumented legal-token mask. Log-domain mixtures avoid cancellation of tiny base "
        "probabilities. For old complete paths, the retained probability is at least 0.95 "
        "of baseline; this is not a guarantee of NDCG retention.",
        "",
        "## Uncertainty",
        "",
        summary["uncertainty"],
        "",
        "Differences below are percentage points (including 100 x differences in NDCG).",
        "",
        "| ScopeRec-C minus | Metric | Mean difference (pp) | 95% interval (pp) |",
        "| --- | --- | ---: | ---: |",
    ]
    for comparison, metrics in summary["paired_intervals"].items():
        comparator = comparison.removeprefix("scoperec_c_minus_")
        for metric, entry in metrics.items():
            lines.append(
                f"| {LABELS[comparator]} | {metric} | {entry['difference'] * 100:.4f} | "
                f"[{entry['low'] * 100:.4f}, {entry['high'] * 100:.4f}] |"
            )
    lines += [
        "",
        "## Invalid Codes and Costs",
        "",
        "| Method | Invalid full SID among top 50 (%) | Inference seconds, 9386 requests |",
        "| --- | ---: | ---: |",
    ]
    for method in METHODS:
        values = method_results[method]
        lines.append(
            f"| {LABELS[method]} | {100 * values['invalid_full_sid_fraction']['mean']:.2f} | "
            f"{values['inference_seconds']['mean']:.2f} |"
        )
    lines += [
        "",
        "Inference times cover the evaluator and CPU transfer, not network serving. "
        "The full model remains about 9.57M parameters; training each seed took about 32-33 "
        "minutes. GenRecEdit edits four 128 x 1024 FFN matrices. ScopeRec stores no learned "
        "patch weights but needs the 17,591 x 768 content vectors (~51.5 MiB), plus tree/query "
        "work. These measurements are not paper-relative speedup claims.",
        "",
        "## Conclusions",
        "",
        "1. We now have a substantially closer released-code benchmark than v1/v2: "
        "5-core splits, Sentence-T5, 256-entry codebooks, user/EOS tokens, full-vocabulary "
        "training/search, and released prefix metrics.",
        "2. TIGER is near the paper's overall scale, but GenRecEdit numerical replication is "
        "not achieved. A functional parity test does not resolve mismatched data/tokenizer "
        "or paper-versus-release choices.",
        "3. ScopeRec-C improves exact cold recall relative to these locally reproduced models. "
        "Its small overall gains must be judged using paired intervals, not rounded means.",
        "4. Uniform suffix allocation is very close here. Content ranking is not established "
        "as the principal cause of the gain under this benchmark. This changes the interpretation "
        "from our earlier custom-window study and is retained as a negative ablation finding.",
        "5. Prefix-only metrics, invalid codes and target-defined cold lists materially affect "
        "the task. Always retain full-item metrics and protocol caveats beside headline numbers.",
        "",
        "## Reproduction",
        "",
        "Run `bash scripts/reproduce_official.sh verify` for tests and actual artifact audit. "
        "`SCOPE_DEVICE=cuda:0 bash scripts/reproduce_official.sh all` fills missing stages "
        "and preserves completed models and locked final results. Artifacts are isolated in "
        "`data/official_software`, `artifacts/official_benchmark`, "
        "and `reports/official_benchmark`.",
        "",
        "This release reruns Software only. Video Games and Cell Phones and Accessories, "
        "the paper-specific 10,000-epoch RQ-VAE/no-extra-token variant, classifier layer search "
        "and exact paper checkpoints remain outside this completed released-default experiment. "
        "Do not delete hard examples or tune on this test to make the paper's numbers appear.",
        "",
        "References: [paper v2](https://arxiv.org/html/2603.14259v2), "
        "[official code](https://github.com/Starrylay/GenRecEdit), "
        "[dataset](https://amazon-reviews-2023.github.io/). Raw data, weights and upstream "
        "code remain excluded from the source-only package.",
        "",
    ]
    if protocol["catalog_items"] != summary["sid"]["items"]:
        raise ValueError("Report catalog counts disagree")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Report released-benchmark results and paper differences."
    )
    parser.parse_args()
    summary = summarize()
    write_json(ROOT / "summary.json", summary)
    plot_results(summary)
    (ROOT / "RESULTS.md").write_text(markdown(summary))
    print(
        json.dumps(
            {
                "report": str(ROOT / "RESULTS.md"),
                "methods": len(METHODS),
                "paper_table_reproduced": False,
                "local_cold_recall20": {
                    method: summary["methods"][method]["exact"]["cold"]["recall@20"]["mean"]
                    for method in METHODS
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
