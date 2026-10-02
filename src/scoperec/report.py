import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np
import pyarrow.parquet as parquet

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scoperec.audit import METHODS
from scoperec.experiment import write_json
from scoperec.metrics import paired_user_bootstrap, per_request_metrics

LABELS = {
    "frozen": "Frozen TIGER-style",
    "local": "ScopeRec",
    "global": "Global low-rank",
    "local_few_shot": "ScopeRec + few-shot",
    "global_few_shot": "Global + few-shot",
    "content": "Content retrieval",
    "popularity": "Popularity",
    "frozen_beam200": "Frozen, beam 200",
    "new_bias": "New-item bias, beam 200",
    "local_no_replay": "ScopeRec, no replay",
    "global_no_replay": "Global, no replay",
}


def mean_std(values):
    values = np.asarray(values, dtype=float)
    return {
        "mean": float(values.mean()),
        "std": float(values.std(ddof=1)) if len(values) > 1 else 0,
    }


def read_json(path):
    return json.loads(path.read_text())


def compile_results(root: Path):
    config = read_json(root / "configs/experiment.json")
    seeds = config["seeds"]
    summaries = {}
    per_request = {}
    traces = {}
    for method in METHODS:
        records = [read_json(root / f"reports/test/seed_{seed}/{method}.json") for seed in seeds]
        summaries[method] = {
            group: {
                metric: mean_std([record["metrics"][group][metric] for record in records])
                for metric in records[0]["metrics"][group]
                if metric != "requests" and records[0]["metrics"][group][metric] is not None
            }
            for group in ("overall", "new", "old", "base_old", "prior_new")
        }
        traces[method] = [
            np.load(root / f"reports/test/seed_{seed}/{method}.npz") for seed in seeds
        ]
        metric_rows = [
            per_request_metrics(trace["predictions"], trace["target"]) for trace in traces[method]
        ]
        per_request[method] = {
            metric: np.mean([record[metric] for record in metric_rows], axis=0)
            for metric in metric_rows[0]
        }
    reference = traces["frozen"][0]
    groups = reference["group"]
    intervals = {}
    for first, second in (
        ("local", "frozen"),
        ("local", "global"),
        ("local", "new_bias"),
        ("local_few_shot", "local"),
        ("content", "local"),
    ):
        comparison = {}
        for group, mask, metric in (
            ("new", groups == 2, "recall@20"),
            ("old", groups != 2, "ndcg@20"),
            ("overall", np.ones(len(groups), dtype=bool), "ndcg@20"),
        ):
            differences = per_request[first][metric] - per_request[second][metric]
            comparison[f"{group}_{metric}"] = paired_user_bootstrap(
                differences[mask], reference["user"][mask]
            )
        intervals[f"{first}_minus_{second}"] = comparison
    active = traces["local"][0]["active"]
    subgroups = {}
    for name, mask in (
        ("old_inside", (groups != 2) & active),
        ("old_outside", (groups != 2) & ~active),
    ):
        subgroups[name] = {
            "requests": int(mask.sum()),
            **{
                method: {
                    metric: float(scores[mask].mean())
                    for metric, scores in per_request[method].items()
                }
                for method in ("frozen", "local", "global")
            },
        }
    prefix_reports = [
        read_json(root / f"reports/test/seed_{seed}/prefix_diagnostics.json") for seed in seeds
    ]
    prefix = {
        depth: {
            metric: mean_std([record[depth][metric] for record in prefix_reports])
            for metric in prefix_reports[0][depth]
        }
        for depth in prefix_reports[0]
    }
    oracle = {}
    historical = {}
    for method in ("frozen", "local", "global"):
        historical[method] = mean_std(
            [
                read_json(root / f"reports/diagnostics/seed_{seed}/historical_{method}.json")[
                    "metrics"
                ]["overall"]["ndcg@20"]
                for seed in seeds
            ]
        )
        if method != "global":
            oracle[method] = mean_std(
                [
                    read_json(root / f"reports/diagnostics/seed_{seed}/oracle_{method}.json")[
                        "metrics"
                    ]["new"]["recall@20"]
                    for seed in seeds
                ]
            )
    frozen = summaries["frozen"]
    local = summaries["local"]
    return {
        "seeds": seeds,
        "methods": summaries,
        "paired_intervals": intervals,
        "scope_subgroups": subgroups,
        "prefix_diagnostics": prefix,
        "historical_holdout_ndcg20": historical,
        "oracle_new_recall20": oracle,
        "relative_old_ndcg20_change": local["old"]["ndcg@20"]["mean"]
        / frozen["old"]["ndcg@20"]["mean"]
        - 1,
        "uncertainty": "Paired user bootstrap on request differences averaged across fixed seeds; "
        "seed std reported separately; not a population-of-trained-models interval",
        "test_requests": len(groups),
        "new_requests": int(np.sum(groups == 2)),
        "test_users": len(np.unique(reference["user"])),
        "configuration": config,
    }, traces


def figures(root: Path, summary):
    destination = root / "reports/figures"
    destination.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    methods = ("frozen", "local", "global", "local_few_shot", "content", "new_bias")
    palette = ("#626262", "#00796b", "#bd3b52", "#36934b", "#c29000", "#4f64a5")
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    for axis, group, metric, label in zip(
        axes,
        ("new", "old"),
        ("recall@20", "ndcg@20"),
        ("New-item Recall@20 (%)", "Old-item NDCG@20 (%)"),
        strict=True,
    ):
        means = [summary["methods"][method][group][metric]["mean"] * 100 for method in methods]
        errors = [summary["methods"][method][group][metric]["std"] * 100 for method in methods]
        axis.barh(np.arange(len(methods)), means, xerr=errors, color=palette, capsize=3)
        axis.set_yticks(np.arange(len(methods)), [LABELS[method] for method in methods])
        axis.invert_yaxis()
        axis.set_xlabel(label)
        axis.grid(axis="x", alpha=0.18)
    figure.suptitle("Software temporal test: three seeds, mean +/- seed standard deviation")
    figure.savefig(destination / "test_comparison.png", dpi=160)
    plt.close(figure)
    selection = read_json(root / "reports/validation/selection.json")
    figure, axis = plt.subplots(figsize=(7.5, 5), constrained_layout=True)
    for mode, color, marker in (("local", "#00796b", "o"), ("global", "#bd3b52", "s")):
        candidates = selection["candidates"][mode]
        axis.scatter(
            [entry["metrics"]["old"]["ndcg@20"] * 100 for entry in candidates],
            [entry["metrics"]["new"]["recall@20"] * 100 for entry in candidates],
            c=color,
            marker=marker,
            alpha=0.65,
            label=LABELS[mode],
        )
        chosen = selection["selected"][mode]
        axis.scatter(
            chosen["metrics"]["old"]["ndcg@20"] * 100,
            chosen["metrics"]["new"]["recall@20"] * 100,
            marker="*",
            s=200,
            color=color,
            edgecolors="black",
        )
    axis.set(
        xlabel="Old-item NDCG@20 (%)",
        ylabel="New-item Recall@20 (%)",
        title="Validation tradeoff; stars were selected before final testing",
    )
    axis.legend()
    axis.grid(alpha=0.18)
    figure.savefig(destination / "validation_tradeoff.png", dpi=160)
    plt.close(figure)
    depths = list(summary["prefix_diagnostics"])
    figure, axis = plt.subplots(figsize=(7.5, 4.5), constrained_layout=True)
    locations = np.arange(len(depths))
    axis.bar(
        locations - 0.18,
        [
            summary["prefix_diagnostics"][depth]["new_prefix_survival"]["mean"] * 100
            for depth in depths
        ],
        width=0.36,
        label="New target prefix survives beam",
        color="#00796b",
    )
    axis.bar(
        locations + 0.18,
        [
            summary["prefix_diagnostics"][depth]["old_catalog_coverage"]["mean"] * 100
            for depth in depths
        ],
        width=0.36,
        label="Old catalog inside active subtrees",
        color="#bd3b52",
    )
    axis.set(
        xticks=locations,
        xticklabels=depths,
        xlabel="Prefix depth",
        ylabel="Percent",
        title="Reachability versus locality on the independent test batch",
        ylim=(0, 110),
    )
    axis.legend()
    figure.savefig(destination / "prefix_tradeoff.png", dpi=160)
    plt.close(figure)


def failure_cases(root, traces):
    catalog = parquet.read_table(root / "data/experiment/catalog.parquet").to_pylist()
    frozen = traces["frozen"][0]
    local = traces["local"][0]
    content = traces["content"][0]
    base_hit = per_request_metrics(frozen["predictions"], frozen["target"])["recall@20"] > 0
    local_hit = per_request_metrics(local["predictions"], local["target"])["recall@20"] > 0
    content_hit = per_request_metrics(content["predictions"], content["target"])["recall@20"] > 0
    masks = {
        "new_helped_by_patch": (frozen["group"] == 2) & local_hit & ~base_hit,
        "old_harmed_in_ranking": (frozen["group"] != 2) & base_hit & ~local_hit,
        "new_prefix_pruned_before_patch": (frozen["group"] == 2) & ~frozen["survival"][:, 1],
        "content_hit_patch_miss": (frozen["group"] == 2) & content_hit & ~local_hit,
    }
    result = {}
    for label, mask in masks.items():
        examples = []
        for request_index in np.flatnonzero(mask)[:3]:
            item = int(frozen["target"][request_index])
            examples.append(
                {
                    "request_index": int(request_index),
                    "parent_asin": catalog[item]["parent_asin"],
                    "title": catalog[item]["title"][:120],
                    "frozen_hit20": bool(base_hit[request_index]),
                    "local_hit20": bool(local_hit[request_index]),
                    "content_hit20": bool(content_hit[request_index]),
                }
            )
        result[label] = {"count_seed17": int(mask.sum()), "examples": examples}
    write_json(root / "reports/failure_cases.json", result)
    return result


def markdown_report(root, summary):
    methods = summary["methods"]
    costs = read_json(root / "reports/cost_benchmark.json")
    sequential = read_json(root / "reports/diagnostics/sequential/summary.json")
    gate = read_json(root / "reports/diagnostics/gate/summary.json")
    selection = read_json(root / "reports/validation/selection.json")
    local_report = read_json(root / "reports/test/seed_17/local.json")
    config = summary["configuration"]
    lines = [
        "# ScopeRec: Completed Software Experiment",
        "",
        "## Bottom Line",
        "",
        "ScopeRec works as a controllable, reversible output-local adaptation mechanism. "
        "It improves new-item recall over a frozen compact TIGER-style model, but it is not "
        "a no-forgetting method and does not dominate the alternatives on every axis.",
        "",
        f"Three-seed test mean: new Recall@20 moves from "
        f"{100 * methods['frozen']['new']['recall@20']['mean']:.3f}% to "
        f"{100 * methods['local']['new']['recall@20']['mean']:.3f}%; old-item NDCG@20 changes by "
        f"{100 * summary['relative_old_ndcg20_change']:.2f}% relative. "
        "The very small frozen-new denominator makes relative-lift headlines misleading.",
        "",
        "This is an independently implemented compact TIGER-style baseline under a custom "
        "temporal protocol. It is **not** a reproduction of GenRecEdit's FFN editing algorithm "
        "or its published numerical tables. No GenRecEdit superiority claim is supported.",
        "",
        "## Data and Protocol",
        "",
        "- Source: Amazon Reviews 2023 Software, official deduplicated 0-core release.",
        "- 4,828,480 interactions, 2,589,466 users, 89,246 interacted items; all have static text.",
        "- Base cutoff: 2018-01-01. Quantizer fitted on 68,175 pre-cutoff items only.",
        "- Training: 300,000 fixed hash-sampled sequential requests from 1,517,465 eligible "
        "pre-cutoff requests; 4,096 pre-cutoff requests from disjoint users form a holdout.",
        "- Validation arrivals: Jan-Jun 2018, edit Jul 1, targets Jul-Dec 2018. "
        "Grid selection used the first 4,096 requests of a group-independent fixed hash sample.",
        "- Test arrivals: Jan-Jun 2019, edit Jul 1, targets Jul-Dec 2019. "
        "12,000 requests from 9,993 users; 945 new, 9,253 base-old, and 1,802 prior-new targets.",
        "- Final catalog: 76,717 items, including 2,344 target-batch newcomers. "
        "9,898 of 96,471 sequential test-window requests target post-snapshot arrivals "
        "and are excluded explicitly, before uniform request sampling.",
        "- Histories contain at most 20 strictly earlier interactions. Equal-time events are "
        "not used as each other's history. No future k-core, rating or test-user filtering.",
        "- All methods use the complete available catalog without sampled negatives or "
        "seen-item filtering. Sequential generation uses beam 50; bias reranking uses beam 200.",
        "- Reviews proxy interactions; first review proxies availability. Static metadata and "
        "the text model lack historical versions. This is not a true online launch test.",
        "- Earlier newcomers are not implicitly adapted in the main test: the backbone remains "
        "frozen at 2018. Their performance is reported separately rather than silently omitted.",
        "",
        "## Implementation",
        "",
        "MiniLM-L6-v2 normalized 384-D text -> training-only 128-D PCA -> three 64-entry "
        "Faiss residual codebooks -> append-only collision token (capacity 512). "
        "All 89,246 complete SIDs are unique; 43,046 distinct semantic triples and a maximum "
        "collision bucket of 101. The encoder revision and artifact hashes are recorded.",
        "",
        "Backbone: two encoder and two decoder T5 layers, hidden size 128, FFN 512, "
        "four heads, 1,271,936 parameters. Eight epochs with AdamW; checkpoint selection "
        "uses pre-cutoff held-out NLL. Seeds: 17, 29, 43. This is intentionally smaller "
        "than the published six-layer TIGER configuration and uses MiniLM instead of sentence-T5.",
        "",
        "Legal successor logits are masked **before** softmax. Each local patch applies "
        "only after its prefix is actually generated; prefix selection is never teacher-forced "
        "during natural recommendation. Patch artifacts bind to base and SID hashes.",
        "",
        "Zero-shot edits use content-nearest old-item histories from base training: no real "
        "new-target interactions. The final batch has 21,638 pseudo requests and 12,000 "
        "old replay requests. Few-shot adds 739 pre-edit, non-arrival events across covered items; "
        "it is not mixed into the zero-shot claim.",
        "",
        "## Locked Selection",
        "",
        selection["rule"] + ".",
        "",
        "The validation grid compared depths 1/2/3, replay weights 0.2/1/5, local rank 8, "
        "and global ranks 8/64, each with 300 steps. Local and global both selected depth 2 "
        "and replay weight 1; global selected rank 64. Content used the last history item; "
        "new-item bias selected +1 log-score within the frozen beam-200 pool.",
        "",
        "Training data sources and step counts are shared, but total parameter/storage budgets "
        "are **not matched**. Local replay is normalized over active old targets; global replay "
        "may use all old targets. These are disclosed algorithmic/resource differences.",
        "",
        "## Primary Results",
        "",
        "Values are percentages; +/- is sample standard deviation across three seeds. "
        "Content and popularity are deterministic baselines, not three independent trials.",
        "",
        "| Method | New R@20 | Old NDCG@20 | Overall R@20 | Overall NDCG@20 |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for method in METHODS:
        values = []
        for group, metric in (
            ("new", "recall@20"),
            ("old", "ndcg@20"),
            ("overall", "recall@20"),
            ("overall", "ndcg@20"),
        ):
            entry = methods[method][group][metric]
            values.append(f"{100 * entry['mean']:.3f} +/- {100 * entry['std']:.3f}")
        lines.append(f"| {LABELS[method]} | " + " | ".join(values) + " |")
    lines += [
        "",
        "All @10/@20/@50 and base-old/prior-new metrics are in [summary.json](summary.json).",
        "",
        "![Primary comparison](figures/test_comparison.png)",
        "",
        "## Uncertainty",
        "",
        summary["uncertainty"] + ".",
        "",
        "Intervals below are 95% paired user bootstrap intervals (1,000 resamples). "
        "They describe this test set and these trained seeds, not arbitrary datasets.",
        "",
        "| Comparison | Metric | Difference (percentage points) | 95% interval |",
        "| --- | --- | ---: | ---: |",
    ]
    for comparison, metrics in summary["paired_intervals"].items():
        for metric, entry in metrics.items():
            lines.append(
                f"| {comparison} | {metric} | {100 * entry['difference']:.3f} | "
                f"[{100 * entry['low']:.3f}, {100 * entry['high']:.3f}] |"
            )
    lines += [
        "",
        "## Mechanism and Limits",
        "",
        "| Prefix depth | New prefix survival (%) | Old catalog covered (%) | "
        "Old requests covered (%) |",
        "| --- | ---: | ---: | ---: |",
    ]
    for depth, values in summary["prefix_diagnostics"].items():
        lines.append(
            f"| {depth} | {100 * values['new_prefix_survival']['mean']:.2f} | "
            f"{100 * values['old_catalog_coverage']['mean']:.2f} | "
            f"{100 * values['old_request_coverage']['mean']:.2f} |"
        )
    lines += [
        "",
        "![Prefix tradeoff](figures/prefix_tradeoff.png)",
        "",
        "Depth 1 reaches most new targets but activates all 64 coarse branches, so its "
        "outside-invariance promise protects no old catalog items in this batch. Depth 3 "
        "is more local but loses most true paths before the patch can help. Depth 2 is "
        "the chosen compromise, not a universal optimum.",
        "",
        f"Forced-correct-prefix diagnostic (depth 2): frozen new R@20 "
        f"{100 * summary['oracle_new_recall20']['frozen']['mean']:.2f}%, patched "
        f"{100 * summary['oracle_new_recall20']['local']['mean']:.2f}%. "
        "These are oracle-only numbers, never official recommendation results.",
        "",
        "Float64 exhaustive toy-tree tests check total mass, branch mass and outside paths "
        "to 1e-12. On actual sampled test target paths, outside-patch score changes are "
        "exactly zero for all three seeds. Outside rankings may still change through competition.",
        "",
        "| Old target scope | Requests | Frozen NDCG@20 (%) | ScopeRec NDCG@20 (%) |",
        "| --- | ---: | ---: | ---: |",
    ]
    for name, values in summary["scope_subgroups"].items():
        lines.append(
            f"| {name} | {values['requests']} | {100 * values['frozen']['ndcg@20']:.3f} | "
            f"{100 * values['local']['ndcg@20']:.3f} |"
        )
    lines += [
        "",
        "The same single-branch weights were evaluated with and without a prefix gate. "
        f"Max outside token log-probability change: gated "
        f"{gate['outside_path_token_score_max_abs_change']['gated']:.1f}; ungated "
        f"{gate['outside_path_token_score_max_abs_change']['ungated']:.3f}. "
        "The branch was chosen by pseudo-request count, not evaluation success. This isolates "
        "scope; it does not establish a better training algorithm or meaningful new-item lift.",
        "",
        "## Retention and Sequential Updates",
        "",
        "Historical holdout uses original pre-base users and the original base catalog:",
        "",
        "| Method | Historical NDCG@20 (%) |",
        "| --- | ---: |",
    ]
    for method, value in summary["historical_holdout_ndcg20"].items():
        lines.append(f"| {LABELS[method]} | {100 * value['mean']:.3f} |")
    lines += [
        "",
        "Sequential diagnostic (seed 17 only): 1,270 Jan-Mar arrivals followed by "
        "1,074 Apr-Jun arrivals; 332 shared active prefixes. Evaluation always uses the same "
        "final July catalog and future requests. The second update uses a fixed total replay "
        "budget of 12,000; the retention version allocates 6,000 to earlier-new pseudo samples.",
        "",
        "| State | First batch R@20 (%) | Second batch R@20 (%) |",
        "| --- | ---: | ---: |",
    ]
    for name, values in sequential["metrics"].items():
        lines.append(
            f"| {name} | {100 * values['first_batch']['recall@20']:.3f} | "
            f"{100 * values['second_batch']['recall@20']:.3f} |"
        )
    lines += [
        "",
        "Do not compare this extra-update diagnostic as a budget-matched replacement "
        "for the primary one-batch result. It shows same-prefix interference and the need for "
        "replay. It uses pseudo retention labels, not future real clicks.",
        "",
        "Closing the patch reproduced identical recommendations and path scores on 256 "
        "requests for each of three seeds, with unchanged history, catalog and SID versions.",
        "",
        "## Cost",
        "",
        f"Synchronized local-artifact update benchmark on one RTX 4090: "
        f"**{costs['total_update_seconds']:.2f} seconds** for 2,344 new items. Includes text "
        "encoder load/new encoding, frozen SID verification, neighbor/sample preparation, "
        "base load, hidden-state caching, patch optimization, save and reload. Excludes "
        "network downloads, initial old vectors/data preparation and base training.",
        "",
        "| Stage | Seconds |",
        "| --- | ---: |",
    ]
    for name, seconds in costs["stages"].items():
        lines.append(f"| {name} | {seconds:.3f} |")
    lines += [
        "",
        f"The chosen local patch has {costs['parameters']:,} parameters across "
        f"{local_report['training']['branches']:,} branches, "
        f"{costs['patch_bytes'] / 2**20:.2f} MiB on disk and "
        f"{costs['cache_bytes'] / 2**20:.2f} MiB cached tensors. "
        "The global rank-64 patch has 53,440 parameters. Local patch storage exceeds this "
        "compact backbone's parameter storage; low rank is per branch, not globally cheap.",
        "",
        "| Serving mode | Frozen median ms | ScopeRec median ms |",
        "| --- | ---: | ---: |",
    ]
    for name, timings in costs["serving"].items():
        lines.append(
            f"| {name} | {timings['frozen']['batch_ms_median']:.2f} | "
            f"{timings['local']['batch_ms_median']:.2f} |"
        )
    lines += [
        "",
        "Serving timings use seven synchronized repetitions after warmup, prebuilt "
        "tree/token tensors, no oracle or target scoring. They are local microbenchmarks, "
        "not network-service throughput guarantees. The full benchmark is in "
        "[cost_benchmark.json](cost_benchmark.json).",
        "",
        "An implementation check found that training's TF32 setting can perturb text vectors "
        "enough to cross quantizer boundaries. Text encoding now explicitly uses highest FP32 "
        "precision; recomputed new semantic codes match the frozen table. Never silently "
        "regenerate SID mappings when serving an existing checkpoint.",
        "",
        "## What We Learned",
        "",
        "1. Prefix-local adaptation is implementable and reversible; its structural claims "
        "are testable independently of recommendation quality.",
        "2. New-item gains require a retention tradeoff. Removing replay improves new recall "
        "but severely damages old ranking, even with local gates.",
        "3. Batch-wide activity can erase the practical meaning of locality. Measure the "
        "union of active subtrees, not just the size of a representative branch.",
        "4. Global adaptation uses far fewer parameters. ScopeRec has a retention advantage "
        "in this configuration, but the new-recall advantage varies across seeds; consult "
        "paired intervals before making a superiority claim.",
        "5. Content retrieval is a strong cold-item alternative and wins new recall here, "
        "while losing substantial overall ranking quality. Bias reranking is cheap but "
        "limited by its frozen candidate coverage.",
        "6. Few-shot and sequential replay can help, but the results are setting-dependent; "
        "single-seed sequential behavior is a diagnostic, not a broad robustness claim.",
        "",
        "## Scope and Reproduction",
        "",
        "Only one category, one base-time cutoff, one validation arrival batch and one final "
        "test batch were used. The model is compact, training is subsampled, and only one "
        "seed selected hyperparameters. No online A/B, GenRecEdit/SpecGR numerical comparison, "
        "full-model-finetuning baseline, or parameter-matched global claim is made.",
        "",
        "Selected success and failure cases are in [failure_cases.json](failure_cases.json). "
        "They are deterministic examples for explanation, not an unbiased performance estimate.",
        "",
        "![Validation grid](figures/validation_tradeoff.png)",
        "",
        "Run `bash scripts/reproduce.sh all` from the project root. "
        "`bash scripts/reproduce.sh verify` audits actual saved predictions and checkpoints "
        "and runs offline unit tests. Raw data, weights and request-level traces stay local; "
        "code and aggregate reports can be prepared for public release subject to licensing.",
        "",
        f"Configuration: `{config['name']}`. Primary model selection was locked before final "
        "testing; mechanism diagnostics do not alter the selected main configurations.",
        "",
    ]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Generate evidence-based experiment reports and figures."
    )
    parser.add_argument("--root", type=Path, default=Path.cwd())
    arguments = parser.parse_args()
    summary, traces = compile_results(arguments.root)
    write_json(arguments.root / "reports/summary.json", summary)
    failure_cases(arguments.root, traces)
    figures(arguments.root, summary)
    (arguments.root / "reports/RESULTS.md").write_text(markdown_report(arguments.root, summary))
    print(
        json.dumps(
            {
                "report": "reports/RESULTS.md",
                "seeds": summary["seeds"],
                "methods": len(summary["methods"]),
                "test_requests": summary["test_requests"],
            }
        )
    )


if __name__ == "__main__":
    main()
