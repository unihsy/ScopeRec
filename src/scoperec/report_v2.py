import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scoperec.audit_v2 import METHODS_V2
from scoperec.experiment import write_json
from scoperec.metrics import paired_user_bootstrap, per_request_metrics

LABELS_V2 = {
    "tiger": "TIGER (common protocol)",
    "genrecedit": "GenRecEdit, new-priority",
    "genrecedit_balanced": "GenRecEdit, balanced",
    "scoperec_v1": "ScopeRec v1, new-priority",
    "scoperec_v1_balanced": "ScopeRec v1, balanced",
    "scoperec_content": "ScopeRec-C, new-priority",
    "scoperec_content_balanced": "ScopeRec-C, balanced",
    "scoperec_content_fixed": "ScopeRec-C, fixed budget",
    "uniform_content_ablation": "Uniform new suffix ablation",
    "content_retrieval": "Content-only retrieval",
}


def read_json(path):
    return json.loads(path.read_text())


def stats(values):
    values = np.asarray(values, dtype=float)
    return {
        "mean": float(values.mean()),
        "std": float(values.std(ddof=1)),
        "per_seed": values.tolist(),
    }


def compile_summary(root):
    config = read_json(root / "configs/experiment_v2.json")
    summary = {
        "seeds": config["seeds"],
        "methods": {},
        "comparisons": {},
        "configuration": config,
        "confirmation_window": config["windows"]["test"],
    }
    per_request = {}
    target = None
    users = None
    groups = None
    for method in METHODS_V2:
        records = [
            read_json(root / f"reports/v2/test/seed_{seed}/{method}.json")
            for seed in config["seeds"]
        ]
        summary["methods"][method] = {
            group: {
                metric: stats([record["metrics"][group][metric] for record in records])
                for metric in records[0]["metrics"][group]
                if metric != "requests" and records[0]["metrics"][group][metric] is not None
            }
            for group in ("overall", "new", "old", "base_old", "prior_new")
        }
        summary["methods"][method]["inference_seconds_including_target_scoring"] = stats(
            [record["inference_seconds"] for record in records]
        )
        metric_rows = []
        for seed in config["seeds"]:
            with np.load(root / f"reports/v2/test/seed_{seed}/{method}.npz") as trace:
                if target is None:
                    target, users, groups = trace["target"], trace["user"], trace["group"]
                else:
                    np.testing.assert_array_equal(target, trace["target"])
                    np.testing.assert_array_equal(users, trace["user"])
                metric_rows.append(per_request_metrics(trace["predictions"], trace["target"]))
        per_request[method] = {
            metric: np.mean([entry[metric] for entry in metric_rows], axis=0)
            for metric in metric_rows[0]
        }
    comparisons = (
        ("scoperec_content", "tiger"),
        ("scoperec_content", "genrecedit"),
        ("scoperec_content", "scoperec_v1"),
        ("scoperec_content_balanced", "tiger"),
        ("scoperec_content_balanced", "genrecedit"),
        ("scoperec_content_balanced", "genrecedit_balanced"),
        ("scoperec_content", "uniform_content_ablation"),
    )
    for first, second in comparisons:
        measures = {}
        for label, selected, metric in (
            ("new_recall20", groups == 2, "recall@20"),
            ("old_ndcg20", groups != 2, "ndcg@20"),
            ("overall_ndcg20", np.ones(len(groups), dtype=bool), "ndcg@20"),
        ):
            difference = per_request[first][metric] - per_request[second][metric]
            measures[label] = paired_user_bootstrap(
                difference[selected], users[selected], repetitions=2000
            )
        summary["comparisons"][f"{first}_minus_{second}"] = measures
    summary["test"] = {
        "requests": len(target),
        "users": len(np.unique(users)),
        "group_counts": {str(group): int(np.sum(groups == group)) for group in np.unique(groups)},
    }
    summary["uncertainty"] = (
        "2000 paired user-cluster bootstrap resamples of per-request differences averaged over "
        "the three fixed models; not a confidence interval over arbitrary model training runs. "
        "Seed standard deviation is separately reported; descriptive comparisons are not "
        "multiple-testing-corrected confirmatory hypothesis tests."
    )
    summary["selection"] = read_json(root / "reports/v2/validation/selection.json")
    summary["official_parity"] = read_json(root / "reports/v2/genrecedit_reference_parity.json")
    summary["integrity_and_latency"] = read_json(
        root / "reports/v2/diagnostics/seed_17/integrity_and_latency.json"
    )
    summary["prefix_diagnostics_seed17"] = read_json(
        root / "reports/v2/test/seed_17/prefix_diagnostics.json"
    )
    summary["cost"] = {
        "base_parameters": read_json(root / "artifacts/v2/baseline/seed_17/summary.json")[
            "trainable_parameters"
        ],
        "base_training_seconds": stats(
            [
                read_json(root / f"artifacts/v2/baseline/seed_{seed}/summary.json")[
                    "training_seconds"
                ]
                for seed in config["seeds"]
            ]
        ),
        "genrecedit_key_covariance_and_z_optimization_seconds": stats(
            [
                read_json(root / f"reports/v2/test/seed_{seed}/genrecedit.json")[
                    "optimization_seconds"
                ]
                for seed in config["seeds"]
            ]
        ),
        "genrecedit_edit_parameters": 4 * config["model"]["d_model"] * config["model"]["d_ff"],
        "scoperec_v1_parameters": read_json(root / "reports/v2/test/seed_17/scoperec_v1.json")[
            "training"
        ]["parameters"],
        "scoperec_content_additional_trainable_parameters": 0,
        "scoperec_content_new_vector_bytes": read_json(
            root / "reports/v2/test/seed_17/scoperec_content.json"
        )["new_index_bytes"],
        "scoperec_content_all_vector_bytes": read_json(
            root / "reports/v2/test/seed_17/scoperec_content.json"
        )["total_vectors_bytes"],
        "update_comparison_warning": "GenRecEdit fitting excludes upstream data/embedding costs; "
        "content model construction assumes existing vectors and SIDs. "
        "These are not end-to-end matched update costs.",
    }
    summary["reserved_validation"] = {
        method: read_json(
            root / f"reports/v2/diagnostics/seed_17/reserved_validation_{method}.json"
        )["metrics"]
        for method in ("tiger", "scoperec_content", "scoperec_content_balanced")
    }
    return summary


def render_figures(root, summary):
    destination = root / "reports/v2/figures"
    destination.mkdir(parents=True, exist_ok=True)
    methods = (
        "tiger",
        "genrecedit",
        "genrecedit_balanced",
        "scoperec_v1",
        "scoperec_content",
        "scoperec_content_balanced",
    )
    colors = ("#777777", "#bb4155", "#db8794", "#407ec1", "#067d62", "#7bb78e")
    plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "font.size": 10})
    figure, axes = plt.subplots(1, 2, figsize=(12.4, 4.8), constrained_layout=True)
    for axis, group, metric, title in zip(
        axes,
        ("new", "overall"),
        ("recall@20", "ndcg@20"),
        ("New-item Recall@20 (%)", "Overall NDCG@20 (%)"),
        strict=True,
    ):
        means = [summary["methods"][method][group][metric]["mean"] * 100 for method in methods]
        errors = [summary["methods"][method][group][metric]["std"] * 100 for method in methods]
        axis.barh(range(len(methods)), means, xerr=errors, color=colors, capsize=3)
        axis.set_yticks(range(len(methods)), [LABELS_V2[method] for method in methods])
        axis.invert_yaxis()
        axis.set_xlabel(title)
        axis.grid(axis="x", alpha=0.2)
    figure.suptitle("Fresh 2020 confirmation window: mean +/- seed standard deviation")
    figure.savefig(destination / "confirmation.png", dpi=160)
    plt.close(figure)
    search = read_json(root / "reports/v2/validation/search.json")
    figure, axis = plt.subplots(figsize=(8, 5), constrained_layout=True)
    for label, color, marker in (
        ("genrecedit", "#bb4155", "s"),
        ("scoperec_v1", "#407ec1", "^"),
        ("scoperec_content", "#067d62", "o"),
    ):
        candidates = search["candidates"][label]
        axis.scatter(
            [entry["metrics"]["old"]["ndcg@20"] * 100 for entry in candidates],
            [entry["metrics"]["new"]["recall@20"] * 100 for entry in candidates],
            label=LABELS_V2[label],
            color=color,
            marker=marker,
            alpha=0.55,
        )
    for label, color in (("scoperec_content", "#067d62"), ("scoperec_content_balanced", "#7bb78e")):
        chosen = summary["selection"]["chosen"][label]
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
        title="Validation candidates; stars fixed before final testing",
    )
    axis.legend()
    axis.grid(alpha=0.2)
    figure.savefig(destination / "validation_tradeoff.png", dpi=160)
    plt.close(figure)


def relative_change(summary, method, group, metric, reference="tiger"):
    baseline = summary["methods"][reference][group][metric]["mean"]
    return summary["methods"][method][group][metric]["mean"] / baseline - 1


def make_markdown(summary):
    methods = summary["methods"]
    parity = summary["official_parity"]
    main_old_change = 100 * relative_change(summary, "scoperec_content", "old", "ndcg@20")
    balanced_old_change = 100 * relative_change(
        summary, "scoperec_content_balanced", "old", "ndcg@20"
    )
    balanced_new = 100 * methods["scoperec_content_balanced"]["new"]["recall@20"]["mean"]
    lines = [
        "# GenRecEdit Reproduction and ScopeRec-C",
        "",
        "## Outcome",
        "",
        "GenRecEdit's actual position-wise FFN editing core has been implemented and checked "
        "against executed, pinned official functions. ScopeRec adds a content-conditional "
        "subtree mixture and a history-based confidence budget. The original learned low-rank "
        "implementation and its first-release results are preserved.",
        "",
        f"On the new 2020 confirmation window, ScopeRec-C's new-priority point reaches "
        f"{100 * methods['scoperec_content']['new']['recall@20']['mean']:.3f}% new Recall@20, "
        f"versus GenRecEdit {100 * methods['genrecedit']['new']['recall@20']['mean']:.3f}% and "
        f"TIGER {100 * methods['tiger']['new']['recall@20']['mean']:.3f}%. "
        f"Old NDCG changes {main_old_change:.2f}% "
        "relative to TIGER; this point does not improve every metric.",
        "",
        f"The separately predeclared balanced point reaches "
        f"{balanced_new:.3f}% new Recall@20 with {balanced_old_change:.2f}% "
        f"relative old NDCG change and "
        f"{100 * relative_change(summary, 'scoperec_content_balanced', 'overall', 'ndcg@20'):.2f}% "
        "relative overall NDCG change. Its aggregate gain over the conservative GenRecEdit point "
        "is small; consult the paired intervals rather than declaring universal dominance.",
        "",
        "## What Was Reproduced",
        "",
        f"Official source: Starrylay/GenRecEdit commit `{parity['upstream_revision']}`. "
        "Downloaded source is kept under local artifacts and excluded from the source package.",
        "",
        "The following official functions were actually executed for parity, not merely cited:",
        "- `genrecedit_optimize_z_vectors`: frozen-backbone target-token probability optimization, "
        "Adam, cosine schedule, norm-relative penalty, clipping and successful-request filtering.",
        "- `genrecedit_solve_weight_delta`: non-centered old-key second moment and the FFN "
        "output-weight least-squares update.",
        "",
        "Four positions use the released default mapping `[0,1,2,3]`. Each decoding step activates "
        "only its mapped FFN layer on the current token (One-One); not all edits act at once. "
        "The official default layer mapping is reproduced; a separate paper classifier-based "
        "layer-localization study is not reproduced.",
        "",
        "| Position | Matched successful requests / 8 | Maximum optimized-delta difference |",
        "| --- | ---: | ---: |",
    ]
    for entry in parity["position_optimization"]:
        lines.append(
            f"| {entry['position']} | {entry['successful']} / 8 | "
            f"{entry['max_delta_abs_difference']:.3e} |"
        )
    lines += [
        "",
        f"Successful-request masks matched at all four positions. Solve maximum error: "
        f"{parity['solve_max_abs_difference']:.3e}. The port vectorizes the same objective, "
        "recomputes short decoder prefixes instead of using the upstream cache, and uses a "
        "linear solve instead of explicitly forming an inverse. Original matrices remain frozen.",
        "",
        "This is an **algorithm-core reproduction and common-protocol comparison**, not a claim "
        "to reproduce SIGIR table numbers. Text features (MiniLM), three 64-entry codebooks, "
        "a training subsample, legal-mask-before-softmax decoding, optimizer and temporal splits "
        "differ from the published experiment. Raw-paper preprocessing, sentence-T5 features, "
        "the complete paper-sized data setup and its numerical tables remain unreplicated.",
        "",
        "## Common Experiment",
        "",
        "- Amazon Reviews 2023 Software; base cutoff 2018-01-01, unchanged frozen SID artifact.",
        "- Same six encoder / six decoder T5 layers, hidden 128, FFN 1024, 8 heads, "
        "key/value head width 64, 9,532,288 parameters for all methods. Attention width matches "
        "the released TIGER configuration; training remains the project's uniform temporal setup.",
        "- 300,000 pre-base training requests, 4,096 disjoint-user pre-base holdout requests, "
        "eight epochs, seeds 17/29/43. No method receives a stronger backbone or different SID.",
        "- Validation: Jan-Jun 2018 arrivals, July edit, Jul-Dec targets; 4,096 tuning requests "
        "including 449 new targets. The first release used this validation window too.",
        "- Fresh confirmation: Jan-Jun 2020 arrivals, July edit, Jul-Dec targets. "
        "12,000 requests / 10,088 users; 707 new, 7,998 base-old and 3,295 prior-new targets. "
        "The earlier 2019 test was already seen and is not reused as fresh evidence.",
        "- Catalog: 81,196 items including 2,152 current newcomers. "
        "4,762 of 68,903 otherwise sequential requests target post-snapshot arrivals and are "
        "excluded before group-independent hash sampling.",
        "- GenRecEdit and v1 ScopeRec use the same 19,879 content-neighbor pseudo requests "
        "and 12,000 pre-base replay/covariance requests. No real new-target interactions are used. "
        "ScopeRec-C uses the same static item vectors, not extra labels or future events.",
        "- All methods share the full candidate snapshot and histories, with no sampled negatives "
        "or seen-item filtering. Natural beam 50 is used. Evaluation targets never choose routes.",
        "- First review time is only an arrival proxy. Text and pretrained embeddings are not "
        "historically versioned, so offline temporal visibility has the same limitations as v1.",
        "",
        "## ScopeRec-C Design",
        "",
        "The v1 diagnosis showed that deep prefixes were unreachable and thousands of independent "
        "low-rank branches were expensive. ScopeRec-C uses a shallow output scope with a small, "
        "explicit probability-transfer budget. It is a new content-mixture variant, **not** the "
        "same low-rank patch claimed to become parameter-free.",
        "",
        "Within active prefix `b`, let `R(i|H,b)` be a temperature-softmax of history-to-new-item "
        "content similarities, normalized over newcomers in that subtree. Let `a(H,b)` be the "
        "maximum mixture budget times a sigmoid of the strongest new-item similarity. Then:",
        "",
        "$$",
        "P_C(i|H)=P_0(b|H)\\left[(1-a(H,b))P_0(i|H,b)+a(H,b)R(i|H,b)\\right].",
        "$$",
        "",
        "`R` is zero on old items. The implementation derives conditional token probabilities "
        "from this complete-path mixture rather than adding arbitrary token bonuses. Thus prefix "
        "mass is preserved, outside paths are unchanged, and any old path inside a branch retains "
        "at least `(1-alpha_max)` of its base probability. These are probability properties, "
        "not guarantees about Recall or NDCG. In this batch, depth 1 activates all 64 branches, "
        "so outside-scope protection is vacuous; the effective protection is the mass budget.",
        "",
        "No new weights are trained. History item vectors and new-item content vectors are "
        "required at serving time; similarity computation and memory are real costs. "
        "The new suffix distribution is query-dependent and selected without the correct target.",
        "",
        "## Selection",
        "",
        "All choices were locked before 2020 evaluation. The new-priority rule maximizes "
        "new Recall@20 times old NDCG@20, with old NDCG at least 95% of frozen on validation. "
        "The balanced rule maximizes overall NDCG among the same eligible candidates. "
        "These two objectives were declared before the fresh confirmation run.",
        "",
        "- GenRecEdit: covariance weights 100/1000/10000/100000; "
        "chosen 100 (new) and 100000 (balanced).",
        "- v1 ScopeRec: depths 1/2/3 and replay weights 0.2/1/5; chosen depth 2 / replay 1 (new), "
        "depth 3 / replay 0.2 (balanced), rank 8.",
        "- ScopeRec-C: 30 fixed-budget settings plus 18 confidence settings. New-priority: "
        "depth 1, alpha 0.2, temperature 0.05, threshold 0.5, confidence width 0.05. "
        "Balanced: depth 1, alpha 0.1, temperature 0.025, threshold 0.7, width 0.05.",
        "- Fixed-budget control is independently validation-selected, not a pure gate-only "
        "ablation. The uniform-suffix control keeps the selected confidence gate but removes "
        "content-based ranking inside new-item branches.",
        "",
        "A decimal-in-filename artifact bug was found during validation and fixed before locking. "
        "Affected content settings were rerun with collision-free names; overwritten preliminary "
        "records are excluded from selection. Final test artifacts were never used for retuning.",
        "",
        "## Confirmation Results",
        "",
        "Percent units; +/- is sample standard deviation over three backbone seeds. "
        "Content retrieval is deterministic, so its repeated rows are not independent replicates.",
        "",
        "| Method | New R@20 | Old NDCG@20 | Overall R@20 | Overall NDCG@20 |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for method in METHODS_V2:
        values = []
        for group, metric in (
            ("new", "recall@20"),
            ("old", "ndcg@20"),
            ("overall", "recall@20"),
            ("overall", "ndcg@20"),
        ):
            entry = methods[method][group][metric]
            values.append(f"{100 * entry['mean']:.3f} +/- {100 * entry['std']:.3f}")
        lines.append(f"| {LABELS_V2[method]} | " + " | ".join(values) + " |")
    lines += [
        "",
        "All @10/20/50, base-old and earlier-new details are in [summary.json](summary.json).",
        "",
        "![Confirmation comparison](figures/confirmation.png)",
        "",
        "## Paired Uncertainty",
        "",
        summary["uncertainty"],
        "",
        "| Difference | Metric | Mean difference (pp) | 95% interval (pp) |",
        "| --- | --- | ---: | ---: |",
    ]
    for comparison, metrics in summary["comparisons"].items():
        for metric, values in metrics.items():
            lines.append(
                f"| {comparison} | {metric} | {100 * values['difference']:.3f} | "
                f"[{100 * values['low']:.3f}, {100 * values['high']:.3f}] |"
            )
    lines += [
        "",
        "New-priority is not the overall-NDCG winner. Balanced ScopeRec-C has "
        "stronger new recall than the conservative GenRecEdit point, but their overall NDCG "
        "difference is small. A positive mean alone is not evidence of a robust broad win.",
        "",
        "## Mechanism and Costs",
        "",
        "The uniform-new-suffix control substantially reduces the gain: content relevance, "
        "not merely a generic new-item boost, matters here. The fixed-budget variant is also "
        "weaker in new recall than the confidence setting, though independently tuned.",
        "",
        "The same Toy-T5 tests verify probability normalization, branch mass and natural "
        "beam/path-score consistency. On real requests, pre-gate probabilities are unchanged; "
        "old path probability ratios obey the 0.8/0.9 minimum budgets. All three seeds reload "
        "identical recommendations and rollback to exactly the base scores. Nothing is written "
        "into the shared backbone. GenRecEdit rollback also bypasses per-prefix BF16 scoring "
        "to avoid numerical differences from the base teacher path.",
        "",
        f"The v2 backbone has {summary['cost']['base_parameters']:,} parameters. GenRecEdit "
        f"stores {summary['cost']['genrecedit_edit_parameters']:,} edited FFN values (~2 MiB). "
        f"v1 ScopeRec stores {summary['cost']['scoperec_v1_parameters']:,} patch parameters. "
        f"ScopeRec-C adds zero trainable parameters, but retains "
        f"{summary['cost']['scoperec_content_all_vector_bytes'] / 2**20:.2f} MiB of item vectors "
        f"in this implementation, including "
        f"{summary['cost']['scoperec_content_new_vector_bytes'] / 2**20:.2f} MiB of new vectors. "
        "Its ~0.7 KiB configuration artifact is not its total memory footprint.",
        "",
        f"GenRecEdit key/covariance collection and target optimization took "
        f"{summary['cost']['genrecedit_key_covariance_and_z_optimization_seconds']['mean']:.2f} "
        "seconds on average per final seed. Matrix solving, loading, raw-data preparation and "
        "text encoding are separate. Content mixture construction takes a fraction of a second "
        "with precomputed vectors; do not compare that with end-to-end fitting as a speedup claim.",
        "",
        "Synchronized one-GPU serving microbenchmark, 15 repeats after warmup; excludes "
        "disk loading, tree construction, history tokenization and target scoring:",
        "",
        "| Method | Batch 1 median ms | Batch 32 median ms |",
        "| --- | ---: | ---: |",
    ]
    timing = summary["integrity_and_latency"]
    lines.append(
        f"| TIGER | {timing['tiger_latency']['batch_1']['median_batch_ms']:.2f} | "
        f"{timing['tiger_latency']['batch_32']['median_batch_ms']:.2f} |"
    )
    for method in ("genrecedit", "scoperec_content", "scoperec_content_balanced"):
        measure = timing["methods"][method]["latency"]
        lines.append(
            f"| {LABELS_V2[method]} | {measure['batch_1']['median_batch_ms']:.2f} | "
            f"{measure['batch_32']['median_batch_ms']:.2f} |"
        )
    lines += [
        "",
        "Content routing trades optimizer/storage complexity for extra query-time "
        "similarity work. The current implementation is intended for a few thousand newcomers, "
        "not claimed to scale unchanged to millions of active new items.",
        "",
        "## Additional Validation",
        "",
        "The remaining 7,904 requests from the 2018 validation sample were evaluated only after "
        "selection was locked, as a diagnostic. This is not a new independent time window:",
        "",
        "| Method | New R@20 (%) | Old NDCG@20 (%) | Overall NDCG@20 (%) |",
        "| --- | ---: | ---: | ---: |",
    ]
    for method, values in summary["reserved_validation"].items():
        lines.append(
            f"| {LABELS_V2[method]} | {100 * values['new']['recall@20']:.3f} | "
            f"{100 * values['old']['ndcg@20']:.3f} | "
            f"{100 * values['overall']['ndcg@20']:.3f} |"
        )
    lines += [
        "",
        "## Limitations and Reproduction",
        "",
        "Single category, reused validation period, 707 fresh-test new targets, three seeds, "
        "and 300k training subsample. The conclusions concern these fixed operating points. "
        "A larger hyperparameter search for GenRecEdit, other layer mappings or more edit steps "
        "may change the frontier. No full-paper numerical, online A/B or multi-category claim.",
        "",
        "The new balanced point is recommended for the project demo because its old-item "
        "loss is small while its new recall and overall mean metrics improve. The new-priority "
        "point is useful when cold-item coverage is the explicit objective. Neither is advertised "
        "as universally superior or no-forgetting.",
        "",
        "Run `bash scripts/reproduce_v2.sh verify` to check tests, both releases' artifact "
        "audits and recorded parity. `bash scripts/reproduce_v2.sh all` builds missing stages "
        "without overwriting locked final results. The official reference remains a downloaded "
        "local dependency; source packaging excludes that code and all weights/data.",
        "",
        "![Validation frontier](figures/validation_tradeoff.png)",
        "",
    ]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Generate the GenRecEdit/ScopeRec-C confirmation report."
    )
    parser.add_argument("--root", type=Path, default=Path.cwd())
    arguments = parser.parse_args()
    summary = compile_summary(arguments.root)
    write_json(arguments.root / "reports/v2/summary.json", summary)
    render_figures(arguments.root, summary)
    (arguments.root / "reports/v2/RESULTS.md").write_text(make_markdown(summary))
    print(
        json.dumps(
            {
                "report": "reports/v2/RESULTS.md",
                "test": summary["test"],
                "comparison_count": len(summary["comparisons"]),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
