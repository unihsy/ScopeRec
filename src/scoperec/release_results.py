import argparse
import copy
import hashlib
import json
import math
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MaxNLocator

RELEASED_METHODS = (
    "tiger",
    "genrecedit_3000",
    "genrecedit_1000",
    "scoperec_c",
    "uniform_ablation",
)
TEMPORAL_METHODS = (
    "tiger",
    "genrecedit",
    "genrecedit_balanced",
    "scoperec_v1",
    "scoperec_content",
    "scoperec_content_balanced",
    "content_retrieval",
)
LABELS = {
    "tiger": "TIGER",
    "genrecedit_3000": "GenRecEdit / 3000",
    "genrecedit_1000": "GenRecEdit / 1000",
    "scoperec_c": "ScopeRec-C",
    "uniform_ablation": "Uniform suffix control",
    "genrecedit": "GenRecEdit / new-priority",
    "genrecedit_balanced": "GenRecEdit / balanced",
    "scoperec_v1": "Original ScopeRec",
    "scoperec_content": "ScopeRec-C / new-priority",
    "scoperec_content_balanced": "ScopeRec-C / balanced",
    "content_retrieval": "Content retrieval",
}
COLORS = {
    "tiger": "#687582",
    "genrecedit_3000": "#b7475c",
    "genrecedit_1000": "#cf8796",
    "scoperec_c": "#07846d",
    "uniform_ablation": "#b2923c",
    "genrecedit": "#b7475c",
    "genrecedit_balanced": "#cf8796",
    "scoperec_v1": "#397bb5",
    "scoperec_content": "#07846d",
    "scoperec_content_balanced": "#399d88",
    "content_retrieval": "#b2923c",
}


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def source_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def curate_released(source):
    summary = {
        "name": "Released Software benchmark",
        "dataset": "Amazon Reviews 2023 / Software / official 5-core timestamp splits",
        "comparison_scope": "Released-code local comparison, not paper-table replication",
        "seeds": source["configuration"]["seeds"],
        "test": source["protocol"]["split_summaries"]["test"],
        "catalog_items": source["protocol"]["catalog_items"],
        "selected_settings": source["selection"]["selected"]["settings"],
        "definitions": {
            "prefix": "First three semantic tokens; matches the released evaluator",
            "exact": "All four SID tokens; exact catalog-item identity",
        },
        "methods": {
            method: {
                name: copy.deepcopy(source["methods"][method][name])
                for name in ("prefix", "exact", "invalid_full_sid_fraction")
            }
            for method in RELEASED_METHODS
        },
        "paired_intervals": source["paired_intervals"],
        "uncertainty": source["uncertainty"],
        "cost": {
            "additional_trainable_parameters": 0,
            "content_vector_bytes": source["protocol"]["catalog_items"] * 768 * 4,
            "base_parameters": source["baseline_training"]["2024"]["parameters"],
        },
        "limitations": [
            "Paper/release tokenizer and layer choices differ; published gains unreplicated",
            "Held-out users and cold target identities participate in released task construction",
            "Uniform suffix allocation is close; a distinct content-ranking gain is unproven",
            "Overall gain over TIGER has a paired interval including zero",
            "Single category; three fixed backbones, not independent cross-dataset validation",
        ],
    }
    summary["test"] = {
        "requests": summary["test"]["tokenized_requests"],
        "cold_requests": summary["test"]["cold_target_requests"],
        "cold_fraction": summary["test"]["cold_target_fraction"],
        "cold_requests_seen_in_training_history": summary["test"][
            "cold_targets_seen_in_training_history"
        ],
    }
    return summary


def curate_temporal(source):
    chosen_names = ("scoperec_content", "scoperec_content_balanced")
    return {
        "name": "2020 temporal confirmation",
        "dataset": "Amazon Reviews 2023 / Software / custom 0-core global-time protocol",
        "comparison_scope": "Separate protocol; never pool with the released benchmark",
        "seeds": source["seeds"],
        "test": source["test"],
        "selected_settings": {
            name: source["selection"]["chosen"][name]["settings"] for name in chosen_names
        },
        "definitions": {"new": "Current arrival-batch targets", "old": "Other catalog targets"},
        "methods": {
            method: {
                name: copy.deepcopy(source["methods"][method][name])
                for name in ("overall", "new", "old")
            }
            for method in TEMPORAL_METHODS
        },
        "paired_intervals": source["comparisons"],
        "uncertainty": source["uncertainty"],
        "cost": {
            "additional_trainable_parameters": 0,
            "content_vector_bytes": source["cost"]["scoperec_content_all_vector_bytes"],
            "base_parameters": source["cost"]["base_parameters"],
            "serving_seed17": {
                "tiger": source["integrity_and_latency"]["tiger_latency"],
                **{
                    name: source["integrity_and_latency"]["methods"][name]["latency"]
                    for name in chosen_names
                },
            },
        },
        "limitations": [
            "Balanced overall NDCG gain over TIGER is not statistically established",
            "New-priority point sacrifices old-item ranking; no no-forgetting guarantee",
            "Validation period was previously used; final 2020 window was locked before evaluation",
        ],
    }


def validate_statistic(statistic):
    values = np.asarray(statistic["per_seed"], dtype=float)
    if len(values) < 2 or not np.isfinite(values).all():
        raise ValueError("Expected at least two finite measured seeds")
    if not math.isclose(statistic["mean"], float(values.mean()), abs_tol=1e-12, rel_tol=0):
        raise ValueError("Published mean differs from recorded seed measurements")
    if not math.isclose(statistic["std"], float(values.std(ddof=1)), abs_tol=1e-12, rel_tol=0):
        raise ValueError("Published variability differs from recorded seed measurements")


def validate_results(payload):
    count = 0

    def visit(value):
        nonlocal count
        if isinstance(value, dict):
            if all(name in value for name in ("mean", "std", "per_seed")):
                validate_statistic(value)
                count += 1
            else:
                for child in value.values():
                    visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    if payload.get("schema_version") != 1:
        raise ValueError("Unknown public-results schema")
    visit(payload["protocols"])
    return count


def export_results(root, destination):
    released_path = root / "reports/official_benchmark/summary.json"
    temporal_path = root / "reports/v2/summary.json"
    audits = {
        "released": read_json(root / "reports/official_benchmark/audit.json"),
        "temporal": read_json(root / "reports/v2/audit.json"),
    }
    if not all(audit["passed"] for audit in audits.values()):
        raise ValueError("Do not publish results that failed artifact audit")
    payload = {
        "schema_version": 1,
        "project": "ScopeRec-C",
        "presentation_policy": "Measured values unchanged; display rounding only",
        "protocols": {
            "released": curate_released(read_json(released_path)),
            "temporal": curate_temporal(read_json(temporal_path)),
        },
        "provenance": {
            "released_summary_sha256": source_hash(released_path),
            "temporal_summary_sha256": source_hash(temporal_path),
            "released_final_evaluations_audited": audits["released"]["final_evaluations_checked"],
            "temporal_final_evaluations_audited": audits["temporal"][
                "confirmation_evaluations_checked"
            ],
            "note": "Run artifacts stay local; figures use this curated final data",
        },
    }
    validate_results(payload)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def draw_panel(axis, methods, statistics, title, unit, labels=False):
    scale = 100 if unit == "percent" else 1
    maximum = max(max(statistics[method]["per_seed"]) * scale for method in methods)
    ceiling = maximum * 1.45 if maximum else 1
    positions = np.arange(len(methods))
    means = [statistics[method]["mean"] * scale for method in methods]
    axis.barh(
        positions, means, height=0.48, color=[COLORS[method] for method in methods], alpha=0.9
    )
    for index, method in enumerate(methods):
        values = np.asarray(statistics[method]["per_seed"]) * scale
        jitter = np.linspace(-0.075, 0.075, len(values))
        axis.scatter(
            values,
            index + jitter,
            s=20,
            color="#203342",
            edgecolors="white",
            linewidths=0.65,
            zorder=4,
        )
        formatted = f"{means[index]:.3f}%" if unit == "percent" else f"{means[index]:.5f}"
        axis.text(
            max(means[index], float(values.max())) + ceiling * 0.028,
            index,
            formatted,
            va="center",
            color="#233b48",
            fontsize=10.5,
            fontweight="semibold",
        )
    axis.set_xlim(0, ceiling)
    axis.set_ylim(len(methods) - 0.48, -0.65)
    axis.set_yticks(positions, [LABELS[method] for method in methods] if labels else [])
    axis.set_title(title, loc="left", fontsize=12, fontweight="bold", pad=16, color="#203342")
    axis.tick_params(axis="both", length=0, pad=8, labelsize=10, colors="#536574")
    axis.xaxis.set_major_locator(MaxNLocator(4))
    axis.xaxis.set_major_formatter(
        FuncFormatter(
            lambda value, position: f"{value:.1f}" if unit == "percent" else f"{value:.2f}"
        )
    )
    axis.grid(axis="x", color="#e9eef1", linewidth=0.8)
    axis.set_axisbelow(True)
    for spine in axis.spines.values():
        spine.set_visible(False)


def make_figure(payload, protocol):
    data = payload["protocols"][protocol]
    if protocol == "released":
        methods = RELEASED_METHODS
        panels = (
            ("Cold Recall@20 (%)", "exact", "cold", "recall@20", "percent"),
            ("Overall NDCG@20", "exact", "overall", "ndcg@20", "ndcg"),
            ("Warm NDCG@20", "exact", "warm", "ndcg@20", "ndcg"),
        )
        title = "Released Software benchmark"
        subtitle = "9,386 requests | 2,226 cold targets | exact four-token item matching"
        footer = (
            "Bars: three-seed mean. Dots: individual training seeds. All axes start at zero.\n"
            "Local released-code-compatible baselines; not published-paper numerical replication."
        )
    else:
        methods = TEMPORAL_METHODS[:-1]
        panels = (
            ("New-item Recall@20 (%)", None, "new", "recall@20", "percent"),
            ("Old-item NDCG@20", None, "old", "ndcg@20", "ndcg"),
            ("Overall NDCG@20", None, "overall", "ndcg@20", "ndcg"),
        )
        title = "Temporal confirmation: coverage and retention"
        subtitle = "2020 final window | 12,000 requests | parameters fixed before evaluation"
        footer = (
            "Bars: three-seed mean. Dots: individual seeds. Separate protocol from main chart.\n"
            "Balanced overall gain versus TIGER is small; its paired interval includes zero."
        )
    with plt.rc_context(
        {
            "font.family": "DejaVu Sans",
            "svg.fonttype": "none",
            "svg.hashsalt": "scoperec-final-charts",
            "font.size": 10,
        }
    ):
        figure, axes = plt.subplots(1, 3, figsize=(15.4, 5.7), facecolor="white")
        figure.subplots_adjust(left=0.205, right=0.975, top=0.73, bottom=0.19, wspace=0.28)
        figure.text(
            0.034,
            0.94,
            "SCOPEREC-C / FINAL RESULTS",
            fontsize=10,
            color="#07846d",
            fontweight="bold",
        )
        figure.text(0.034, 0.875, title, fontsize=21, color="#203342", fontweight="bold")
        figure.text(0.034, 0.82, subtitle, fontsize=11, color="#536574")
        for panel_index, (heading, definition, group, metric, unit) in enumerate(panels):
            statistics = {}
            for method in methods:
                measurements = data["methods"][method]
                if definition:
                    measurements = measurements[definition]
                statistics[method] = measurements[group][metric]
            draw_panel(
                axes[panel_index], methods, statistics, heading, unit, labels=panel_index == 0
            )
        figure.text(0.034, 0.065, footer, fontsize=9.5, color="#536574", linespacing=1.6)
    return figure


def render_assets(payload, directory):
    validate_results(payload)
    directory.mkdir(parents=True, exist_ok=True)
    data_sha = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    paths = []
    for protocol, name in (("released", "benchmark_final"), ("temporal", "temporal_final")):
        figure = make_figure(payload, protocol)
        for extension in ("png", "svg"):
            path = directory / f"{name}.{extension}"
            metadata = {"Description": f"Final measured results, canonical data SHA-256 {data_sha}"}
            if extension == "svg":
                metadata["Date"] = None
            with plt.rc_context({"svg.fonttype": "none", "svg.hashsalt": "scoperec-final-charts"}):
                figure.savefig(path, dpi=200, facecolor="white", metadata=metadata)
            paths.append(str(path))
        plt.close(figure)
    return paths


def main():
    parser = argparse.ArgumentParser(
        description="Curate measured final results and render release charts."
    )
    parser.add_argument("stage", choices=("export", "figures", "verify", "all"))
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--data", type=Path, default=Path("docs/results/final_metrics.json"))
    parser.add_argument("--assets", type=Path, default=Path("docs/assets"))
    arguments = parser.parse_args()
    data_path = arguments.root / arguments.data
    if arguments.stage in ("export", "all"):
        payload = export_results(arguments.root, data_path)
    else:
        payload = read_json(data_path)
    checked = validate_results(payload)
    figures = (
        render_assets(payload, arguments.root / arguments.assets)
        if arguments.stage in ("figures", "all")
        else []
    )
    print(
        json.dumps(
            {
                "measured_statistics_verified": checked,
                "figures": figures,
                "results": str(data_path),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
