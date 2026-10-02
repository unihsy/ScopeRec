import copy
import hashlib
import json
import xml.etree.ElementTree as element_tree
from pathlib import Path

import matplotlib.pyplot as plt
import pytest
from PIL import Image

from scoperec.package import source_files
from scoperec.release_results import RELEASED_METHODS, make_figure, render_assets, validate_results


def statistic():
    return {"mean": 0.2, "std": 0.1, "per_seed": [0.1, 0.2, 0.3]}


def test_public_metrics_reject_altered_means_and_variability():
    payload = {"schema_version": 1, "protocols": {"released": {"measurement": statistic()}}}
    assert validate_results(payload) == 1
    for field in ("mean", "std"):
        changed = copy.deepcopy(payload)
        changed["protocols"]["released"]["measurement"][field] += 0.01
        with pytest.raises(ValueError, match="differs"):
            validate_results(changed)


def test_main_figure_includes_control_and_zero_based_axes():
    method = {
        "exact": {
            "cold": {"recall@20": statistic()},
            "overall": {"ndcg@20": statistic()},
            "warm": {"ndcg@20": statistic()},
        }
    }
    payload = {
        "schema_version": 1,
        "protocols": {
            "released": {"methods": {name: copy.deepcopy(method) for name in RELEASED_METHODS}}
        },
    }
    figure = make_figure(payload, "released")
    try:
        assert len(figure.axes) == 3
        assert all(axis.get_xlim()[0] == 0 for axis in figure.axes)
        labels = [label.get_text() for label in figure.axes[0].get_yticklabels()]
        assert "Uniform suffix control" in labels
        assert len(figure.axes[0].patches) == len(RELEASED_METHODS)
        figure.canvas.draw()
        renderer = figure.canvas.get_renderer()
        bounds = figure.bbox
        for axis in figure.axes:
            for label in axis.texts:
                assert bounds.contains(*label.get_window_extent(renderer).get_points()[0])
                assert bounds.contains(*label.get_window_extent(renderer).get_points()[1])
    finally:
        plt.close(figure)


def test_public_readme_table_matches_final_measured_data():
    root = Path(__file__).resolve().parents[1]
    payload = json.loads((root / "docs/results/final_metrics.json").read_text())
    assert validate_results(payload) > 300
    labels = {
        "tiger": "TIGER",
        "genrecedit_3000": "GenRecEdit, lambda 3000",
        "genrecedit_1000": "GenRecEdit, lambda 1000",
        "scoperec_c": "ScopeRec-C",
        "uniform_ablation": "Uniform suffix control",
    }
    lines = (root / "README.md").read_text().replace("**", "").splitlines()
    for method, label in labels.items():
        measurements = payload["protocols"]["released"]["methods"][method]["exact"]
        line = next(line for line in lines if line.startswith(f"| {label} |"))
        columns = [column.strip() for column in line.split("|")[1:-1]]
        assert columns[1:] == [
            f"{100 * measurements['cold']['recall@20']['mean']:.3f}%",
            f"{measurements['warm']['ndcg@20']['mean']:.5f}",
            f"{measurements['overall']['ndcg@20']['mean']:.5f}",
        ]


def test_final_assets_have_valid_dimensions_and_data_fingerprints():
    root = Path(__file__).resolve().parents[1]
    payload = json.loads((root / "docs/results/final_metrics.json").read_text())
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    for name in ("benchmark_final", "temporal_final"):
        with Image.open(root / f"docs/assets/{name}.png") as image:
            assert image.width >= 2400
            assert 2.0 < image.width / image.height < 3.0
            assert digest in image.info["Description"]
            image.verify()
        svg_path = root / f"docs/assets/{name}.svg"
        document = element_tree.parse(svg_path)
        assert document.getroot().tag.endswith("svg")
        assert digest in svg_path.read_text()


def test_repeat_plot_export_has_identical_png_and_svg_bytes(tmp_path):
    root = Path(__file__).resolve().parents[1]
    payload = json.loads((root / "docs/results/final_metrics.json").read_text())
    first = tmp_path / "first"
    second = tmp_path / "second"
    render_assets(payload, first)
    render_assets(payload, second)
    for path in first.iterdir():
        assert path.read_bytes() == (second / path.name).read_bytes(), path.name


def test_owner_architecture_image_is_valid_linked_and_in_source_release():
    root = Path(__file__).resolve().parents[1]
    path = root / "docs/assets/architecture.png"
    with Image.open(path) as image:
        assert image.format == "PNG"
        assert image.width >= 1600 and image.height >= 900
        assert 1.6 < image.width / image.height < 2.0
        image.verify()
    assert path in source_files(root)
    assert "![ScopeRec-C" in (root / "README.md").read_text()
    assert "](docs/assets/architecture.png)" in (root / "README.md").read_text()
    assert "](assets/architecture.png)" in (root / "docs/ARCHITECTURE.md").read_text()
