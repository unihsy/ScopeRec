import hashlib
import json
import shutil
import subprocess
import zipfile
from pathlib import Path

import pytest

from scoperec.package import package_source, source_files, stage_source


def test_package_excludes_data_weights_traces_and_environment(tmp_path):
    for name in (
        "README.md",
        "LICENSE",
        "src/scoperec/model.py",
        "configs/experiment.json",
        "reports/test/seed_17/local.json",
        "reports/test/seed_17/local.pt",
        "reports/test/seed_17/local.npz",
        "data/private.json",
        "artifacts/model.pt",
        ".venv/private.py",
        "GenRecEdit.pdf",
        "reports/failure_cases.json",
        "reports/environment.json",
        "todo.md",
        "docs/RESULTS.md",
        "docs/results/final_metrics.json",
        "docs/assets/final_comparison.png",
    ):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("example")
    destination = tmp_path / "dist/source.zip"
    result = package_source(tmp_path, destination)
    assert result["published_remotely"] is False
    with zipfile.ZipFile(destination) as archive:
        assert set(archive.namelist()) == {
            "ScopeRec/README.md",
            "ScopeRec/LICENSE",
            "ScopeRec/src/scoperec/model.py",
            "ScopeRec/configs/experiment.json",
            "ScopeRec/docs/RESULTS.md",
            "ScopeRec/docs/results/final_metrics.json",
            "ScopeRec/docs/assets/final_comparison.png",
            "ScopeRec/SOURCE_MANIFEST.json",
        }
        manifest = json.loads(archive.read("ScopeRec/SOURCE_MANIFEST.json"))
        assert manifest["license_status"] == "Apache-2.0"
        for record in manifest["files"]:
            payload = archive.read("ScopeRec/" + record["path"])
            assert hashlib.sha256(payload).hexdigest() == record["sha256"]


def test_package_rejects_source_symlinks(tmp_path):
    outside = tmp_path / "private.txt"
    outside.write_text("not distributable")
    project = tmp_path / "project"
    project.mkdir()
    (project / "README.md").symlink_to(outside)
    with pytest.raises(ValueError, match="symlink"):
        source_files(project)


def test_package_excludes_historical_summaries_and_completed_markers(tmp_path):
    for name in (
        "reports/v2/RESULTS.md",
        "reports/v2/summary.json",
        "reports/v2/test/seed_17/completed.json",
        "reports/v2/validation/selection.json",
        "artifacts/v2/upstream/genrecedit/editor.py",
    ):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("example")
    assert source_files(tmp_path) == []


def test_official_working_reports_are_not_public_release_documents(tmp_path):
    for name in (
        "reports/official_benchmark/RESULTS.md",
        "reports/official_benchmark/summary.json",
        "reports/official_benchmark/selection.json",
        "reports/official_benchmark/test/seed_2024/completed.json",
        "artifacts/official_benchmark/upstream/genrec/models/TIGER/model.py",
        "data/official_software/test.npz",
    ):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("example")
    assert source_files(tmp_path) == []


def test_stage_can_update_owned_copies_but_preserves_unrecognized_or_modified_files(tmp_path):
    (tmp_path / "README.md").write_text("first version")
    (tmp_path / "LICENSE").write_text("owner-selected license")
    stage = tmp_path / "release/github-ready"
    stage_source(tmp_path, stage)
    assert (stage / "LICENSE").read_bytes() == (tmp_path / "LICENSE").read_bytes()
    manifest = json.loads((stage / "SOURCE_MANIFEST.json").read_text())
    assert manifest["license_status"] == "Apache-2.0"
    (tmp_path / "README.md").write_text("second version")
    stage_source(tmp_path, stage)
    assert (stage / "README.md").read_text() == "second version"
    (stage / "my_notes.md").write_text("user content")
    with pytest.raises(ValueError, match="Unknown staged"):
        stage_source(tmp_path, stage)
    assert (stage / "my_notes.md").read_text() == "user content"
    (stage / "my_notes.md").unlink()
    (stage / "README.md").write_text("edited staged copy")
    with pytest.raises(ValueError, match="Locally modified"):
        stage_source(tmp_path, stage)
    assert (stage / "README.md").read_text() == "edited staged copy"


def test_stage_rejects_external_destinations_and_does_not_copy_working_outputs(tmp_path):
    (tmp_path / "README.md").write_text("public")
    (tmp_path / "reports").mkdir()
    (tmp_path / "reports/private.json").write_text("not public")
    with pytest.raises(ValueError, match="release directory"):
        stage_source(tmp_path, tmp_path / "other-project")
    result = stage_source(tmp_path, tmp_path / "release/github-ready")
    assert result["published_remotely"] is False
    assert not (tmp_path / "release/github-ready/reports").exists()


def test_gitignore_excludes_legacy_root_notes_but_keeps_public_docs(tmp_path):
    git = shutil.which("git")
    if git is None:
        pytest.skip("Git is required to validate publication ignore rules")
    root = Path(__file__).resolve().parents[1]
    (tmp_path / ".gitignore").write_bytes((root / ".gitignore").read_bytes())
    subprocess.run([git, "init", "--quiet", str(tmp_path)], check=True)
    result = subprocess.run(
        [
            git,
            "-C",
            str(tmp_path),
            "check-ignore",
            "--",
            "DATA.md",
            "ARCHITECTURE.md",
            "docs/DATA.md",
            "docs/ARCHITECTURE.md",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert set(result.stdout.splitlines()) == {"DATA.md", "ARCHITECTURE.md"}
