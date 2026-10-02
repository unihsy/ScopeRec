import argparse
import hashlib
import json
import zipfile
from pathlib import Path

ROOT_FILES = (
    "README.md",
    "LICENSE",
    ".gitignore",
    "pyproject.toml",
    "constraints-runtime.txt",
)
SOURCE_PATTERNS = (
    "src/**/*.py",
    "scripts/*.py",
    "scripts/*.sh",
    "tests/*.py",
    "configs/*.json",
    ".github/workflows/*.yml",
    "docs/*.md",
    "docs/results/*.json",
    "docs/assets/*.png",
    "docs/assets/*.svg",
)
REPORT_FILES = ()


def source_files(root: Path) -> list[Path]:
    root = root.resolve()
    paths = {root / name for name in (*ROOT_FILES, *REPORT_FILES) if (root / name).is_file()}
    for pattern in SOURCE_PATTERNS:
        paths.update(path for path in root.glob(pattern) if path.is_file())
    for path in paths:
        if path.is_symlink() or not path.resolve().is_relative_to(root):
            raise ValueError(f"Refusing external/symlink source: {path.name}")
    return sorted(paths)


def package_source(root: Path, destination: Path) -> dict:
    root = root.resolve()
    paths = source_files(root)
    manifest = {
        "format": 1,
        "scope": "Source, configs, tests and aggregate reports only",
        "license_status": "Apache-2.0",
        "excluded": [
            "data",
            "weights",
            "request-level traces",
            "reference PDFs",
            "environment",
            "caches",
            "metadata examples",
            "intermediate research reports",
            "validation grids and local task notes",
        ],
        "files": [],
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".zip.part")
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in paths:
            relative = path.relative_to(root).as_posix()
            payload = path.read_bytes()
            manifest["files"].append(
                {
                    "path": relative,
                    "bytes": len(payload),
                    "sha256": hashlib.sha256(payload).hexdigest(),
                }
            )
            archive.writestr(f"ScopeRec/{relative}", payload)
        archive.writestr("ScopeRec/SOURCE_MANIFEST.json", json.dumps(manifest, indent=2) + "\n")
    with zipfile.ZipFile(temporary) as archive:
        if archive.testzip() is not None:
            raise ValueError("Source archive integrity check failed")
    temporary.replace(destination)
    return {
        "archive": str(destination),
        "files": len(paths),
        "bytes": destination.stat().st_size,
        "sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
        "published_remotely": False,
    }


def stage_source(root: Path, destination: Path) -> dict:
    root = root.resolve()
    destination = destination.absolute()
    if not destination.resolve().is_relative_to(root / "release"):
        raise ValueError("Stage must be inside this project's release directory")
    for parent in (destination, *destination.parents):
        if parent == root:
            break
        if parent.is_symlink():
            raise ValueError("Stage path cannot contain a symlink")
    manifest_path = destination / "SOURCE_MANIFEST.json"
    previous = {}
    if destination.exists():
        if not destination.is_dir():
            raise ValueError("Stage destination is not a directory")
        existing = [path for path in destination.rglob("*") if path.is_file() or path.is_symlink()]
        if existing and not manifest_path.is_file():
            raise ValueError("Existing stage has no ownership manifest; refusing overwrite")
        if manifest_path.is_file():
            manifest = json.loads(manifest_path.read_text())
            previous = {entry["path"]: entry for entry in manifest["files"]}
        for path in existing:
            if path.is_symlink():
                raise ValueError("Existing stage contains a symlink")
            if path == manifest_path:
                continue
            relative = path.relative_to(destination).as_posix()
            if relative not in previous:
                raise ValueError(f"Unknown staged file: {relative}; preserve it before rebuilding")
            if hashlib.sha256(path.read_bytes()).hexdigest() != previous[relative]["sha256"]:
                raise ValueError(f"Locally modified stage file: {relative}; refusing overwrite")
    paths = source_files(root)
    payloads = {path.relative_to(root).as_posix(): path.read_bytes() for path in paths}
    for relative in payloads:
        parent = (destination / relative).parent
        if parent.exists() and not parent.is_dir():
            raise ValueError(f"Staging path is blocked by a file: {relative}")
    destination.mkdir(parents=True, exist_ok=True)
    manifest = {
        "format": 1,
        "scope": "Curated source and final presentation only",
        "license_status": "Apache-2.0",
        "published_remotely": False,
        "files": [
            {"path": relative, "bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}
            for relative, payload in payloads.items()
        ],
    }
    for relative in previous.keys() - payloads.keys():
        path = destination / relative
        if path.exists():
            path.unlink()
    for relative, payload in payloads.items():
        path = destination / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    for entry in manifest["files"]:
        path = destination / entry["path"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != entry["sha256"]:
            raise ValueError("Staged source validation failed")
    return {
        "directory": str(destination),
        "files": len(paths),
        "bytes": sum(len(payload) for payload in payloads.values()),
        "published_remotely": False,
    }


def main():
    parser = argparse.ArgumentParser(description="Package source without data or model artifacts.")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, default=Path("dist/scoperec-source.zip"))
    parser.add_argument("--stage", type=Path)
    arguments = parser.parse_args()
    result = package_source(arguments.root, arguments.output)
    if arguments.stage is not None:
        destination = (
            arguments.stage if arguments.stage.is_absolute() else arguments.root / arguments.stage
        )
        result["stage"] = stage_source(arguments.root, destination)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
