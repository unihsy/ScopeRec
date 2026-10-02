import argparse
import base64
import importlib
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from scoperec.prepare import file_fingerprint

REVISION = "e6878d9c7c6e57479e840ccb8c045b11a2bd69b5"
FILES = (
    "genrecedit/__init__.py",
    "genrecedit/editor.py",
    "genrecedit/hparams.py",
    "genrecedit/cov_cache.py",
    "genrecedit/io_utils.py",
    "util/nethook.py",
    "util/runningstats.py",
)

BENCHMARK_FILES = FILES + (
    "genrec/datasets/AmazonReviews2023/dataset.py",
    "genrec/datasets/AmazonReviews2023/config.yaml",
    "genrec/models/TIGER/tokenizer.py",
    "genrec/models/TIGER/model.py",
    "genrec/models/TIGER/config.yaml",
    "genrec/trainer.py",
    "genrec/evaluator.py",
    "genrec/default.yaml",
    "genrec/utils.py",
    "Scripts/rec_train.sh",
    "Scripts/prepare_data.sh",
    "Scripts/edit.sh",
    "prepare_edit_data.py",
)


def download_reference(destination: Path, files=FILES):
    destination.mkdir(parents=True, exist_ok=True)
    manifest = {"repository": "Starrylay/GenRecEdit", "revision": REVISION, "files": []}
    session = requests.Session()
    session.mount("https://", HTTPAdapter(max_retries=Retry(total=2, backoff_factor=0.5)))
    for name in files:
        path = destination / name
        if not path.is_file():
            try:
                response = session.get(
                    f"https://raw.githubusercontent.com/Starrylay/GenRecEdit/{REVISION}/{name}",
                    timeout=(10, 30),
                )
                response.raise_for_status()
                content = response.content
            except requests.RequestException:
                response = session.get(
                    f"https://api.github.com/repos/Starrylay/GenRecEdit/contents/{name}",
                    params={"ref": REVISION},
                    timeout=(10, 30),
                )
                response.raise_for_status()
                payload = response.json()
                if payload["encoding"] != "base64":
                    raise ValueError("Unexpected GitHub source encoding")
                content = base64.b64decode(payload["content"])
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
            print(f"Downloaded official reference: {name}", flush=True)
        manifest["files"].append(file_fingerprint(path))
    session.close()
    (destination / "source_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def load_reference(root: Path):
    for name in FILES:
        if not (root / name).is_file():
            raise FileNotFoundError("Download the pinned official source with scoperec.upstream")
    sys.path.insert(0, str(root.resolve()))
    bundle_name = "genrecedit.model_bundle"
    if bundle_name not in sys.modules:
        annotation_module = ModuleType(bundle_name)
        annotation_module.GenRecEditModelBundle = SimpleNamespace
        sys.modules[bundle_name] = annotation_module
    editor = importlib.import_module("genrecedit.editor")
    hparams = importlib.import_module("genrecedit.hparams")
    editor.tqdm = lambda iterable, **kwargs: QuietProgress(iterable)
    return editor.GenRecEdit, hparams.GenRecEditHyperParams


class QuietProgress:
    def __init__(self, iterable):
        self.iterable = iterable

    def __iter__(self):
        return iter(self.iterable)

    def set_postfix(self, **kwargs):
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Download fixed official GenRecEdit reference code."
    )
    parser.add_argument("--output", type=Path, default=Path("artifacts/v2/upstream"))
    parser.add_argument("--benchmark", action="store_true")
    arguments = parser.parse_args()
    if arguments.benchmark and arguments.output == Path("artifacts/v2/upstream"):
        parser.error("Use a separate --output directory to preserve the locked v2 source manifest")
    manifest = download_reference(
        arguments.output, BENCHMARK_FILES if arguments.benchmark else FILES
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
