import argparse
import gzip
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

CHUNK_SIZE = 1024 * 1024
RANGE_SIZE = 2 * CHUNK_SIZE
TRANSFER_ATTEMPTS = 5


def transfer_archive(source: dict, temporary: Path, session: requests.Session) -> dict:
    expected_bytes = source["expected_bytes"]
    metadata = {}
    failures = 0
    while True:
        offset = temporary.stat().st_size if temporary.exists() else 0
        if offset == expected_bytes:
            return metadata
        if offset > expected_bytes:
            raise ValueError(f"Partial download exceeded pinned size for {temporary.name}")
        range_end = min(offset + RANGE_SIZE, expected_bytes) - 1
        headers = {"Accept-Encoding": "identity", "Range": f"bytes={offset}-{range_end}"}
        try:
            with session.get(
                source["url"], stream=True, timeout=(15, 45), headers=headers
            ) as response:
                response.raise_for_status()
                if response.status_code == 206:
                    expected_range = f"bytes {offset}-{range_end}/{expected_bytes}"
                    if response.headers.get("Content-Range") != expected_range:
                        raise ValueError(f"Unexpected Content-Range for {source['filename']}")
                    mode = "ab" if offset else "wb"
                    expected_end = range_end + 1
                elif response.status_code == 200:
                    offset = 0
                    mode = "wb"
                    expected_end = expected_bytes
                else:
                    raise ValueError(f"Expected HTTP 200 or 206 for {source['filename']}")
                received_bytes = offset
                with temporary.open(mode) as stream:
                    for chunk in response.iter_content(chunk_size=64 * 1024):
                        received_bytes += len(chunk)
                        if received_bytes > expected_end:
                            raise ValueError(
                                f"Download exceeded pinned size for {source['filename']}"
                            )
                        stream.write(chunk)
                metadata = {
                    "etag": response.headers.get("ETag"),
                    "last_modified": response.headers.get("Last-Modified"),
                }
                if received_bytes != expected_end:
                    raise requests.ConnectionError(
                        f"Size mismatch for {source['filename']}: "
                        f"{received_bytes} != {expected_end}"
                    )
                failures = 0
                print(
                    f"{source['filename']}: {received_bytes / CHUNK_SIZE:.1f} / "
                    f"{expected_bytes / CHUNK_SIZE:.1f} MiB", flush=True,
                )
        except (
            requests.ConnectionError,
            requests.Timeout,
            requests.exceptions.ChunkedEncodingError,
        ):
            failures += 1
            if failures == TRANSFER_ATTEMPTS:
                raise
            print(f"Transfer interrupted; retry {failures}/{TRANSFER_ATTEMPTS - 1}", flush=True)


def validate_archive(path: Path, source: dict) -> dict:
    size = path.stat().st_size
    if size != source["expected_bytes"]:
        raise ValueError(f"Size mismatch for {path.name}: {size} != {source['expected_bytes']}")
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    if source.get("sha256") and digest != source["sha256"]:
        raise ValueError(f"SHA-256 mismatch for {path.name}")
    with gzip.open(path, "rb") as stream:
        while stream.read(CHUNK_SIZE):
            pass
    return {"bytes": size, "sha256": digest, "gzip_integrity_passed": True}


def download_source(source: dict, output_directory: Path, session: requests.Session) -> dict:
    filename = source["filename"]
    if Path(filename).name != filename or filename in (".", ".."):
        raise ValueError("Source filename must be a plain filename")
    output_directory.mkdir(parents=True, exist_ok=True)
    destination = output_directory / filename
    if destination.exists():
        verified = validate_archive(destination, source)
        print(f"Verified existing {filename}", flush=True)
        return {**source, **verified, "reused": True}

    required_space = source["expected_bytes"] + 256 * CHUNK_SIZE
    if shutil.disk_usage(output_directory).free < required_space:
        raise OSError(f"Insufficient free disk space for {filename}")

    temporary = destination.with_name(destination.name + ".part")
    print(f"Downloading {filename} ({source['expected_bytes'] / CHUNK_SIZE:.1f} MiB)", flush=True)
    metadata = transfer_archive(source, temporary, session)
    verified = validate_archive(temporary, source)
    temporary.replace(destination)
    print(f"Verified {filename}: {verified['sha256']}", flush=True)
    return {**source, **metadata, **verified, "reused": False}


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and verify pinned Amazon data archives.")
    parser.add_argument("--config", type=Path, default=Path("configs/software.json"))
    parser.add_argument(
        "--output-dir", type=Path, default=Path("data/raw/amazon_reviews_2023/Software")
    )
    parser.add_argument(
        "--manifest", type=Path, default=Path("reports/software_download_manifest.json")
    )
    arguments = parser.parse_args()
    configuration = json.loads(arguments.config.read_text(encoding="utf-8"))
    retry = Retry(total=3, backoff_factor=1, status_forcelist=(429, 500, 502, 503, 504))
    with requests.Session() as session:
        session.mount("https://", HTTPAdapter(max_retries=retry))
        files = [
            download_source(source, arguments.output_dir, session)
            for source in configuration["sources"]
        ]
    manifest = {
        "dataset": configuration["dataset"],
        "category": configuration["category"],
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "configuration_sha256": hashlib.sha256(arguments.config.read_bytes()).hexdigest(),
        "files": files,
    }
    arguments.manifest.parent.mkdir(parents=True, exist_ok=True)
    arguments.manifest.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Manifest: {arguments.manifest}")


if __name__ == "__main__":
    main()