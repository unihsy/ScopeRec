import argparse
import csv
import gzip
import html
import json
import re
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as parquet

from scoperec.experiment import write_json
from scoperec.prepare import file_fingerprint


def source_rows(path):
    with gzip.open(path, "rt", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        expected = {"user_id", "parent_asin", "rating", "timestamp", "history"}
        if set(reader.fieldnames or []) != expected:
            raise ValueError("Unexpected official split CSV schema")
        for ordinal, record in enumerate(reader):
            if record["history"]:
                record["source_row"] = ordinal
                yield record


def selected_test_users(records, fraction):
    if not 0 < fraction <= 1:
        raise ValueError("max_rows must be in (0, 1]")
    users = set()
    for count, row in enumerate(records, start=1):
        users.add(row["user_id"])
        if count > fraction * len(records):
            break
    return users


def clean_metadata_text(value):
    if isinstance(value, list):
        value = ", ".join(map(str, value))
    text = html.unescape(str(value or "")).strip()
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"[\n\t]", " ", text)
    text = re.sub(r" +", " ", text)
    return re.sub(r"[^\x00-\x7F]", " ", text)


def official_metadata(record):
    parts = []
    for field in ("title", "features", "categories", "description"):
        value = record.get(field, "")
        if isinstance(value, float):
            parts.append(str(value) + ". ")
        elif isinstance(value, list) and value:
            parts.append(", ".join(clean_metadata_text(item) for item in value) + ". ")
        else:
            parts.append(clean_metadata_text(value) + " ")
    return "".join(parts)


def request_arrays(rows, history_items, train=False):
    count = sum(len(row["sequence"]) - 1 for row in rows) if train else len(rows)
    history = np.full((count, history_items), -1, dtype=np.int32)
    target = np.empty(count, dtype=np.int32)
    users = np.empty(count, dtype=np.int64)
    timestamps = np.empty(count, dtype=np.int64)
    source_row = np.empty(count, dtype=np.int64)
    filled = 0
    for row in rows:
        sequence = row["sequence"]
        positions = range(1, len(sequence)) if train else [len(sequence) - 1]
        for position in positions:
            previous = sequence[max(0, position - history_items) : position]
            history[filled, : len(previous)] = previous
            target[filled] = sequence[position]
            users[filled] = row["user"]
            timestamps[filled] = row["timestamp"]
            source_row[filled] = row["source_row"]
            filled += 1
    return {
        "history": history,
        "target": target,
        "user": users,
        "source_row_timestamp": timestamps,
        "source_row": source_row,
    }


def prepare(raw, metadata_path, output, config):
    output.mkdir(parents=True, exist_ok=True)
    test_rows = list(source_rows(raw / "Software.test.csv.gz"))
    selected = selected_test_users(test_rows, config["max_rows"])
    item_lookup = {}
    user_lookup = {}
    splits = {}
    before_filter = {}
    for split in ("train", "valid", "test"):
        retained = []
        before_filter[split] = 0
        iterable = test_rows if split == "test" else source_rows(raw / f"Software.{split}.csv.gz")
        for record in iterable:
            before_filter[split] += 1
            for item in [record["parent_asin"], *record["history"].split(" ")]:
                if item not in item_lookup:
                    item_lookup[item] = len(item_lookup)
            user = record["user_id"]
            if user not in user_lookup:
                user_lookup[user] = len(user_lookup)
            if user in selected:
                sequence = [item_lookup[item] for item in record["history"].split(" ")]
                sequence.append(item_lookup[record["parent_asin"]])
                retained.append(
                    {
                        "user": user_lookup[user],
                        "sequence": sequence,
                        "timestamp": int(record["timestamp"]),
                        "source_row": record["source_row"],
                    }
                )
        splits[split] = retained
    train_targets = set(row["sequence"][-1] for row in splits["train"])
    train_items = set(item for row in splits["train"] for item in row["sequence"])
    train_mask = np.array([item in train_items for item in range(len(item_lookup))])
    np.save(output / "quantizer_train_mask.npy", train_mask)
    summaries = {}
    for split, rows in splits.items():
        arrays = request_arrays(rows, config["history_items"], train=split == "train")
        arrays["group"] = np.array(
            [0 if item in train_targets else 2 for item in arrays["target"]], dtype=np.int8
        )
        np.savez_compressed(output / f"{split}.npz", **arrays)
        unique_examples = np.unique(
            np.column_stack([arrays["user"], arrays["history"], arrays["target"]]), axis=0
        )
        summaries[split] = {
            "source_rows_after_history_filter": before_filter[split],
            "retained_rows": len(rows),
            "tokenized_requests": len(arrays["target"]),
            "users": len(np.unique(arrays["user"])),
            "unique_user_history_target_requests": len(unique_examples),
            "cold_target_requests": int(np.sum(arrays["group"] == 2)),
            "cold_target_fraction": float(np.mean(arrays["group"] == 2)),
            "cold_targets_seen_in_training_history": int(
                np.sum((arrays["group"] == 2) & train_mask[arrays["target"]])
            ),
        }
        if split != "train":
            groups = np.zeros(len(item_lookup), dtype=np.int8)
            groups[np.unique(arrays["target"][arrays["group"] == 2])] = 2
            np.save(output / f"{split}_groups.npy", groups)
        write_json(output / f"{split}_source_sequences.json", rows)
    metadata = {}
    with gzip.open(metadata_path, "rt", encoding="utf-8") as stream:
        for line in stream:
            record = json.loads(line)
            item = record["parent_asin"]
            if item in item_lookup:
                metadata[item] = {
                    "text": official_metadata(record),
                    "title": record.get("title", ""),
                }
    missing = sorted(set(item_lookup) - metadata.keys())
    if missing:
        raise ValueError(f"Official item catalog missing metadata for {len(missing)} items")
    catalog = [
        {"item_id": index, "parent_asin": item, **metadata[item]}
        for item, index in item_lookup.items()
    ]
    parquet.write_table(
        pa.Table.from_pylist(catalog), output / "catalog.parquet", compression="zstd"
    )
    report = {
        "configuration": config,
        "split_summaries": summaries,
        "selected_test_users": len(selected),
        "catalog_items": len(item_lookup),
        "training_target_items": len(train_targets),
        "quantizer_training_items": len(train_items),
        "source_files": [
            file_fingerprint(raw / f"Software.{split}.csv.gz")
            for split in ("train", "valid", "test")
        ],
        "caveats": [
            "Exact released test-prefix user filtering, including one extra row",
            "ID mapping and PCA see the whole published catalog",
            "Cold group uses filtered training row targets, not the union of training histories",
            "Expanded training prefixes intentionally preserve official duplicate examples",
            "Expanded-prefix timestamp is its source row timestamp, not the earlier event time",
            "5-core/test-user filtering are benchmark rules, not causal online preprocessing",
        ],
    }
    write_json(output / "protocol.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(
        description="Prepare the exact released Software benchmark rows."
    )
    parser.add_argument("--config", type=Path, default=Path("configs/official_software.json"))
    parser.add_argument("--raw", type=Path, default=Path("data/raw/official_software"))
    parser.add_argument(
        "--metadata",
        type=Path,
        default=Path("data/raw/amazon_reviews_2023/Software/meta_Software.jsonl.gz"),
    )
    parser.add_argument("--output", type=Path, default=Path("data/official_software"))
    arguments = parser.parse_args()
    print(
        json.dumps(
            prepare(
                arguments.raw,
                arguments.metadata,
                arguments.output,
                json.loads(arguments.config.read_text()),
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
