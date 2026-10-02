import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path

import duckdb
import numpy as np
import pyarrow.parquet as parquet


def timestamp_ms(value: str) -> int:
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        raise ValueError("Experiment dates must include an explicit timezone")
    return int(parsed.timestamp() * 1000)


def earlier_histories(events: list[tuple[int, int]], limit: int) -> list[list[int]]:
    if limit < 1:
        raise ValueError("History limit must be positive")
    if events != sorted(events):
        raise ValueError("Events must be sorted by timestamp and item")
    history = []
    pending = []
    previous_time = None
    output = []
    for timestamp, item in events:
        if timestamp != previous_time:
            history.extend(pending)
            history = history[-limit:]
            pending = []
        output.append(history.copy())
        pending.append(item)
        previous_time = timestamp
    return output


def classify_items(first_seen: np.ndarray, base: int, start: int, edit: int) -> np.ndarray:
    if not base <= start < edit:
        raise ValueError("Expected base <= arrival_start < edit_at")
    group = np.full(len(first_seen), -1, dtype=np.int8)
    group[first_seen < base] = 0
    group[(first_seen >= base) & (first_seen < start)] = 1
    group[(first_seen >= start) & (first_seen < edit)] = 2
    return group


def save_requests(table, path: Path, history_items: int) -> dict:
    histories = table.column("history").to_pylist()
    padded = np.full((len(histories), history_items), -1, dtype=np.int32)
    for row_index, history in enumerate(histories):
        values = history[-history_items:]
        padded[row_index, :len(values)] = values
    arrays = {
        "history": padded,
        "target": table.column("item_id").to_numpy().astype(np.int32),
        "timestamp": table.column("timestamp").to_numpy().astype(np.int64),
        "user": table.column("user_key").to_numpy().astype(np.uint64),
    }
    if "group_id" in table.column_names:
        arrays["group"] = table.column("group_id").to_numpy().astype(np.int8)
    np.savez_compressed(path, **arrays)
    return {
        "requests": len(histories),
        "users": int(len(np.unique(arrays["user"]))),
        "minimum_timestamp": int(arrays["timestamp"].min()) if histories else None,
        "maximum_timestamp": int(arrays["timestamp"].max()) if histories else None,
        "group_counts": {
            str(group): int(np.sum(arrays["group"] == group))
            for group in np.unique(arrays.get("group", []))
        },
    }


def build_protocol(processed: Path, output: Path, config: dict) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    base = timestamp_ms(config["base_cutoff"])
    history_items = config["history_items"]
    sampling_seed = int(config["sampling_seed"])
    report = {"configuration": config, "groups": {"0": "base_old", "1": "prior_new", "2": "new"}}
    connection = duckdb.connect(config={"threads": 4, "memory_limit": "6GB"})
    with connection:
        connection.read_parquet(str(processed / "interactions.parquet")).create_view("events")
        connection.read_parquet(str(processed / "items.parquet")).create_view("metadata")
        connection.execute("""
            CREATE TABLE item_map AS
            SELECT CAST(row_number() OVER (ORDER BY first_seen, parent_asin) - 1 AS INTEGER)
                       AS item_id,
                   parent_asin, first_seen, title, features, description
            FROM (SELECT parent_asin, min(timestamp) AS first_seen FROM events GROUP BY parent_asin)
            LEFT JOIN metadata USING (parent_asin)
        """)
        item_table = connection.execute("SELECT * FROM item_map ORDER BY item_id").to_arrow_table()
        parquet.write_table(item_table, output / "catalog.parquet", compression="zstd")
        report["base_items"] = int(np.sum(item_table["first_seen"].to_numpy() < base))
        print("Constructing strictly earlier histories...", flush=True)
        connection.execute(f"""
            CREATE TABLE requests AS
            SELECT hash(user_id) AS user_key, item_id, timestamp, first_seen,
                   list_slice(list(item_id ORDER BY timestamp, item_id) OVER (
                       PARTITION BY user_id ORDER BY timestamp
                       RANGE BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
                   ), -{int(history_items)}, -1) AS history
            FROM events JOIN item_map USING (parent_asin)
        """)
        connection.execute("DELETE FROM requests WHERE history IS NULL OR len(history) = 0")
        training_count = connection.execute(
            "SELECT count(*) FROM requests WHERE timestamp < ?", [base]
        ).fetchone()[0]
        report["eligible_pre_base_requests"] = training_count
        for split, predicate, limit in (
            ("train", "user_key % 20 != 0", config["max_train_requests"]),
            ("holdout", "user_key % 20 = 0", config["max_holdout_requests"]),
        ):
            table = connection.execute(f"""
                SELECT user_key, item_id, timestamp, history FROM requests
                WHERE timestamp < ? AND {predicate}
                ORDER BY hash(user_key, item_id, timestamp, {sampling_seed})
                LIMIT {int(limit)}
            """, [base]).to_arrow_table()
            report[split] = save_requests(table, output / f"{split}.npz", history_items)
        for name, window in config["windows"].items():
            start = timestamp_ms(window["arrival_start"])
            edit = timestamp_ms(window["edit_at"])
            end = timestamp_ms(window["end"])
            if not base <= start < edit < end:
                raise ValueError(f"Invalid chronological window: {name}")
            mask = classify_items(item_table["first_seen"].to_numpy(), base, start, edit)
            np.save(output / f"{name}_groups.npy", mask)
            counts = connection.execute("""
                SELECT count(*), count(*) FILTER (WHERE first_seen >= ?)
                FROM requests WHERE timestamp >= ? AND timestamp < ?
            """, [edit, edit, end]).fetchone()
            table = connection.execute(f"""
                SELECT user_key, item_id, timestamp, history,
                       CASE WHEN first_seen < {base} THEN 0
                            WHEN first_seen < {start} THEN 1 ELSE 2 END AS group_id
                FROM requests WHERE timestamp >= ? AND timestamp < ? AND first_seen < ?
                ORDER BY hash(user_key, item_id, timestamp, {sampling_seed})
                LIMIT {int(config['max_eval_requests'])}
            """, [edit, end, edit]).to_arrow_table()
            report[name] = {
                **save_requests(table, output / f"{name}.npz", history_items),
                "all_sequential_requests_in_window": counts[0],
                "requests_outside_catalog": counts[1],
                "catalog_items": int(np.sum(mask >= 0)),
                "new_items": int(np.sum(mask == 2)),
            }
            few_shot = connection.execute("""
                SELECT user_key, item_id, timestamp, history
                FROM requests WHERE first_seen >= ? AND first_seen < ?
                     AND timestamp > first_seen AND timestamp < ?
                QUALIFY row_number() OVER (PARTITION BY item_id ORDER BY timestamp, user_key) <= ?
                ORDER BY timestamp, item_id, user_key
            """, [start, edit, edit, config["adaptation"]["few_shot_per_item"]]).to_arrow_table()
            report[f"{name}_few_shot"] = save_requests(
                few_shot, output / f"{name}_few_shot.npz", history_items
            )
    report["configuration_sha256"] = hashlib.sha256(
        json.dumps(config, sort_keys=True).encode()
    ).hexdigest()
    report["history_policy"] = "Only strictly earlier events; latest 20 items; no rating filter"
    report["sampling_policy"] = "Fixed hash sampling per window, independent of target group"
    (output / "protocol.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a fixed temporal cold-start protocol.")
    parser.add_argument("--config", type=Path, default=Path("configs/experiment.json"))
    parser.add_argument("--processed", type=Path, default=Path("data/processed/software"))
    parser.add_argument("--output", type=Path, default=Path("data/experiment"))
    arguments = parser.parse_args()
    report = build_protocol(
        arguments.processed, arguments.output, json.loads(arguments.config.read_text())
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()