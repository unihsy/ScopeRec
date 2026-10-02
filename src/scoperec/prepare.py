import argparse
import gzip
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import pyarrow as pa
import pyarrow.csv as arrow_csv
import pyarrow.parquet as parquet

INTERACTION_SCHEMA = pa.schema(
    [
        ("user_id", pa.string()),
        ("parent_asin", pa.string()),
        ("rating", pa.float32()),
        ("timestamp", pa.int64()),
    ]
)
TEXT_FIELDS = ("title", "features", "description")
ITEM_SCHEMA = pa.schema(
    [("parent_asin", pa.string()), *[(field, pa.string()) for field in TEXT_FIELDS]]
)


def convert_interactions(source: Path, destination: Path) -> int:
    temporary = destination.with_name(destination.name + ".part")
    destination.parent.mkdir(parents=True, exist_ok=True)
    options = arrow_csv.ConvertOptions(column_types=dict(zip(
        INTERACTION_SCHEMA.names, INTERACTION_SCHEMA.types, strict=True
    )))
    row_count = 0
    with arrow_csv.open_csv(source, convert_options=options) as reader:
        if reader.schema.names != INTERACTION_SCHEMA.names:
            raise ValueError(f"Unexpected interaction columns: {reader.schema.names}")
        with parquet.ParquetWriter(temporary, INTERACTION_SCHEMA, compression="zstd") as writer:
            for batch in reader:
                if any(column.null_count for column in batch.columns):
                    raise ValueError("Null interaction fields are not supported")
                writer.write_batch(batch)
                row_count += batch.num_rows
    if not row_count:
        raise ValueError("Interaction archive contains no rows")
    temporary.replace(destination)
    return row_count


def normalize_text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return " ".join(value.split())
    if isinstance(value, list):
        return " ".join(text for part in value if (text := normalize_text(part)))
    raise ValueError(f"Unexpected metadata text type: {type(value).__name__}")


def metadata_record(record: dict) -> dict:
    identifier = record.get("parent_asin")
    if not isinstance(identifier, str) or not identifier.strip():
        raise ValueError("Metadata requires a nonempty parent_asin")
    return {
        "parent_asin": identifier,
        **{field: normalize_text(record.get(field)) for field in TEXT_FIELDS},
    }


def convert_metadata(source: Path, destination: Path, batch_size: int = 2048) -> int:
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    temporary = destination.with_name(destination.name + ".part")
    destination.parent.mkdir(parents=True, exist_ok=True)
    row_count = 0
    pending = []
    with gzip.open(source, "rt", encoding="utf-8") as stream:
        with parquet.ParquetWriter(temporary, ITEM_SCHEMA, compression="zstd") as writer:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                try:
                    pending.append(metadata_record(json.loads(line)))
                except (ValueError, AttributeError) as error:
                    raise ValueError(f"Invalid metadata at line {line_number}: {error}") from error
                if len(pending) >= batch_size:
                    writer.write_table(pa.Table.from_pylist(pending, schema=ITEM_SCHEMA))
                    row_count += len(pending)
                    pending.clear()
            if pending:
                writer.write_table(pa.Table.from_pylist(pending, schema=ITEM_SCHEMA))
                row_count += len(pending)
    if not row_count:
        raise ValueError("Metadata archive contains no rows")
    temporary.replace(destination)
    return row_count


def query_records(connection: duckdb.DuckDBPyConnection, query: str) -> list[dict]:
    result = connection.execute(query)
    names = [column[0] for column in result.description]
    return [dict(zip(names, row, strict=True)) for row in result.fetchall()]


def utc_timestamp(milliseconds: int) -> str:
    return datetime.fromtimestamp(milliseconds / 1000, tz=timezone.utc).isoformat()


def profile_dataset(interactions: Path, items: Path, threads: int = 4) -> dict:
    if threads < 1:
        raise ValueError("threads must be positive")
    with duckdb.connect(config={"threads": threads, "memory_limit": "4GB"}) as connection:
        connection.read_parquet(str(interactions)).create_view("interactions")
        connection.read_parquet(str(items)).create_view("items")
        invalid = connection.execute("""
            SELECT count(*) FROM interactions
            WHERE user_id IS NULL OR trim(user_id) = ''
               OR parent_asin IS NULL OR trim(parent_asin) = ''
               OR timestamp IS NULL OR timestamp <= 0
               OR rating IS NULL OR NOT isfinite(rating) OR rating < 1 OR rating > 5
        """).fetchone()[0]
        if invalid:
            raise ValueError(f"Found {invalid} invalid interaction rows")
        duplicate_items = connection.execute("""
            SELECT count(*) - count(DISTINCT parent_asin) FROM items
        """).fetchone()[0]
        if duplicate_items:
            raise ValueError(f"Found {duplicate_items} duplicate metadata identifiers")
        totals = query_records(connection, """
            SELECT count(*) AS interactions,
                   count(DISTINCT user_id) AS users,
                   count(DISTINCT parent_asin) AS items,
                   min(timestamp) AS first_timestamp_ms,
                   max(timestamp) AS last_timestamp_ms,
                   count(*) - count(DISTINCT (user_id, parent_asin)) AS repeated_user_item_rows
            FROM interactions
        """)[0]
        totals["first_timestamp_utc"] = utc_timestamp(totals["first_timestamp_ms"])
        totals["last_timestamp_utc"] = utc_timestamp(totals["last_timestamp_ms"])
        history = query_records(connection, """
            WITH counts AS (SELECT user_id, count(*) AS events FROM interactions GROUP BY user_id)
            SELECT count(*) FILTER (WHERE events = 1) AS users_with_one_event,
                   count(*) FILTER (WHERE events >= 2) AS users_with_at_least_two_events,
                   count(*) FILTER (WHERE events >= 5) AS users_with_at_least_five_events,
                   sum(events - 1) AS targets_with_at_least_one_prior_event_upper_bound,
                   quantile_disc(events, [0.5, 0.9, 0.99]) AS event_count_p50_p90_p99,
                   max(events) AS maximum_events_per_user
            FROM counts
        """)[0]
        coverage = query_records(connection, """
            SELECT count(*) AS interaction_items,
                   count(*) FILTER (WHERE items.parent_asin IS NOT NULL) AS items_with_metadata,
                   count(*) FILTER (WHERE length(concat_ws(
                       '', items.title, items.features, items.description
                   )) > 0) AS items_with_text,
                   sum(events) FILTER (WHERE items.parent_asin IS NULL) AS events_missing_metadata
            FROM (SELECT parent_asin, count(*) AS events FROM interactions GROUP BY parent_asin)
            LEFT JOIN items USING (parent_asin)
        """)[0]
        coverage["events_missing_metadata"] = coverage["events_missing_metadata"] or 0
        coverage["metadata_rows"] = connection.execute("SELECT count(*) FROM items").fetchone()[0]
        yearly = query_records(connection, """
            SELECT year(epoch_ms(timestamp)) AS year, count(*) AS interactions,
                   count(DISTINCT user_id) AS users, count(DISTINCT parent_asin) AS items
            FROM interactions GROUP BY year ORDER BY year
        """)
        arrivals = query_records(connection, """
            SELECT year(epoch_ms(first_seen)) AS year, count(*) AS first_observed_items
            FROM (SELECT parent_asin, min(timestamp) AS first_seen
                  FROM interactions GROUP BY parent_asin)
            GROUP BY year ORDER BY year
        """)
    return {
        "totals": totals,
        "user_histories_full_period_diagnostic_only": history,
        "metadata_coverage": coverage,
        "yearly_interactions": yearly,
        "yearly_first_observed_items": arrivals,
        "text_fields": list(TEXT_FIELDS),
        "local_filters_applied": [],
        "split_created": False,
        "caveats": [
            "The source is the official deduplicated 0-core release, not raw review events.",
            "Review timestamps are proxies for interaction time, not purchase or click logs.",
            "Metadata has no historical versions; static text can still have temporal uncertainty.",
            "First observed interaction is not a true product launch date.",
            "Full-period counts are diagnostic only and must not select training users or items.",
            "Missing text or metadata does not remove interaction rows.",
            "Equal timestamps need explicit handling before constructing earlier-only histories.",
        ],
    }


def file_fingerprint(path: Path) -> dict:
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare and profile Software without future filters."
    )
    parser.add_argument(
        "--raw-dir", type=Path, default=Path("data/raw/amazon_reviews_2023/Software")
    )
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed/software"))
    parser.add_argument("--report", type=Path, default=Path("reports/software_profile.json"))
    parser.add_argument("--threads", type=int, default=4)
    arguments = parser.parse_args()
    interaction_source = arguments.raw_dir / "Software.csv.gz"
    metadata_source = arguments.raw_dir / "meta_Software.jsonl.gz"
    for source in (interaction_source, metadata_source):
        if not source.is_file():
            parser.error(f"Missing archive: {source}; run python -m scoperec.download first")
    interactions = arguments.output_dir / "interactions.parquet"
    items = arguments.output_dir / "items.parquet"
    print("Converting interactions...", flush=True)
    interaction_count = convert_interactions(interaction_source, interactions)
    print(f"Preserved {interaction_count:,} interactions; converting metadata...", flush=True)
    item_count = convert_metadata(metadata_source, items)
    print(f"Preserved {item_count:,} metadata records; profiling...", flush=True)
    report = {
        "dataset": "Amazon Reviews 2023",
        "category": "Software",
        "prepared_at": datetime.now(timezone.utc).isoformat(),
        "timestamp_unit": "milliseconds",
        "sources": [file_fingerprint(path) for path in (interaction_source, metadata_source)],
        "outputs": [file_fingerprint(path) for path in (interactions, items)],
        **profile_dataset(interactions, items, threads=arguments.threads),
    }
    arguments.report.parent.mkdir(parents=True, exist_ok=True)
    temporary_report = arguments.report.with_name(arguments.report.name + ".part")
    temporary_report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    temporary_report.replace(arguments.report)
    print(json.dumps({"report": str(arguments.report), **report["totals"]}, indent=2))


if __name__ == "__main__":
    main()