"""Build reproducible, leakage-checked AdvertiseGen SFT files.

The source files are treated as immutable. Exact conversation duplicates are
removed from training, and validation examples duplicated in the test set are
removed from validation so the final test set remains stable.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any


SPLITS = ("train", "val", "test")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            validate_row(row, path, line_number)
            rows.append(row)
    return rows


def validate_row(row: dict[str, Any], path: Path, line_number: int) -> None:
    conversations = row.get("conversations")
    if not isinstance(conversations, list) or len(conversations) != 3:
        raise ValueError(f"{path}:{line_number}: expected exactly three messages")
    roles = [message.get("role") for message in conversations]
    if roles != ["system", "user", "assistant"]:
        raise ValueError(f"{path}:{line_number}: invalid roles: {roles}")
    for message in conversations:
        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            raise ValueError(f"{path}:{line_number}: empty message content")
    metadata = row.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError(f"{path}:{line_number}: missing metadata object")


def conversation_key(row: dict[str, Any]) -> str:
    return "\0".join(
        message["content"].strip() for message in row["conversations"]
    )


def deduplicate(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    seen: set[str] = set()
    output: list[dict[str, Any]] = []
    for row in rows:
        key = conversation_key(row)
        if key in seen:
            continue
        seen.add(key)
        output.append(row)
    return output, len(rows) - len(output)


def style_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts = Counter(row["metadata"].get("input_style", "unknown") for row in rows)
    return dict(sorted(counts.items()))


def stratified_sample(
    rows: list[dict[str, Any]], size: int, seed: int
) -> list[dict[str, Any]]:
    if size >= len(rows):
        return list(rows)
    rng = random.Random(seed)
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        style = row["metadata"].get("input_style", "unknown")
        groups.setdefault(style, []).append(row)

    sampled: list[dict[str, Any]] = []
    remaining = size
    ordered_styles = sorted(groups)
    for index, style in enumerate(ordered_styles):
        group = groups[style]
        if index == len(ordered_styles) - 1:
            take = remaining
        else:
            take = round(size * len(group) / len(rows))
            take = min(take, remaining)
        sampled.extend(rng.sample(group, take))
        remaining -= take
    rng.shuffle(sampled)
    return sampled


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
            handle.write("\n")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--smoke-size", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    source = {
        split: read_jsonl(args.source_dir / f"advertisegen_sft_{split}.jsonl")
        for split in SPLITS
    }

    train, removed_train_duplicates = deduplicate(source["train"])
    test, removed_test_duplicates = deduplicate(source["test"])
    test_keys = {conversation_key(row) for row in test}

    val_unique, removed_val_duplicates = deduplicate(source["val"])
    val = [row for row in val_unique if conversation_key(row) not in test_keys]
    removed_val_test_overlap = len(val_unique) - len(val)

    train_keys = {conversation_key(row) for row in train}
    val_keys = {conversation_key(row) for row in val}
    assert not (train_keys & val_keys)
    assert not (train_keys & test_keys)
    assert not (val_keys & test_keys)

    smoke = stratified_sample(train, args.smoke_size, args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "train": (args.output_dir / "advertisegen_sft_train.jsonl", train),
        "train_smoke_5000": (
            args.output_dir / "advertisegen_sft_train_smoke_5000.jsonl",
            smoke,
        ),
        "val": (args.output_dir / "advertisegen_sft_val.jsonl", val),
        "test": (args.output_dir / "advertisegen_sft_test.jsonl", test),
    }
    for path, rows in outputs.values():
        write_jsonl(path, rows)

    report = {
        "source_dir": str(args.source_dir.resolve()),
        "policy": {
            "train": "remove exact conversation duplicates, keep first occurrence",
            "validation": "remove exact duplicates, then remove conversations found in test",
            "test": "remove exact duplicates only; test takes precedence over validation",
            "smoke": "deterministic input_style-stratified sample from cleaned train",
        },
        "seed": args.seed,
        "source_counts": {split: len(source[split]) for split in SPLITS},
        "clean_counts": {
            "train": len(train),
            "train_smoke_5000": len(smoke),
            "val": len(val),
            "test": len(test),
        },
        "removed": {
            "train_exact_duplicates": removed_train_duplicates,
            "val_exact_duplicates": removed_val_duplicates,
            "test_exact_duplicates": removed_test_duplicates,
            "val_rows_overlapping_test": removed_val_test_overlap,
        },
        "input_style_counts": {
            name: style_counts(rows) for name, (_, rows) in outputs.items()
        },
        "overlap_after_cleaning": {
            "train_val": len(train_keys & val_keys),
            "train_test": len(train_keys & test_keys),
            "val_test": len(val_keys & test_keys),
        },
        "files": {
            name: {"path": path.name, "sha256": sha256(path)}
            for name, (path, _) in outputs.items()
        },
    }
    report_path = args.output_dir / "advertisegen_sft_clean_report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
