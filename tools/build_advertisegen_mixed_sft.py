"""Build a deterministic replay mix for coverage-enhancement SFT.

The enhancement rows are kept once.  Replay rows are sampled from the cleaned
training set after excluding exact enhancement rows, then the combined rows are
shuffled deterministically.  Validation and test data are never read.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "dataset" / "advertisegen"


def read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
            if not isinstance(row.get("conversations"), list):
                raise ValueError(f"{path}:{line_number}: missing conversations")
            rows.append(row)
    if not rows:
        raise ValueError(f"{path}: empty dataset")
    return rows


def canonical(row: dict) -> str:
    return json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def input_style(row: dict) -> str:
    return str(row.get("metadata", {}).get("input_style", "unknown"))


def stratified_sample(rows: list[dict], target: int, rng: random.Random) -> list[dict]:
    """Sample proportionally by input_style, then fill any rounding remainder."""
    if target > len(rows):
        raise ValueError(f"replay target {target} exceeds {len(rows)} available rows")
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[input_style(row)].append(row)
    for group in groups.values():
        rng.shuffle(group)

    quotas = {style: target * len(group) // len(rows) for style, group in groups.items()}
    selected = [row for style, group in groups.items() for row in group[:quotas[style]]]
    leftovers = [row for style, group in groups.items() for row in group[quotas[style]:]]
    rng.shuffle(leftovers)
    selected.extend(leftovers[: target - len(selected)])
    return selected


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--clean-source", type=Path,
        default=DATA_DIR / "advertisegen_sft_train_no_phrase_repeat.jsonl",
    )
    parser.add_argument(
        "--enhancement-source", type=Path,
        default=DATA_DIR / "advertisegen_sft_coverage_enhancement_5000.jsonl",
    )
    parser.add_argument(
        "--output", type=Path,
        default=DATA_DIR / "advertisegen_sft_mixed_coverage_20000.jsonl",
    )
    parser.add_argument(
        "--report", type=Path,
        default=DATA_DIR / "advertisegen_sft_mixed_coverage_20000_report.json",
    )
    parser.add_argument("--replay-count", type=int, default=15000)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    if args.replay_count < 1:
        parser.error("--replay-count must be positive")
    clean_path = args.clean_source.resolve()
    enhancement_path = args.enhancement_source.resolve()
    output_path = args.output.resolve()
    report_path = args.report.resolve()
    if output_path in {clean_path, enhancement_path}:
        parser.error("output must not overwrite either source")

    clean_rows = read_jsonl(clean_path)
    enhancement_rows = read_jsonl(enhancement_path)
    enhancement_keys = {canonical(row) for row in enhancement_rows}
    if len(enhancement_keys) != len(enhancement_rows):
        raise ValueError("enhancement dataset contains exact duplicate rows")

    replay_candidates = [row for row in clean_rows if canonical(row) not in enhancement_keys]
    rng = random.Random(args.seed)
    replay_rows = stratified_sample(replay_candidates, args.replay_count, rng)
    mixed_rows = enhancement_rows + replay_rows
    rng.shuffle(mixed_rows)
    write_jsonl(output_path, mixed_rows)

    style_counts = lambda rows: dict(sorted(Counter(input_style(row) for row in rows).items()))
    report = {
        "policy": {
            "enhancement_rows_kept_once": True,
            "exact_enhancement_rows_excluded_from_replay": True,
            "replay_sampling": "proportional by metadata.input_style",
            "final_order": "deterministic shuffle",
            "seed": args.seed,
        },
        "counts": {
            "clean_source": len(clean_rows),
            "enhancement": len(enhancement_rows),
            "replay_candidates": len(replay_candidates),
            "replay_selected": len(replay_rows),
            "mixed": len(mixed_rows),
        },
        "input_style_counts": {
            "enhancement": style_counts(enhancement_rows),
            "replay": style_counts(replay_rows),
            "mixed": style_counts(mixed_rows),
        },
        "sources": {
            "clean": str(clean_path),
            "clean_sha256": sha256(clean_path),
            "enhancement": str(enhancement_path),
            "enhancement_sha256": sha256(enhancement_path),
        },
        "output": str(output_path),
        "output_sha256": sha256(output_path),
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
