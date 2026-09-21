"""Build cleaned and coverage-enhancement AdvertiseGen SFT datasets.

The source file is never overwritten.  The enhancement set is a subset of the
cleaned training set: it keeps the original conversations unchanged and uses
literal input/output alignment only as a conservative quality gate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict, deque
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.metrics import is_loop, parse_attributes, repetition_ratio


DEFAULT_SOURCE = ROOT / "dataset/advertisegen/advertisegen_sft_train.jsonl"
PHRASE = "穿着舒适"
KEY_ATTRIBUTES = {"类型", "颜色"}


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def get_texts(row: dict) -> tuple[str, str]:
    messages = row.get("conversations")
    if not isinstance(messages, list):
        raise ValueError("missing conversations list")
    users = [item.get("content", "") for item in messages if item.get("role") == "user"]
    assistants = [item.get("content", "") for item in messages if item.get("role") == "assistant"]
    if len(users) != 1 or len(assistants) != 1:
        raise ValueError("expected exactly one user and one assistant message")
    return users[0], assistants[0]


def template_signature(text: str, attributes: list[tuple[str, str]]) -> str:
    """Normalize for diversity analysis only; this text is never used for training."""
    result = text
    replacements = sorted(attributes, key=lambda item: len(item[1]), reverse=True)
    for key, value in replacements:
        if value:
            result = result.replace(value, "{" + key + "}")
    result = re.sub(r"\d+(?:\.\d+)?", "{数值}", result)
    result = re.sub(r"\s+", "", result)
    return result


def stable_key(row: dict) -> str:
    payload = json.dumps(row, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def analyse(row: dict) -> dict:
    prompt, answer = get_texts(row)
    attributes = parse_attributes(prompt)
    hits = [value in answer for _, value in attributes]
    key_positions = [i for i, (key, _) in enumerate(attributes) if key in KEY_ATTRIBUTES]
    key_covered = all(hits[i] for i in key_positions)
    type_value = next((value for key, value in attributes if key == "类型"), "<无类型>")
    structure = (type_value, tuple(sorted({key for key, _ in attributes})))
    normalized = re.sub(r"\s+", "", answer)
    return {
        "answer": answer,
        "attributes": attributes,
        "coverage": sum(hits) / len(hits),
        "all_covered": all(hits),
        "key_covered": key_covered,
        "phrase_count": answer.count(PHRASE),
        "char4_repetition": repetition_ratio(list(normalized), 4),
        "loop": is_loop(answer),
        "structure": structure,
        "template": template_signature(answer, attributes),
    }


def choose_diverse(candidates: list[tuple[dict, dict]], target: int) -> list[dict]:
    """Round-robin across input structures and templates, without sample weights."""
    structures: dict[tuple, dict[str, list[tuple[dict, dict]]]] = defaultdict(lambda: defaultdict(list))
    for row, info in candidates:
        structures[info["structure"]][info["template"]].append((row, info))

    structure_queues: dict[tuple, deque[tuple[dict, dict]]] = {}
    for structure, templates in structures.items():
        ordered_templates = sorted(templates.items(), key=lambda item: (len(item[1]), item[0]))
        queue: deque[tuple[dict, dict]] = deque()
        pools = [deque(sorted(items, key=lambda pair: stable_key(pair[0]))) for _, items in ordered_templates]
        while pools:
            remaining = []
            for pool in pools:
                queue.append(pool.popleft())
                if pool:
                    remaining.append(pool)
            pools = remaining
        structure_queues[structure] = queue

    selected: list[dict] = []
    active = deque(sorted(structure_queues, key=lambda key: (len(structure_queues[key]), repr(key))))
    while active and len(selected) < target:
        structure = active.popleft()
        selected.append(structure_queues[structure].popleft()[0])
        if structure_queues[structure]:
            active.append(structure)
    return selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--clean-output", type=Path,
                        default=DEFAULT_SOURCE.with_name("advertisegen_sft_train_no_phrase_repeat.jsonl"))
    parser.add_argument("--enhancement-output", type=Path,
                        default=DEFAULT_SOURCE.with_name("advertisegen_sft_coverage_enhancement_5000.jsonl"))
    parser.add_argument("--report", type=Path,
                        default=DEFAULT_SOURCE.with_name("advertisegen_sft_coverage_enhancement_report.json"))
    parser.add_argument("--target", type=int, default=5000)
    parser.add_argument("--max-char4-repetition", type=float, default=0.10)
    args = parser.parse_args()

    source_rows = read_jsonl(args.source.resolve())
    analysed: list[tuple[dict, dict]] = []
    invalid = []
    for index, row in enumerate(source_rows, 1):
        try:
            analysed.append((row, analyse(row)))
        except (KeyError, TypeError, ValueError) as exc:
            invalid.append({"line": index, "error": str(exc)})

    removed_phrase = [(row, info) for row, info in analysed if info["phrase_count"] >= 2]
    cleaned = [(row, info) for row, info in analysed if info["phrase_count"] < 2]
    eligible = [
        (row, info) for row, info in cleaned
        if info["all_covered"]
        and info["key_covered"]
        and not info["loop"]
        and info["char4_repetition"] <= args.max_char4_repetition
    ]
    selected = choose_diverse(eligible, min(args.target, len(eligible)))

    write_jsonl(args.clean_output.resolve(), [row for row, _ in cleaned])
    write_jsonl(args.enhancement_output.resolve(), selected)

    selected_info = [analyse(row) for row in selected]
    report = {
        "source": str(args.source.resolve()),
        "policy": {
            "clean": f"remove rows whose assistant answer contains {PHRASE!r} at least twice",
            "enhancement_quality_gates": {
                "all_input_attribute_values_literal_covered": True,
                "key_attributes_literal_covered": sorted(KEY_ATTRIBUTES),
                "loop_detected": False,
                "max_char4_repetition": args.max_char4_repetition,
            },
            "enhancement_selection": (
                "round-robin across (type value, attribute-key set), preferring different "
                "normalized output templates; original conversations are unchanged"
            ),
        },
        "counts": {
            "source": len(source_rows),
            "valid": len(analysed),
            "invalid": len(invalid),
            "removed_phrase_repeated_at_least_twice": len(removed_phrase),
            "clean": len(cleaned),
            "eligible_for_enhancement": len(eligible),
            "enhancement": len(selected),
        },
        "enhancement": {
            "input_structures": len({info["structure"] for info in selected_info}),
            "normalized_templates": len({info["template"] for info in selected_info}),
            "type_counts": dict(Counter(info["structure"][0] for info in selected_info).most_common()),
        },
        "invalid_examples": invalid[:20],
        "outputs": {
            "clean": str(args.clean_output.resolve()),
            "enhancement": str(args.enhancement_output.resolve()),
        },
    }
    args.report.resolve().write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
