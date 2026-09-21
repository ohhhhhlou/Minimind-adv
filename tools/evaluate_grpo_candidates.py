"""Evaluate generated GRPO candidates with the fixed AdvertiseGen metrics."""
import argparse
import json
from pathlib import Path

from evaluation.metrics import VERSION, aggregate, score
from tools.generate_grpo_candidates import digest


def evaluate_group(group):
    attributes = [tuple(pair) for pair in group["attributes"]]
    results = []
    seen = set()
    for candidate in group["candidates"]:
        candidate_id = candidate["candidate_id"]
        if candidate_id in seen:
            raise ValueError(f"Duplicate candidate ID: {candidate_id}")
        seen.add(candidate_id)
        content_ids = candidate.get("content_token_ids")
        if content_ids is not None and not isinstance(content_ids, list):
            raise ValueError(f"content_token_ids must be a list: {candidate_id}")
        metrics = score(
            candidate["text"],
            attributes,
            token_ids=content_ids,
            eos=candidate.get("eos_normal"),
        )
        results.append(
            dict(
                candidate_id=candidate_id,
                seed=candidate.get("seed"),
                text=candidate["text"],
                metrics=metrics,
            )
        )
    if len(results) < 2:
        raise ValueError(f"At least two candidates are required: {group['group_id']}")
    return dict(
        group_id=group["group_id"],
        line_number=group.get("line_number"),
        metadata=group.get("metadata", {}),
        attributes=group["attributes"],
        candidates=results,
    )


def evaluate_file(candidates_path, output_dir):
    candidates_path = Path(candidates_path).resolve()
    output_dir = Path(output_dir).resolve()
    if output_dir.drive.lower() == "c:":
        raise ValueError("Writing evaluation artifacts to C: is forbidden")
    if not candidates_path.is_file():
        raise FileNotFoundError(candidates_path)
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite: {output_dir}")

    groups = []
    with candidates_path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            try:
                groups.append(evaluate_group(json.loads(line)))
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                raise ValueError(f"{candidates_path}:{line_number}: {exc}") from exc
    if not groups:
        raise ValueError("Empty candidate file")

    all_scores = [candidate["metrics"] for group in groups for candidate in group["candidates"]]
    group_summaries = []
    for group in groups:
        group_summaries.append(
            dict(
                group_id=group["group_id"],
                candidates=aggregate([candidate["metrics"] for candidate in group["candidates"]]),
            )
        )

    output_dir.mkdir(parents=True)
    with (output_dir / "candidate_metrics.jsonl").open("w", encoding="utf-8") as destination:
        for group in groups:
            destination.write(json.dumps(group, ensure_ascii=False) + "\n")
    (output_dir / "summary.json").write_text(
        json.dumps(
            dict(
                status="complete",
                protocol=VERSION,
                groups=len(groups),
                candidates=len(all_scores),
                overall=aggregate(all_scores),
                by_group=group_summaries,
            ),
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    (output_dir / "manifest.json").write_text(
        json.dumps(
            dict(
                protocol=VERSION,
                candidates_sha256=digest(candidates_path),
                script_sha256=digest(Path(__file__)),
                metrics_sha256=digest(
                    Path(__file__).resolve().parents[1] / "evaluation" / "metrics.py"
                ),
                score_policy="fixed literal metrics; no external reward or fitted weights",
            ),
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return output_dir


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(evaluate_file(args.candidates, args.output_dir))


if __name__ == "__main__":
    main()
