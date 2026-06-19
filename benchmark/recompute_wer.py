#!/usr/bin/env python3
"""Recompute WER for existing benchmark result files using new normalization.

Reads ``benchmark/results_full/*.json`` files, recomputes per-sample and
aggregate WER with ``common.compute_wer`` and ``common.bootstrap_ci``, writes
``*_renorm.json`` files, and emits a Markdown before/after summary.
"""

import argparse
import json
import sys
from pathlib import Path

# common.py lives in the same directory as this script.
_SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPT_DIR))

import common  # noqa: E402


def _replace_suffix(path: Path) -> Path:
    return path.with_name(f"{path.stem}_renorm.json")


def recompute_file(input_path: Path, output_path: Path) -> list[dict]:
    """Recompute WER for a single result file and return summary rows."""
    with open(input_path, encoding="utf-8") as f:
        data = json.load(f)

    dataset = data.get("dataset", input_path.stem)
    rows: list[dict] = []

    for runner in data.get("runners", []):
        old_wer = runner.get("wer")
        old_ci_low = runner.get("ci_low")
        old_ci_high = runner.get("ci_high")

        total_errors = 0
        total_ref_words = 0
        per_sample: list[tuple[int, int]] = []
        total_naive_errors = 0
        total_naive_ref_words = 0
        naive_per_sample: list[tuple[int, int]] = []

        for detail in runner.get("details", []):
            ref = detail.get("reference", "")
            hyp = detail.get("hypothesis", "")

            wer, errors, ref_count = common.compute_wer(ref, hyp)
            naive_wer, naive_errors, naive_ref_count = common.compute_wer_naive(ref, hyp)

            detail["wer"] = wer
            detail["errors"] = errors
            detail["ref_words"] = ref_count
            detail["naive_wer"] = naive_wer
            detail["naive_errors"] = naive_errors
            detail["naive_ref_words"] = naive_ref_count

            total_errors += errors
            total_ref_words += ref_count
            per_sample.append((ref_count, errors))
            total_naive_errors += naive_errors
            total_naive_ref_words += naive_ref_count
            naive_per_sample.append((naive_ref_count, naive_errors))

        overall_wer = (
            (total_errors / total_ref_words * 100.0)
            if total_ref_words > 0
            else 0.0
        )
        overall_naive_wer = (
            (total_naive_errors / total_naive_ref_words * 100.0)
            if total_naive_ref_words > 0
            else 0.0
        )
        ci_low, ci_high = common.bootstrap_ci(per_sample, iterations=1000)
        naive_ci_low, naive_ci_high = common.bootstrap_ci(naive_per_sample, iterations=1000)

        runner["wer"] = overall_wer
        runner["ci_low"] = ci_low
        runner["ci_high"] = ci_high
        runner["total_errors"] = total_errors
        runner["total_ref_words"] = total_ref_words
        runner["naive_wer"] = overall_naive_wer
        runner["naive_ci_low"] = naive_ci_low
        runner["naive_ci_high"] = naive_ci_high
        runner["naive_total_errors"] = total_naive_errors
        runner["naive_total_ref_words"] = total_naive_ref_words
        runner["naive_delta"] = overall_wer - overall_naive_wer

        rows.append(
            {
                "dataset": dataset,
                "engine": runner.get("name", "unknown"),
                "old_wer": old_wer,
                "old_ci_low": old_ci_low,
                "old_ci_high": old_ci_high,
                "new_wer": overall_wer,
                "new_ci_low": ci_low,
                "new_ci_high": ci_high,
            }
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")

    return rows


def _fmt(value) -> str:
    if value is None:
        return "—"
    return f"{value:.2f}"


def _fmt_ci(low, high) -> str:
    if low is None or high is None:
        return "—"
    return f"{_fmt(low)}–{_fmt(high)}"


def build_summary_table(rows: list[dict]) -> str:
    sorted_rows = sorted(rows, key=lambda r: (r["dataset"], r["engine"]))

    lines = [
        "# WER Re-normalization Summary",
        "",
        "| Dataset | Engine | Old WER | Old CI | New WER | New CI | Δ WER |",
        "|---|---|---|---|---|---|---|",
    ]

    current_dataset = None
    for row in sorted_rows:
        delta = (
            None
            if row["old_wer"] is None or row["new_wer"] is None
            else row["new_wer"] - row["old_wer"]
        )

        # Show dataset only on first row of each group.
        dataset_cell = row["dataset"] if row["dataset"] != current_dataset else ""
        current_dataset = row["dataset"]

        delta_str = "—" if delta is None else f"{delta:+.2f}"

        lines.append(
            f"| {dataset_cell} | {row['engine']} | "
            f"{_fmt(row['old_wer'])} | {_fmt_ci(row['old_ci_low'], row['old_ci_high'])} | "
            f"{_fmt(row['new_wer'])} | {_fmt_ci(row['new_ci_low'], row['new_ci_high'])} | "
            f"{delta_str} |"
        )

    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Recompute WER for existing benchmark result files."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=_SCRIPT_DIR / "results_full",
        help="Directory containing original *.json result files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_SCRIPT_DIR / "results_full",
        help="Directory where *_renorm.json files and summary are written",
    )
    args = parser.parse_args()

    input_files = sorted(args.input_dir.glob("*.json"))
    all_rows: list[dict] = []

    for input_path in input_files:
        if input_path.name.endswith("_renorm.json"):
            continue

        output_path = args.output_dir / _replace_suffix(input_path)
        rows = recompute_file(input_path, output_path)
        all_rows.extend(rows)
        print(f"Recomputed {input_path.name} -> {output_path.name}", file=sys.stderr)

    summary_md = build_summary_table(all_rows)

    summary_path = args.output_dir / "renorm_summary.md"
    summary_path.write_text(summary_md, encoding="utf-8")

    print(summary_md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
