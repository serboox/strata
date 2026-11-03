import argparse
import json
import sys
from pathlib import Path

BUDGET_ORDER = ["8G", "12G", "16G", "22G", "none"]


def load_results(paths: list) -> list:
    """The result files quant_bench.py wrote; a file that is not one is reported on stderr and skipped."""
    results = []
    for path in paths:
        try:
            result = json.loads(Path(path).read_text(encoding="utf-8"))
            result["file"] = Path(path).name
            results.append(result)
        except (OSError, ValueError) as error:
            print(f"bench-table: {path}: {error}", file=sys.stderr)
    return results


def sort_key(result: dict) -> tuple:
    budget = result.get("budget", "")
    return (BUDGET_ORDER.index(budget) if budget in BUDGET_ORDER else len(BUDGET_ORDER), result.get("label", ""))


def row(result: dict) -> list:
    summary = next(iter(result.get("summary", {}).values()), {})
    median = summary.get("median")
    speeds = summary.get("tok_s") or []
    share = summary.get("accepted_share")
    notes = []
    if result.get("aborted"):
        notes.append("aborted")
    if summary.get("left_out"):
        notes.append(f"{summary['left_out']} left out")
    if result.get("cleanup_problems"):
        notes.append("cleanup problem")
    return [result.get("label", "?"), result.get("budget", "?"),
            f"{median:.1f}" if median is not None else "-",
            f"{min(speeds):.1f}-{max(speeds):.1f}" if speeds else "-",
            f"{share * 100:.0f} %" if share is not None else "-",
            f"{summary['peak_c']:.0f}" if summary.get("peak_c") is not None else "-",
            result.get("turbo", "?"), str(summary.get("step_ms", {}).get("total", "-")), str(summary.get("tokens_per_step", "-")),
            ", ".join(notes) or "ok"]


def table(results: list) -> str:
    header = ["Run", "Budget", "tok/s median", "tok/s range", "Drafts accepted", "Peak C", "Turbo", "ms/step", "tok/step", "Notes"]
    lines = [header] + [row(result) for result in sorted(results, key=sort_key)]
    widths = [max(len(line[i]) for line in lines) for i in range(len(header))]
    rendered = ["| " + " | ".join(cell.ljust(widths[i]) for i, cell in enumerate(line)) + " |" for line in lines]
    rendered.insert(1, "| " + " | ".join("-" * width for width in widths) + " |")
    return "\n".join(rendered)


def main() -> int:
    parser = argparse.ArgumentParser(description="Print a table of quant_bench.py result files.")
    parser.add_argument("files", nargs="+", type=Path)
    args = parser.parse_args()
    results = load_results(args.files)
    if not results:
        return 1
    print(table(results))
    return 0


if __name__ == "__main__":
    sys.exit(main())
