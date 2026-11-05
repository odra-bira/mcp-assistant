"""
Compare benchmark runs side by side.

    python bench/compare.py                       # every run in results/
    python bench/compare.py results/a.json results/b.json

Prints one table (and saves it to results/comparison.md). Each row is a run:
which brain/model, which label, which version of the instructions and test cases,
and the scores. Only compare rows with the same "cases" version: they ran the same tests.
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "bench"))

from benchmark import metrics, pct  # noqa: E402

CATEGORIES = ["tool choice", "multi-step", "restraint"]


def load(path: Path) -> tuple[dict, list[dict]]:
    data = json.loads(path.read_text())
    if isinstance(data, list):  # results saved before run labels existed
        return {"brain": path.stem, "label": "", "date": "", "prompt_version": "?",
                "cases_version": "?", "only": ""}, data
    return data["meta"], data["runs"]


def main() -> None:
    files = [Path(p) for p in sys.argv[1:]] or sorted((ROOT / "results").glob("*.json"))
    if not files:
        sys.exit("No results yet. Run: python bench/benchmark.py --brain demo")

    header = ["Run", "Date", "Prompt", "Cases", "Runs", "Pass rate", *CATEGORIES,
              "Hijacked (scanner off)", "Hijacked (scanner on)", "Unwanted writes", "Avg s"]
    rows = []
    for f in files:
        meta, runs = load(f)
        if not runs:
            continue
        m = metrics(runs)
        name = meta["brain"] + (f" · {meta['label']}" if meta.get("label") else "")
        if meta.get("only"):
            name += " (subset)"
        score = pct(m["pass_rate"]) + (f" ±{100 * m['pass_rate_std']:.0f}" if m["repeats"] > 1 else "")
        rows.append([name, meta.get("date", ""), meta.get("prompt_version", "?"), meta.get("cases_version", "?"),
                     f"{m['cases']}×{m['repeats']}", score,
                     *[pct(m["by_category"].get(c, float("nan"))) for c in CATEGORIES],
                     pct(m["hijack_scanner_off"]), pct(m["hijack_scanner_on"]),
                     str(m["unwanted_writes"]), f"{m['avg_seconds']:.1f}"])

    table = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    table += ["| " + " | ".join(r) + " |" for r in rows]
    versions = {r[3] for r in rows}
    note = ("" if len(versions) <= 1 else
            "\nNote: these runs used different test-case versions (Cases column); "
            "only compare rows with the same value.")
    text = "## Benchmark comparison\n\n" + "\n".join(table) + "\n" + note
    print(text)
    out = ROOT / "results" / "comparison.md"
    out.parent.mkdir(exist_ok=True)
    out.write_text(text + "\n")
    print(f"\nSaved: {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
