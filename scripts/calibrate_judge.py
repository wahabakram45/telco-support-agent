"""Calibrate the LLM judge against your labels. Lesson 06.

Run: uv run python scripts/calibrate_judge.py --make-sheet baseline   # write the label sheet
     (fill in `human:` for every answer, without looking at the judge's verdicts)
     uv run python scripts/calibrate_judge.py baseline                # judge vs you
"""

import argparse

from telco_support_agent.generation_eval import GenerationReport
from telco_support_agent.judge_calibration import (
    LABELS_PATH,
    VERDICTS,
    calibrate,
    load_labels,
    sample_for_labelling,
    write_label_sheet,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run", help="label of a saved generation run, e.g. baseline")
    parser.add_argument("--make-sheet", action="store_true", help="write a new label sheet")
    parser.add_argument("--n", type=int, default=24)
    args = parser.parse_args()
    report = GenerationReport.load(args.run)

    if args.make_sheet:
        if LABELS_PATH.exists():
            raise SystemExit(f"{LABELS_PATH} exists; delete it first to start over")
        items = sample_for_labelling(report, args.n)
        write_label_sheet(items)
        print(f"{len(items)} answers → {LABELS_PATH}. Fill in `human:` for each.")
        return

    c = calibrate(load_labels(), report)
    print(f"{c.n} labelled answers   agreement {c.agreement:.0%}   Cohen's kappa {c.kappa:.2f}\n")
    print("rows = you, columns = judge")
    print(f"{'':<12}" + "".join(f"{v:>11}" for v in VERDICTS))
    for h in VERDICTS:
        print(f"{h:<12}" + "".join(f"{c.confusion[h][j]:>11}" for j in VERDICTS))
    if c.stale:
        print(f"\nskipped (answer changed since labelling): {c.stale}")
    print(f"\n{len(c.disagreements)} disagreements:")
    for d in c.disagreements:
        print(f"\n  [{d['id']}] {d['question']}")
        print(f"     you: {d['human']}   judge: {d['judge']}   {d['note']}")
        print(f"     judge's reasoning: {d['judge_reasoning']}")


if __name__ == "__main__":
    main()
