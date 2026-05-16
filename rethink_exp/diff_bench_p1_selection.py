"""
diff_bench_p1_selection.py
--------------------------
Compare Phase 1 HP selection between the (LEAKY) test-regret-based pick and
the correct val-regret-based pick.

Reads:
  - bench_p1_best.json       (metric=test, written by collect_bench_p1.py)
  - bench_p1_best_val.json   (metric=val,  written by collect_bench_p1.py --metric val)

For each (method, problem) present in both, reports:
  - old (lr, batch)  -> new (lr, batch)
  - test regret at old config vs test regret at new config
  - whether the selected config changed
  - source of the new val metric (val_eval vs val_pred_mse)

Usage:
    python rethink_exp/diff_bench_p1_selection.py
    python rethink_exp/diff_bench_p1_selection.py --markdown docs/tables/phase1_leakage_diff.md
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from collect_bench_p1 import (  # noqa: E402
    PROBLEMS, METHODS, USE_ABSOLUTE,
    results_path, load_test_regret,
)


def test_regret_at(prob, method, batch, lr):
    """Return (abs_regret, rel_regret) from results.npy for a specific config."""
    if batch is None or lr is None:
        return None, None
    return load_test_regret(results_path(prob, method, batch, lr))


def reported_metric(abs_r, rel_r, prob):
    """Mirrors collect_bench_p1: absolute for USE_ABSOLUTE, else relative."""
    if abs_r is None:
        return None
    if prob in USE_ABSOLUTE:
        return abs_r
    return rel_r


def fmt_cfg(entry):
    if entry is None:
        return "—"
    return f"{entry['batch']}/{entry['lr']}"


def fmt_num(v):
    if v is None:
        return "—"
    return f"{v:.4f}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_json", default="bench_p1_best.json")
    parser.add_argument("--val_json",  default="bench_p1_best_val.json")
    parser.add_argument("--markdown",  default=None,
                        help="If set, also write a markdown diff table to this path.")
    args = parser.parse_args()

    with open(args.test_json) as f:
        test_sel = json.load(f)
    with open(args.val_json) as f:
        val_sel = json.load(f)

    rows = []  # list of dicts for later printing
    n_changed = 0
    n_same = 0
    n_only_test = 0
    n_only_val = 0
    regret_deltas = []  # new_test_regret - old_test_regret (positive = worse)

    for method in METHODS:
        for prob in PROBLEMS:
            t_entry = test_sel.get(method, {}).get(prob)
            v_entry = val_sel.get(method, {}).get(prob)

            if t_entry is None and v_entry is None:
                continue
            if t_entry is not None and v_entry is None:
                n_only_test += 1
                rows.append({
                    "method": method, "prob": prob,
                    "old_cfg": fmt_cfg(t_entry), "new_cfg": "—",
                    "old_test_reg": None, "new_test_reg": None,
                    "changed": "N/A", "val_source": "—",
                    "note": "no val-based selection",
                })
                continue
            if t_entry is None and v_entry is not None:
                n_only_val += 1
                rows.append({
                    "method": method, "prob": prob,
                    "old_cfg": "—", "new_cfg": fmt_cfg(v_entry),
                    "old_test_reg": None, "new_test_reg": None,
                    "changed": "N/A", "val_source": v_entry.get("val_source", "—"),
                    "note": "no test-based selection",
                })
                continue

            same = (t_entry["batch"] == v_entry["batch"]
                    and t_entry["lr"] == v_entry["lr"])
            if same:
                n_same += 1
            else:
                n_changed += 1

            # Test regret at each selection
            old_abs, old_rel = test_regret_at(prob, method,
                                              t_entry["batch"], t_entry["lr"])
            new_abs, new_rel = test_regret_at(prob, method,
                                              v_entry["batch"], v_entry["lr"])
            old_report = reported_metric(old_abs, old_rel, prob)
            new_report = reported_metric(new_abs, new_rel, prob)

            delta = None
            if old_report is not None and new_report is not None:
                delta = new_report - old_report
                regret_deltas.append(delta)

            rows.append({
                "method": method, "prob": prob,
                "old_cfg": fmt_cfg(t_entry), "new_cfg": fmt_cfg(v_entry),
                "old_test_reg": old_report, "new_test_reg": new_report,
                "delta": delta,
                "changed": "" if same else "CHANGED",
                "val_source": v_entry.get("val_source", "val_eval"),
            })

    # ---- Print console summary ----
    print("=" * 110)
    print(f"Phase 1 HP-selection diff: {args.test_json}  (LEAKY, test-regret)")
    print(f"                     vs:  {args.val_json}    (val-regret)")
    print("=" * 110)
    print(f"{'method':>10} {'problem':>18} {'old (b/lr)':>14} {'new (b/lr)':>14} "
          f"{'old_test':>10} {'new_test':>10} {'Δ':>10}  {'src':>12}  status")
    print("-" * 110)
    for r in rows:
        delta = r.get("delta")
        print(f"{r['method']:>10} {r['prob']:>18} {r['old_cfg']:>14} {r['new_cfg']:>14} "
              f"{fmt_num(r['old_test_reg']):>10} {fmt_num(r['new_test_reg']):>10} "
              f"{fmt_num(delta):>10}  {r.get('val_source','—'):>12}  {r.get('changed','')}")

    print("-" * 110)
    total = n_same + n_changed
    print(f"\nSummary: {total} cells present in both selections")
    print(f"  same config:    {n_same}  ({100*n_same/total:.1f}%)" if total else "")
    print(f"  changed config: {n_changed}  ({100*n_changed/total:.1f}%)" if total else "")
    print(f"  only in test:   {n_only_test}")
    print(f"  only in val:    {n_only_val}")

    if regret_deltas:
        deltas = np.array(regret_deltas)
        changed_deltas = deltas[deltas != 0]
        print(f"\nTest-regret movement (new − old) on changed cells only "
              f"(n={len(changed_deltas)}):")
        if len(changed_deltas):
            print(f"  mean:   {changed_deltas.mean():+.4f}")
            print(f"  median: {np.median(changed_deltas):+.4f}")
            print(f"  max:    {changed_deltas.max():+.4f}")
            print(f"  min:    {changed_deltas.min():+.4f}")
            print(f"  # worse (new>old): {int((changed_deltas>0).sum())}")
            print(f"  # better (new<old): {int((changed_deltas<0).sum())}")

    # ---- Optional markdown output ----
    if args.markdown:
        os.makedirs(os.path.dirname(args.markdown) or ".", exist_ok=True)
        with open(args.markdown, "w") as f:
            f.write("# Phase 1 HP-selection diff: test-leakage fix\n\n")
            f.write(f"Old selection: `{args.test_json}` (picks by test regret — leaky)\n\n")
            f.write(f"New selection: `{args.val_json}` (picks by val regret — correct)\n\n")
            f.write("Reported test regret uses the same relative/absolute "
                    "convention as `collect_bench_p1.py` (abs for `portfolio`, rel otherwise).\n\n")
            f.write("| method | problem | old (batch/lr) | new (batch/lr) | old_test | new_test | Δ | val_src | status |\n")
            f.write("|---|---|---|---|---|---|---|---|---|\n")
            for r in rows:
                f.write(f"| {r['method']} | {r['prob']} | {r['old_cfg']} | {r['new_cfg']} "
                        f"| {fmt_num(r['old_test_reg'])} | {fmt_num(r['new_test_reg'])} "
                        f"| {fmt_num(r.get('delta'))} | {r.get('val_source','—')} "
                        f"| {r.get('changed','')} |\n")
            f.write(f"\n**Summary.** {n_same} unchanged, {n_changed} changed "
                    f"out of {total} shared cells.\n")
            if regret_deltas:
                deltas = np.array(regret_deltas)
                changed_deltas = deltas[deltas != 0]
                if len(changed_deltas):
                    f.write(f"Test-regret movement on changed cells (new − old): "
                            f"mean {changed_deltas.mean():+.4f}, "
                            f"median {np.median(changed_deltas):+.4f}, "
                            f"max {changed_deltas.max():+.4f}, "
                            f"min {changed_deltas.min():+.4f}. "
                            f"{int((changed_deltas>0).sum())} cells worsen, "
                            f"{int((changed_deltas<0).sum())} improve.\n")
        print(f"\nMarkdown diff written to {args.markdown}")


if __name__ == "__main__":
    main()
