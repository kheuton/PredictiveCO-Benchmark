"""
collect_bench_p2_val.py
-----------------------
Pick the val-best Phase-2 hyperparameter per (method, problem) and write
``bench_p2_best_val.json``.

For every (method, problem) in ``bench_p1_best_val.json``:
  * Phase-2-tunable methods (dfl, blackbox, qptl, listLTR, lodl, perturb, pg,
    dad): enumerate every prefix that the Phase-2 sweep produced, read its
    ``val_logs.csv``, take the min of the ``eval`` column, pick the HP value
    whose run had the lowest min val regret. ``perturb`` is treated as a
    single method with two sub-sweeps (sigma at n=10, n_samples at sigma=1.0)
    — all 10 candidates compete jointly.
  * Other methods (mse, identity, spo, nce, pointLTR, pairLTR, cpLayer):
    no method-specific HP sweep, so the winner is the Phase-1 best entry.

For each winner we also record:
  * ``test_regret`` — read from ``results.npy`` so downstream scripts can
    quote the test number that corresponds to the val-selected config.
  * the run directory ``run_dir`` and ``prefix`` so collect_loss_matrix.py
    can skip its own HP search.

Usage::

    python rethink_exp/collect_bench_p2_val.py
    python rethink_exp/collect_bench_p2_val.py --problem knapsack
    python rethink_exp/collect_bench_p2_val.py --method perturb
"""

import argparse
import csv
import json
import os
import re
import sys

import numpy as np

# Regex for "val MSE (no solver)" lines emitted by --skip_solver_eval runs.
_VAL_MSE_LINE = re.compile(
    r"Iter\s+(\d+),\s*val MSE \(no solver\):\s*([0-9eE+\-\.]+)"
)

# ---- Configuration (mirrors collect_bench_p2.py / collect_loss_matrix.py) ----

PROBLEMS = ["knapsack", "knapsack-real", "energy", "budgetalloc",
            "cubic", "bipartitematching", "portfolio", "asurv", "cook_county",
            "speed_humps", "sp_synth", "sp_planted", "pg_misspec", "shortestpath"]

ALL_METHODS = ["mse", "mse_train", "mse_val", "dfl", "identity", "spo", "nce", "blackbox",
               "pointLTR", "pairLTR", "listLTR", "lodl", "perturb", "pg",
               "qptl", "cpLayer", "dad"]

PROB_ARG = {
    "knapsack": "knapsack", "knapsack-real": "knapsack",
    "energy": "energy", "budgetalloc": "budgetalloc",
    "cubic": "cubic", "bipartitematching": "bipartitematching",
    "portfolio": "portfolio", "asurv": "asurv", "cook_county": "cook_county",
    "speed_humps": "speed_humps", "sp_synth": "sp_synth",
    "sp_planted": "sp_planted", "pg_misspec": "pg_misspec",
    "shortestpath": "shortestpath",
}

PROB_VERSION = {
    "knapsack": "gen", "knapsack-real": "energy", "energy": "energy",
    "budgetalloc": "real", "cubic": "gen", "bipartitematching": "cora",
    "portfolio": "real", "asurv": "real", "cook_county": "real",
    "speed_humps": "real", "sp_synth": "synth", "sp_planted": "planted",
    "pg_misspec": "v3", "shortestpath": "warcraft",
}

METHOD_PROBLEMS = {
    "qptl":    {"knapsack", "bipartitematching", "portfolio"},
    "cpLayer": {"knapsack", "bipartitematching", "portfolio"},
    "pg":      {"knapsack", "knapsack-real", "energy", "cubic", "bipartitematching",
                "portfolio", "asurv", "cook_county", "speed_humps",
                "sp_synth", "sp_planted", "pg_misspec"},
}

# Phase-2 sweeps: per method, a list of sub-sweeps. Each sub-sweep enumerates
# values for one HP; the prefix tag includes the *other* perturb HP at its
# default. The default HP value for each sweep (marked ``*`` in
# submit_bench_p2.sh) is what defines the "fixed" cell of the tuning-benefit
# table downstream; record it too.
HP_SWEEPS = {
    "dfl":      [dict(hp="dflalpha",    default="0.1",
                      vals=["0.001", "0.01", "0.1", "1.0", "10.0"],
                      tag_fn=lambda v: f"alpha{v}")],
    "blackbox": [dict(hp="lambd",       default="0.1",
                      vals=["0.01", "0.05", "0.1", "0.5", "1.0"],
                      tag_fn=lambda v: f"lam{v}")],
    "qptl":     [dict(hp="tau",         default="1.0",
                      vals=["0.1", "0.5", "1.0", "5.0", "10.0"],
                      tag_fn=lambda v: f"tau{v}")],
    "listLTR":  [dict(hp="tau",         default="1",
                      vals=["0.1", "0.5", "1", "5", "10"],
                      tag_fn=lambda v: f"tau{v}")],
    "lodl":     [dict(hp="num_samples", default="500",
                      vals=["100", "250", "500", "1000", "2000"],
                      tag_fn=lambda v: f"ns{v}")],
    "pg":       [dict(hp="sigma",       default="0.1",
                      vals=["0.01", "0.05", "0.1", "0.5", "1.0"],
                      tag_fn=lambda v: f"s{v.replace('.', 'p')}")],
    "perturb":  [dict(hp="sigma",       default="1.0",
                      vals=["0.1", "0.5", "1.0", "2.0", "5.0"],
                      tag_fn=lambda v: f"s{v.replace('.', 'p')}_n10"),
                 dict(hp="n_samples",   default="10",
                      vals=["5", "10", "25", "50", "100"],
                      tag_fn=lambda v: f"s1p0_n{v}")],
    "dad":      [dict(hp="stein_weight", default="1.0",
                      vals=["0.1", "0.5", "1.0", "2.0", "5.0"],
                      tag_fn=lambda v: f"sw{v.replace('.', 'p')}")],
}

USE_ABSOLUTE = {"portfolio"}
RESULTS_ROOT = "saved_records"
P1_BEST_PATH = "bench_p1_best_val.json"
OUT_PATH = "bench_p2_best_val.json"


def run_dir(prob, method, prefix):
    return os.path.join(RESULTS_ROOT, f"{PROB_ARG[prob]}-{PROB_VERSION[prob]}",
                        method, prefix)


def p1_prefix(method, lr, batch):
    return f"bench_p1_{method}_{batch}_lr{lr}"


def p2_prefix(method, hp_tag, lr, batch):
    return f"bench_p2_{method}_{hp_tag}_{batch}_lr{lr}"


def load_min_val_regret(dirpath):
    """Min of 'eval' column in val_logs.csv. Falls back to parsing
    'val MSE (no solver)' lines in log.txt for --skip_solver_eval runs
    (energy mse/dfl/identity). Returns None if neither source is usable."""
    path = os.path.join(dirpath, "val_logs.csv")
    if os.path.exists(path):
        try:
            min_v = None
            with open(path, newline="") as f:
                reader = csv.DictReader(f)
                if "eval" in (reader.fieldnames or []):
                    for row in reader:
                        try:
                            v = float(row["eval"])
                        except (TypeError, ValueError):
                            continue
                        if min_v is None or v < min_v:
                            min_v = v
            if min_v is not None:
                return min_v
        except Exception:
            pass

    # Fallback: --skip_solver_eval runs only log val pred MSE to log.txt.
    log_path = os.path.join(dirpath, "log.txt")
    if not os.path.exists(log_path):
        return None
    try:
        min_v = None
        with open(log_path) as f:
            for line in f:
                m = _VAL_MSE_LINE.search(line)
                if not m:
                    continue
                try:
                    v = float(m.group(2))
                except ValueError:
                    continue
                if min_v is None or v < min_v:
                    min_v = v
        return min_v
    except Exception:
        return None


def load_test_regret(dirpath, prob):
    """Return scalar test regret (absolute for portfolio, else relative)."""
    rpath = os.path.join(dirpath, "results.npy")
    if not os.path.exists(rpath):
        return None
    try:
        r = np.load(rpath, allow_pickle=True)
        regret = np.asarray(r[1], dtype=float)
        opt    = np.asarray(r[0], dtype=float)
        abs_r  = float(np.mean(regret))
        if prob in USE_ABSOLUTE:
            return abs_r
        mean_opt = float(np.mean(np.abs(opt)))
        return abs_r / mean_opt if mean_opt > 0 else None
    except Exception:
        return None


def pick_winner(prob, method, lr, batch, p1_val=None):
    """
    For a Phase-2 method, enumerate every (sweep, value) candidate, read each
    val_logs.csv, return (best_val, best_hp_name, best_hp_value, best_tag,
    best_prefix, n_done, n_total). For a non-Phase-2 method, return the Phase-1
    entry instead. None on failure.
    """
    if method not in HP_SWEEPS:
        # PtO methods (no Phase-2 HP sweep): propagate Phase-1 winner. We do
        # NOT re-read val regret here, because mse_train selects on training
        # MSE and mse_val on val MSE — those signals would be inaccessible
        # via val_logs.csv. Caller passes p1_val from bench_p1_best_val.json
        # which already has the correct selection-metric value for the method.
        prefix = p1_prefix(method, lr, batch)
        d = run_dir(prob, method, prefix)
        v = p1_val if p1_val is not None else load_min_val_regret(d)
        if v is None:
            return None
        return dict(val=v, hp_name=None, hp_value=None, hp_tag=None,
                    prefix=prefix, run_dir=d, n_done=1, n_total=1)

    candidates = []  # list of (val, hp_name, hp_value, tag, prefix, dirpath)
    n_total = 0
    for sweep in HP_SWEEPS[method]:
        for v in sweep["vals"]:
            n_total += 1
            tag = sweep["tag_fn"](v)
            prefix = p2_prefix(method, tag, lr, batch)
            d = run_dir(prob, method, prefix)
            min_val = load_min_val_regret(d)
            if min_val is None:
                continue
            candidates.append((min_val, sweep["hp"], v, tag, prefix, d))

    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0])
    val, hp_name, hp_value, tag, prefix, d = candidates[0]
    return dict(val=val, hp_name=hp_name, hp_value=hp_value, hp_tag=tag,
                prefix=prefix, run_dir=d, n_done=len(candidates), n_total=n_total)


def fixed_default_entry(prob, method, lr, batch):
    """
    For a Phase-2 method, return the entry where every method-specific HP
    is at its default (* in submit_bench_p2.sh). For perturb, that means
    (sigma=1.0, n_samples=10), which is tagged 's1p0_n10' in both sub-sweeps —
    use the first matching prefix found. Returns same dict shape as pick_winner
    so downstream code can compare apples to apples.
    """
    if method not in HP_SWEEPS:
        prefix = p1_prefix(method, lr, batch)
        d = run_dir(prob, method, prefix)
        v = load_min_val_regret(d)
        if v is None:
            return None
        return dict(val=v, hp_name=None, hp_value=None, hp_tag=None,
                    prefix=prefix, run_dir=d)

    for sweep in HP_SWEEPS[method]:
        tag = sweep["tag_fn"](sweep["default"])
        prefix = p2_prefix(method, tag, lr, batch)
        d = run_dir(prob, method, prefix)
        min_val = load_min_val_regret(d)
        if min_val is not None:
            return dict(val=min_val, hp_name=sweep["hp"],
                        hp_value=sweep["default"], hp_tag=tag,
                        prefix=prefix, run_dir=d)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--problem", default=None)
    ap.add_argument("--method", default=None)
    ap.add_argument("--p1_best_json", default=P1_BEST_PATH)
    ap.add_argument("--out", default=OUT_PATH)
    args = ap.parse_args()

    if not os.path.exists(args.p1_best_json):
        print(f"error: {args.p1_best_json} not found. "
              f"Run `python rethink_exp/collect_bench_p1.py --metric val` first.")
        sys.exit(1)

    with open(args.p1_best_json) as f:
        p1_best = json.load(f)

    problems = [args.problem] if args.problem else PROBLEMS
    methods  = [args.method]  if args.method  else ALL_METHODS

    out = {}
    for method in methods:
        out[method] = {}
        for prob in problems:
            allowed = METHOD_PROBLEMS.get(method)
            if allowed is not None and prob not in allowed:
                continue
            cfg = p1_best.get(method, {}).get(prob)
            if not cfg:
                continue
            lr = cfg.get("lr")
            batch = cfg.get("batch")
            if lr is None or batch is None:
                continue

            winner = pick_winner(prob, method, lr, batch, p1_val=cfg.get("val"))
            if winner is None:
                continue
            test = load_test_regret(winner["run_dir"], prob)
            entry = {
                "lr": lr, "batch": batch,
                "hp_name":  winner["hp_name"],
                "hp_value": winner["hp_value"],
                "hp_tag":   winner["hp_tag"],
                "prefix":   winner["prefix"],
                "run_dir":  winner["run_dir"],
                "val":  round(winner["val"], 6),
                "test": None if test is None else round(test, 6),
                "n_done":  winner["n_done"],
                "n_total": winner["n_total"],
                "phase":   1 if method not in HP_SWEEPS else 2,
            }
            # Also record the fixed-default cell so the tuning-benefit script
            # can read it without re-doing path math.
            if method in HP_SWEEPS:
                fixed = fixed_default_entry(prob, method, lr, batch)
                if fixed is not None:
                    entry["fixed"] = {
                        "hp_name":  fixed["hp_name"],
                        "hp_value": fixed["hp_value"],
                        "hp_tag":   fixed["hp_tag"],
                        "prefix":   fixed["prefix"],
                        "run_dir":  fixed["run_dir"],
                        "val":  round(fixed["val"], 6),
                        "test": None if (t := load_test_regret(fixed["run_dir"], prob)) is None
                                else round(t, 6),
                    }
            out[method][prob] = entry

    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)

    # Summary
    n_full = n_partial = n_missing = 0
    for method in ALL_METHODS:
        allowed = METHOD_PROBLEMS.get(method)
        for prob in PROBLEMS:
            if allowed is not None and prob not in allowed:
                continue
            entry = out.get(method, {}).get(prob)
            if entry is None:
                n_missing += 1
            elif entry["n_done"] == entry["n_total"]:
                n_full += 1
            else:
                n_partial += 1
    print(f"wrote {args.out}")
    print(f"full: {n_full}   partial: {n_partial}   missing: {n_missing}")


if __name__ == "__main__":
    main()
