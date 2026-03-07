"""
Shared helpers for SoftDecision Perturbed experiment notebooks.

Usage:
    from exp_helpers import read_result, read_traces, read_full_traces, get_best_val_regret
"""

import ast
import glob
import os
import re

import numpy as np
import pandas as pd


# ── Result reading ────────────────────────────────────────────────────

def read_result(data_name, prefix, model="perturb"):
    """Read final results from log.txt.  Returns dict or None."""
    log_path = os.path.join("saved_records", data_name, model, prefix, "log.txt")
    if not os.path.exists(log_path):
        return None
    with open(log_path, "r") as f:
        lines = f.readlines()
    if not lines:
        return None
    last = lines[-1].strip()
    parts = last.split()
    try:
        vals = [float(x) for x in parts[-4:]]
        return {
            "objective": vals[0],
            "regret": vals[1],
            "train_time": vals[2],
            "test_time": vals[3],
        }
    except (ValueError, IndexError):
        return None


# ── Training trace reading ────────────────────────────────────────────

_METRIC_PAT = re.compile(
    r"Iter\s+(\d+),\s+(train|val)\s+Objective:\s+([\d.eE+-]+),\s+"
    r"Loss:\s+([\d.eE+-]+)\s+Pred Loss:\s+([\d.eE+-]+),\s+"
    r"(?:regret|uplift):\s+([\d.eE+-]+)"
)

_DIAG_PAT = re.compile(
    r"epoch\s+(\d+):\s+\|coeff_hat\|\s+=\s+([\d.]+),\s+sigma\s+=\s+([\d.]+),\s+"
    r"sigma/\|coeff_hat\|\s+=\s+([\d.]+)"
)


def read_traces(data_name, prefix, model="perturb"):
    """Return (train_df, val_df) with columns [epoch_num, obj, loss, pred_loss, eval].

    Tries CSV first, falls back to parsing log.txt.
    Returns (None, None) if unavailable.
    """
    base = os.path.join("saved_records", data_name, model, prefix)
    train_csv = os.path.join(base, "train_logs.csv")
    val_csv = os.path.join(base, "val_logs.csv")

    if os.path.exists(train_csv) and os.path.exists(val_csv):
        try:
            tdf = pd.read_csv(train_csv)
            vdf = pd.read_csv(val_csv)
            tdf["epoch_num"] = tdf["epoch"].str.extract(r"(\d+)").astype(int)
            vdf["epoch_num"] = vdf["epoch"].str.extract(r"(\d+)").astype(int)
            return tdf, vdf
        except Exception:
            pass

    # Fallback: parse log.txt
    log_path = os.path.join(base, "log.txt")
    if not os.path.exists(log_path):
        return None, None

    train_rows, val_rows = [], []
    with open(log_path) as f:
        for line in f:
            m = _METRIC_PAT.search(line)
            if m:
                row = {
                    "epoch_num": int(m.group(1)),
                    "obj": float(m.group(3)),
                    "loss": float(m.group(4)),
                    "pred_loss": float(m.group(5)),
                    "eval": float(m.group(6)),
                }
                if m.group(2) == "train":
                    train_rows.append(row)
                else:
                    val_rows.append(row)

    if not train_rows and not val_rows:
        return None, None
    tdf = pd.DataFrame(train_rows) if train_rows else None
    vdf = pd.DataFrame(val_rows) if val_rows else None
    return tdf, vdf


def read_full_traces(data_name, prefix, model="perturb"):
    """Read training logs including |coeff_hat|, sigma diagnostics.

    Returns (train_df, val_df, diag_df).  Any may be None.
    """
    base = os.path.join("saved_records", data_name, model, prefix)
    log_path = os.path.join(base, "log.txt")
    if not os.path.exists(log_path):
        return None, None, None

    train_rows, val_rows, diag_rows = [], [], []
    with open(log_path) as f:
        for line in f:
            m = _METRIC_PAT.search(line)
            if m:
                row = {
                    "epoch": int(m.group(1)),
                    "obj": float(m.group(3)),
                    "loss": float(m.group(4)),
                    "pred_loss": float(m.group(5)),
                    "regret": float(m.group(6)),
                }
                if m.group(2) == "train":
                    train_rows.append(row)
                else:
                    val_rows.append(row)
                continue
            m = _DIAG_PAT.search(line)
            if m:
                diag_rows.append({
                    "epoch": int(m.group(1)),
                    "coeff_hat_mag": float(m.group(2)),
                    "sigma": float(m.group(3)),
                    "sigma_over_coeff": float(m.group(4)),
                })

    tdf = pd.DataFrame(train_rows) if train_rows else None
    vdf = pd.DataFrame(val_rows) if val_rows else None
    ddf = pd.DataFrame(diag_rows) if diag_rows else None
    return tdf, vdf, ddf


def get_best_val_regret(val_df):
    """Return the minimum validation regret (eval or regret column)."""
    if val_df is None or len(val_df) == 0:
        return float("inf")
    if "regret" in val_df.columns:
        return val_df["regret"].min()
    return val_df["eval"].min()


# ── Experiment status helpers ─────────────────────────────────────────

def collect_results(registry, data_name):
    """Collect results for a dict of {prefix: hp_dict} experiments.

    Returns a DataFrame with columns from hp_dict plus
    objective, regret, train_time, test_time, status.
    """
    rows = []
    for prefix, hp in registry.items():
        result = read_result(data_name, prefix)
        row = {"prefix": prefix, **hp}
        if result:
            row.update(result)
            row["status"] = "done"
        else:
            row["status"] = "pending"
        rows.append(row)
    return pd.DataFrame(rows)


def scan_slurm_errors(pattern_glob, keywords=None):
    """Scan SLURM .err files matching a glob for error keywords.

    Returns number of problematic files found.
    """
    import glob as _glob

    if keywords is None:
        keywords = ["error", "traceback", "exception", "killed", "oom"]

    err_files = sorted(_glob.glob(pattern_glob))
    print(f"Found {len(err_files)} error files\n")

    problems = 0
    for f in err_files:
        with open(f) as fh:
            content = fh.read()
        lines = content.strip().split("\n")
        error_lines = [
            l for l in lines
            if any(kw in l.lower() for kw in keywords)
            and "deprecat" not in l.lower()
            and "warning" not in l.lower()
        ]
        if error_lines:
            problems += 1
            short_name = os.path.basename(f)
            print(f"⚠ {short_name}:")
            for l in error_lines[:5]:
                print(f"   {l.strip()}")
            print()

    if problems == 0:
        print("✓ No errors detected.")
    return problems


# ── Config extraction ─────────────────────────────────────────────────

def _parse_model_configs(log_path):
    """Parse first 'model configs: {…}' dict from a log file."""
    with open(log_path, "r", errors="replace") as f:
        for line in f:
            m = re.search(r"model configs:\s*(\{.*\})", line)
            if m:
                try:
                    return ast.literal_eval(m.group(1))
                except (ValueError, SyntaxError):
                    return {}
    return {}


def _parse_args_namespace(log_path):
    """Parse first 'args: Namespace(…)' from a log file into a dict."""
    with open(log_path, "r", errors="replace") as f:
        text = f.read()
    m = re.search(r"args:\s*Namespace\((.+?)\)\s*\n", text, re.DOTALL)
    if not m:
        return {}
    raw = m.group(1)
    d = {}
    # Namespace prints key=value pairs separated by ', '
    for token in re.split(r",\s+(?=\w+=)", raw):
        token = token.strip()
        if "=" not in token:
            continue
        k, v = token.split("=", 1)
        try:
            d[k.strip()] = ast.literal_eval(v.strip())
        except (ValueError, SyntaxError):
            d[k.strip()] = v.strip().strip("'\"")
    return d


def extract_experiment_config(data_name, prefix, model="perturb"):
    """Extract hyperparameters for a single experiment from its log.txt.

    Returns a dict with normalised keys:
        problem, prefix, sigma, n_samples, output_activation,
        sigma_schedule, sigma_start, sigma_end, sigma_n_epochs,
        sigma_warmup_epochs, pred_loss_weight, loss_type, lr,
        n_epochs, patience, val_frac, wave
    """
    log_path = os.path.join("saved_records", data_name, model, prefix, "log.txt")
    if not os.path.exists(log_path):
        return None

    mc = _parse_model_configs(log_path)
    ns = _parse_args_namespace(log_path)

    # Infer wave from prefix
    wm = re.search(r"_w(\d+)_", prefix)
    wave = int(wm.group(1)) if wm else 0

    return {
        "problem": data_name,
        "prefix": prefix,
        "wave": wave,
        "sigma": mc.get("sigma"),
        "n_samples": mc.get("n_samples"),
        "output_activation": mc.get("output_activation", "none"),
        "sigma_schedule": mc.get("sigma_schedule", "constant"),
        "sigma_start": mc.get("sigma_start"),
        "sigma_end": mc.get("sigma_end"),
        "sigma_n_epochs": mc.get("sigma_n_epochs"),
        "sigma_warmup_epochs": mc.get("sigma_warmup_epochs"),
        "pred_loss_weight": mc.get("pred_loss_weight", 0),
        "loss_type": mc.get("loss_type", "regret"),
        "lr": ns.get("lr"),
        "n_epochs": ns.get("n_epochs"),
        "patience": ns.get("patience"),
        "val_frac": ns.get("val_frac", 0.2),
    }


def catalog_all_experiments(base="saved_records", model="perturb"):
    """Scan all problems and return a DataFrame of every experiment's config."""
    rows = []
    for prob_dir in sorted(glob.glob(os.path.join(base, "*/", model))):
        data_name = prob_dir.split(os.sep)[-2]
        perturb_dir = os.path.join(base, data_name, model)
        if not os.path.isdir(perturb_dir):
            continue
        for prefix in sorted(os.listdir(perturb_dir)):
            cfg = extract_experiment_config(data_name, prefix, model=model)
            if cfg is not None:
                rows.append(cfg)
    df = pd.DataFrame(rows)
    return df
