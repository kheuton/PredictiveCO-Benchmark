"""
export_sweep_status_xlsx.py
---------------------------
Build a colored Excel spreadsheet summarising Phase 1 / Phase 2 completion status
for every (method, task) cell in the benchmark re-run sweep.

Categories per cell:
  - COMPLETE       (green)  : P1 fully done AND (PtO-only OR P2 fully done)
  - P2_INPROGRESS  (yellow) : P1 fully done, P2 has remaining jobs (running or missing)
  - P1_INPROGRESS  (red)    : P1 still has running or missing jobs
  - NA             (gray)   : cell not applicable (e.g. qptl on non-LP tasks, pg/shortestpath)

The manifests `sweep_manifest_p1.json` and `sweep_manifest_p2.json` reflect the
*current* best (lr, batch) per (method, task) — i.e. post 5-LR expansion — so
cells whose best (lr, batch) flipped recently are checked against the freshly
resubmitted P2 jobs automatically.
"""

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from sweep_status import cell_status, get_queued_job_names, prefix_to_job_name

from openpyxl import Workbook
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Display order matching CLAUDE.md / sweep_status.py
METHODS = [
    "mse", "dfl", "identity", "spo", "nce", "blackbox",
    "pointLTR", "pairLTR", "listLTR", "lodl", "perturb",
    "pg", "qptl", "cpLayer", "dad",
]
TASKS = [
    "knapsack", "knapsack-real", "energy", "budgetalloc", "cubic",
    "bipartitematching", "portfolio", "asurv", "cook_county", "speed_humps",
    "sp_synth", "sp_planted", "shortestpath",
]

# Methods that have NO Phase 2 (pure PtO baselines + cpLayer)
NO_P2_METHODS = {"mse", "identity", "spo", "nce", "pointLTR", "pairLTR", "cpLayer"}


def load_manifests():
    with open(os.path.join(ROOT, "sweep_manifest_p1.json")) as f:
        m1 = json.load(f)
    with open(os.path.join(ROOT, "sweep_manifest_p2.json")) as f:
        m2 = json.load(f)
    p1 = {}
    p2 = {}
    for e in m1:
        p1.setdefault((e["opt_model"], e["prob"]), []).append(e["prefix"])
    for e in m2:
        p2.setdefault((e["opt_model"], e["prob"]), []).append(e["prefix"])
    return p1, p2


def summarize(prefixes, method, task, queued, phase):
    """Return (n_done, n_running, n_stranded, n_missing, n_total)."""
    counts = {"done": 0, "running": 0, "stranded": 0, "missing": 0}
    for pfx in prefixes:
        s = cell_status(task, method, pfx, queued=queued, phase=phase)
        counts[s] += 1
    return counts, len(prefixes)


def classify(p1_counts, p1_total, p2_prefs, p2_counts, p2_total, has_p2):
    """Return category string + short text label."""
    p1_done = p1_counts["done"]
    p1_remain = p1_counts["running"] + p1_counts["stranded"] + p1_counts["missing"]

    if p1_total == 0:
        return "NA", ""

    if p1_remain > 0:
        # P1 stragglers
        label = f"P1 {p1_done}/{p1_total}"
        if p1_counts["stranded"]:
            label += f" (T{p1_counts['stranded']})"
        return "P1_INPROGRESS", label

    # P1 fully done
    if not has_p2:
        return "COMPLETE", f"P1 ✓ ({p1_total}/{p1_total})"

    if p2_total == 0:
        # P2 expected but no entries — shouldn't really happen for valid cells
        return "P2_INPROGRESS", "P2 not submitted"

    p2_done = p2_counts["done"]
    p2_remain = p2_counts["running"] + p2_counts["stranded"] + p2_counts["missing"]
    if p2_remain == 0:
        return "COMPLETE", f"P1 ✓  P2 ✓ ({p2_total}/{p2_total})"

    extra = []
    if p2_counts["running"]:
        extra.append(f"R{p2_counts['running']}")
    if p2_counts["stranded"]:
        extra.append(f"T{p2_counts['stranded']}")
    if p2_counts["missing"]:
        extra.append(f"M{p2_counts['missing']}")
    tag = "/".join(extra) if extra else ""
    return "P2_INPROGRESS", f"P2 {p2_done}/{p2_total} ({tag})"


def main():
    p1_table, p2_table = load_manifests()
    queued = get_queued_job_names()

    wb = Workbook()
    ws = wb.active
    ws.title = "Sweep Status"

    # Styles
    fills = {
        "COMPLETE":      PatternFill("solid", fgColor="63BE7B"),  # green
        "P2_INPROGRESS": PatternFill("solid", fgColor="FFEB84"),  # yellow
        "P1_INPROGRESS": PatternFill("solid", fgColor="F8696B"),  # red
        "NA":            PatternFill("solid", fgColor="D9D9D9"),  # gray
    }
    header_fill = PatternFill("solid", fgColor="305496")
    header_font = Font(bold=True, color="FFFFFF", size=11)
    body_font = Font(size=8)
    center = Alignment(horizontal="center", vertical="center", wrap_text=True)
    thin = Side(border_style="thin", color="BFBFBF")
    border = Border(left=thin, right=thin, top=thin, bottom=thin)

    # Header row
    ws.cell(row=1, column=1, value="method \\ task")
    ws.cell(row=1, column=1).fill = header_fill
    ws.cell(row=1, column=1).font = header_font
    ws.cell(row=1, column=1).alignment = center
    ws.cell(row=1, column=1).border = border

    for j, task in enumerate(TASKS, start=2):
        c = ws.cell(row=1, column=j, value=task)
        c.fill = header_fill
        c.font = header_font
        c.alignment = center
        c.border = border

    # Body rows
    summary = {"COMPLETE": 0, "P2_INPROGRESS": 0, "P1_INPROGRESS": 0, "NA": 0}

    for i, method in enumerate(METHODS, start=2):
        # Row label
        c = ws.cell(row=i, column=1, value=method)
        c.fill = header_fill
        c.font = header_font
        c.alignment = center
        c.border = border

        for j, task in enumerate(TASKS, start=2):
            p1_prefs = p1_table.get((method, task), [])
            p2_prefs = p2_table.get((method, task), [])

            has_p2 = (method not in NO_P2_METHODS)
            p1_counts, p1_total = summarize(p1_prefs, method, task, queued=queued, phase=1)
            p2_counts, p2_total = summarize(p2_prefs, method, task, queued=queued, phase=2)

            category, label = classify(
                p1_counts, p1_total, p2_prefs, p2_counts, p2_total, has_p2,
            )
            summary[category] += 1

            cell = ws.cell(row=i, column=j, value=label)
            cell.fill = fills[category]
            cell.font = body_font
            cell.alignment = center
            cell.border = border

    # Sizing: row+col labels large, grid cells small
    ws.column_dimensions["A"].width = 14
    for j in range(2, 2 + len(TASKS)):
        ws.column_dimensions[get_column_letter(j)].width = 11
    ws.row_dimensions[1].height = 26
    for i in range(2, 2 + len(METHODS)):
        ws.row_dimensions[i].height = 22

    # Freeze panes
    ws.freeze_panes = "B2"

    # Legend on a second sheet
    legend = wb.create_sheet("Legend")
    legend_rows = [
        ("Category", "Meaning"),
        ("COMPLETE", "P1 fully done AND (PtO-only OR P2 fully done with current best LR/batch)"),
        ("P2_INPROGRESS", "P1 done; P2 has running/stranded/missing jobs (incl. cells flipped by 5-LR expansion)"),
        ("P1_INPROGRESS", "P1 still has running/stranded jobs"),
        ("NA", "Method × task combination not part of the sweep"),
        ("", ""),
        ("Counts", ""),
        ("COMPLETE", summary["COMPLETE"]),
        ("P2_INPROGRESS", summary["P2_INPROGRESS"]),
        ("P1_INPROGRESS", summary["P1_INPROGRESS"]),
        ("NA", summary["NA"]),
    ]
    for r, (a, b) in enumerate(legend_rows, start=1):
        legend.cell(row=r, column=1, value=a)
        legend.cell(row=r, column=2, value=b)
        if a in fills:
            legend.cell(row=r, column=1).fill = fills[a]
    legend.column_dimensions["A"].width = 18
    legend.column_dimensions["B"].width = 80

    out_path = os.path.join(ROOT, "sweep_status.xlsx")
    wb.save(out_path)
    print(f"Wrote {out_path}")
    print("Summary:", summary)


if __name__ == "__main__":
    main()
