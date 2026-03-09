"""
Figure: Knapsack benchmark-scale adaptive sigma sweep results.

Wave 3: 320+80 train/val, 200 test (Knapsack_7.pkl), dense model only.
Grid: 3 targets × 3 LRs × 2 controllers × 2 solvers = 36 runs.

Layout: 2 rows (DP solver / Gurobi solver) × 3 panels
  Panel 1 — OCV_Y target vs best val regret
            (global=solid, per-instance=dashed, MSE baseline=gray hline)
  Panel 2 — OCV_Y target vs final sigma (log-scale)
  Panel 3 — FracImproving for best run per (LR, controller)

Colors by LR; line style by controller (solid=global, dashed=per-instance).

MSE baseline: from small-scale results (dense lr=5e-2 → 0.064).
Override with --mse_baseline if benchmark-scale MSE is available.

Usage:
    python rethink_exp/fig_kn_bench_results.py
    python rethink_exp/fig_kn_bench_results.py --solver dp    # DP only
    python rethink_exp/fig_kn_bench_results.py --solver grb   # Gurobi only
    python rethink_exp/fig_kn_bench_results.py --out fig.png
    python rethink_exp/fig_kn_bench_results.py --mse_baseline 0.058
"""

import argparse
import glob
import os

import matplotlib.pyplot as plt
import numpy as np

ADAPTIVE_BASE = 'saved_records/knapsack-gen/perturb_adaptive_sweep'
MSE_BASE      = 'saved_records/knapsack-gen/mse'

STUCK_THRESH = 0.40

LR_COLORS = {'5e-2': '#e66101', '1e-2': '#5e3c99', '5e-3': '#1a9641'}

# (prefix_pattern, lr, controller_label, linestyle, stem)
def build_configs(solver, lrs=None, controllers=None):
    """Build list of (prefix, lr_label, ctrl_label, ls, stem) tuples."""
    all_lrs  = lrs        or ['5e-2', '1e-2', '5e-3']
    all_ctrls = controllers or ['global', 'per_instance']
    configs = []
    for ctrl in all_ctrls:
        ls   = '--' if ctrl == 'per_instance' else '-'
        stem = 'pi' if ctrl == 'per_instance' else 'prop'
        for lr in all_lrs:
            prefix = f'kn_bench_{solver}_{ctrl}_lr{lr}'
            configs.append((prefix, lr, ctrl, ls, stem))
    return configs


# ---- Loaders ----

def load_bench_mse():
    """Return best normalized test regret across LRs from benchmark-scale MSE runs.

    Reads saved_records/knapsack-gen/mse/kn_bench_mse_dense_lr{lr}/log.txt.
    Returns (best_regret, lr_label) or (None, None) if no runs are complete.
    """
    best = None
    best_lr = None
    for lr in ['5e-2', '1e-2', '5e-3']:
        log_path = os.path.join(MSE_BASE, f'kn_bench_mse_dense_lr{lr}', 'log.txt')
        if not os.path.exists(log_path):
            continue
        try:
            with open(log_path) as f:
                lines = [l.strip() for l in f if l.strip()]
            avg_regret = float(lines[-1].split()[-3])
            opt_line = next(l for l in reversed(lines) if '[Optimal Obj]' in l)
            opt_obj = float(opt_line.split('[Optimal Obj]:')[1].split()[0])
            regret = avg_regret / (opt_obj + 1e-8)
            if best is None or regret < best:
                best = regret
                best_lr = lr
        except Exception:
            pass
    return best, best_lr


def load_adaptive(prefix, model='dense', stem='prop'):
    rows = []
    pat = os.path.join(ADAPTIVE_BASE, prefix, f'{stem}_*_{model}.npz')
    for f in sorted(glob.glob(pat)):
        d    = np.load(f, allow_pickle=True)
        name = os.path.basename(f).replace('.npz', '')
        # filename: prop_tX.XXX_s1_dense  or  pi_tX.XXX_s1_dense
        target = float(name.split('_')[1][1:])
        rows.append(dict(
            target      = target,
            best_val    = float(d['val_regret'].min()),
            test_regret = float(d['test_regret']) if 'test_regret' in d else float('nan'),
            final_sigma = float(d['adaptive_sigma'][-1]),
            frac_traj   = d['frac_improving'],
            val_traj    = d['val_regret'],
            n_epochs    = len(d['val_regret']),
        ))
    rows.sort(key=lambda r: r['target'])
    return rows


# ---- Main ----

def plot_solver_row(ax_val, ax_sigma, ax_frac, configs, mse_baseline, solver_label):
    has_data = False

    for prefix, lr_label, ctrl_label, ls, stem in configs:
        rows = load_adaptive(prefix, 'dense', stem)
        if not rows:
            continue
        has_data = True
        color = LR_COLORS[lr_label]
        ctrl_short = 'pi' if ctrl_label == 'per_instance' else 'global'

        targets = np.array([r['target']      for r in rows])
        vals    = np.array([r['best_val']    for r in rows])
        sigmas  = np.array([r['final_sigma'] for r in rows])
        stuck   = vals >= STUCK_THRESH

        label = f'lr={lr_label} ({ctrl_short})'

        if (~stuck).any():
            ax_val.plot(targets[~stuck], vals[~stuck], 'o' + ls,
                        color=color, lw=1.8, ms=6, label=label)
            ax_sigma.semilogy(targets[~stuck], sigmas[~stuck], 'o' + ls,
                              color=color, lw=1.8, ms=6, label=label)
        if stuck.any():
            ax_val.scatter(targets[stuck], vals[stuck],
                           marker='x', color=color, s=80, lw=2, zorder=5)
            ax_sigma.scatter(targets[stuck], sigmas[stuck],
                             marker='x', color=color, s=80, lw=2, zorder=5)

    # MSE baseline hline
    if mse_baseline is not None:
        ax_val.axhline(mse_baseline, color='gray', lw=2.0, ls=':',
                       label=f'MSE dense best ({mse_baseline:.4f})', zorder=3)
    elif not has_data:
        ax_val.text(0.5, 0.5, 'No results yet', transform=ax_val.transAxes,
                    ha='center', va='center', fontsize=12, color='gray')

    ax_val.set_xscale('log')
    ax_val.set_yscale('log')
    ax_val.set_xlabel('OCV_Y target', fontsize=11)
    ax_val.set_ylabel('Best val regret', fontsize=11)
    ax_val.set_title(f'{solver_label} — val regret vs OCV_Y target', fontsize=11)
    ax_val.legend(fontsize=7, ncol=2)
    ax_val.grid(True, alpha=0.3, which='both')

    ax_sigma.set_xscale('log')
    ax_sigma.set_xlabel('OCV_Y target', fontsize=11)
    ax_sigma.set_ylabel('Final sigma', fontsize=11)
    ax_sigma.set_title(f'{solver_label} — final sigma vs OCV_Y target', fontsize=11)
    ax_sigma.legend(fontsize=7, ncol=2)
    ax_sigma.grid(True, alpha=0.3, which='both')

    # Panel 3: FracImproving for best run per (LR, controller)
    for prefix, lr_label, ctrl_label, ls, stem in configs:
        rows = load_adaptive(prefix, 'dense', stem)
        if not rows:
            continue
        best  = min(rows, key=lambda r: r['best_val'])
        color = LR_COLORS[lr_label]
        ctrl_short = 'pi' if ctrl_label == 'per_instance' else 'global'
        epochs = np.arange(1, len(best['frac_traj']) + 1)
        ax_frac.plot(
            epochs, best['frac_traj'],
            color=color, ls=ls, lw=1.5, alpha=0.85,
            label=f"lr={lr_label} {ctrl_short} t={best['target']:.3f} "
                  f"val={best['best_val']:.4f}",
        )

    ax_frac.set_xlabel('Epoch', fontsize=11)
    ax_frac.set_ylabel('FracImproving', fontsize=11)
    ax_frac.set_title(f'{solver_label} — FracImproving (best per LR+ctrl)', fontsize=11)
    ax_frac.set_ylim(-0.05, 1.05)
    ax_frac.axhline(0, color='k', lw=0.8, ls='--', alpha=0.4)
    ax_frac.legend(fontsize=7)
    ax_frac.grid(True, alpha=0.3)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--solver', choices=['dp', 'grb', 'both'], default='both',
                        help='Which solver results to show (default: both)')
    parser.add_argument('--mse_baseline', type=float, default=None,
                        help='Override MSE baseline value (auto-loaded from benchmark MSE runs if available)')
    parser.add_argument('--out', type=str, default=None)
    args = parser.parse_args()

    # Auto-load benchmark-scale MSE baseline; fall back to small-scale value
    mse_baseline = args.mse_baseline
    if mse_baseline is None:
        mse_val, mse_lr = load_bench_mse()
        if mse_val is not None:
            mse_baseline = mse_val
            print(f'MSE benchmark baseline: {mse_val:.4f} (lr={mse_lr})')
        else:
            mse_baseline = 0.064
            print('MSE benchmark baseline not yet available; using small-scale fallback (0.064)')

    solvers = ['dp', 'grb'] if args.solver == 'both' else [args.solver]
    solver_labels = {'dp': 'DP solver (n=100, 300ep)', 'grb': 'Gurobi (n=10, 100ep)'}

    n_rows = len(solvers)
    fig, axes = plt.subplots(n_rows, 3, figsize=(18, 5 * n_rows))
    if n_rows == 1:
        axes = axes[np.newaxis, :]  # ensure 2D

    fig.suptitle(
        'Knapsack benchmark scale — OCV_Y adaptive sigma (dense model, 320+80 train/val)',
        fontsize=13, fontweight='bold',
    )

    for row_idx, solver in enumerate(solvers):
        configs = build_configs(solver)
        ax_val, ax_sigma, ax_frac = axes[row_idx]
        plot_solver_row(ax_val, ax_sigma, ax_frac, configs,
                        args.mse_baseline, solver_labels[solver])

    plt.tight_layout()

    out_path = args.out or os.path.join(ADAPTIVE_BASE, 'fig_kn_bench_results.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f'Saved → {out_path}')
    plt.close()

    # ---- Summary table ----
    print('\n--- Best val regret summary (benchmark scale) ---')
    for solver in solvers:
        configs = build_configs(solver)
        for prefix, lr_label, ctrl_label, _, stem in configs:
            rows = load_adaptive(prefix, 'dense', stem)
            if rows:
                best_row = min(rows, key=lambda r: r['best_val'])
                ctrl_short = 'pi' if ctrl_label == 'per_instance' else 'global'
                test_str = f'  test={best_row["test_regret"]:.4f}' \
                           if not np.isnan(best_row["test_regret"]) else ''
                print(f'  {solver} {ctrl_short:6s} lr={lr_label} dense: '
                      f'val={best_row["best_val"]:.4f}{test_str} at t={best_row["target"]:.3f} '
                      f'({best_row["n_epochs"]} epochs)')
    print(f'\n  MSE dense baseline: {mse_baseline:.4f}')


if __name__ == '__main__':
    main()
