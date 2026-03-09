"""
Compare DP (n=100) vs Gurobi (n=10) training trajectories for first 100 epochs.

Both solvers are exact for knapsack — the only difference is n_samples.
Shows val_regret and frac_improving side-by-side, matched by LR and controller.

Usage:
    conda run -n pco_bench_rhel7 python rethink_exp/fig_kn_dp_vs_grb_trajectories.py
    conda run -n pco_bench_rhel7 python rethink_exp/fig_kn_dp_vs_grb_trajectories.py --ctrl pi
    conda run -n pco_bench_rhel7 python rethink_exp/fig_kn_dp_vs_grb_trajectories.py --ctrl global
"""

import argparse
import glob
import os

import matplotlib.pyplot as plt
import numpy as np

BASE = 'saved_records/knapsack-gen/perturb_adaptive_sweep'
LR_COLORS = {'5e-2': '#e66101', '1e-2': '#5e3c99', '5e-3': '#1a9641'}
N_EPOCHS = 100  # compare first 100 epochs only


def load_run(solver, ctrl, lr, target=None):
    """Load npz for a given (solver, ctrl, lr), optionally at a fixed target.

    If target is None, picks the best-val target.
    If target is given, finds the file whose filename target is closest.
    """
    stem = 'pi' if ctrl == 'per_instance' else 'prop'
    prefix = f'kn_bench_{solver}_{ctrl}_lr{lr}'
    pat = os.path.join(BASE, prefix, f'{stem}_*_dense.npz')
    files = sorted(glob.glob(pat))
    if not files:
        return None

    def file_target(f):
        return float(os.path.basename(f).split('_')[1][1:])

    if target is None:
        best_f, best_val = None, float('inf')
        for f in files:
            d = np.load(f, allow_pickle=True)
            v = float(d['val_regret'].min())
            if v < best_val:
                best_val = v
                best_f = f
        chosen = best_f
    else:
        chosen = min(files, key=lambda f: abs(file_target(f) - target))

    d = np.load(chosen, allow_pickle=True)
    t = file_target(chosen)
    return dict(
        val_regret     = d['val_regret'],
        frac_improving = d['frac_improving'],
        ocv_y          = d['ocv_y'],
        adaptive_sigma = d['adaptive_sigma'],
        target         = t,
        best_val       = float(d['val_regret'].min()),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ctrl', choices=['global', 'per_instance', 'both'], default='both')
    parser.add_argument('--target', type=float, default=0.066,
                        help='Fixed OCV_Y target to compare (default: 0.066)')
    parser.add_argument('--out', default=None)
    args = parser.parse_args()

    lrs = ['5e-2', '1e-2', '5e-3']
    ctrls = ['global', 'per_instance'] if args.ctrl == 'both' else [args.ctrl]
    ctrl_labels = {'global': 'global', 'per_instance': 'per-instance'}

    metrics = ['val_regret', 'frac_improving', 'ocv_y', 'adaptive_sigma']
    metric_labels = ['Val regret', 'FracImproving', 'OCV_Y', 'Sigma']
    metric_logy = [True, False, False, True]

    n_rows = len(ctrls)
    n_cols = len(metrics)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows), squeeze=False)

    fig.suptitle(
        f'DP (n=100, solid) vs Gurobi (n=10, dashed) — first {N_EPOCHS} epochs, target≈{args.target}\n'
        'Both solvers exact for knapsack; difference is n_samples only',
        fontsize=12, fontweight='bold',
    )

    epochs = np.arange(1, N_EPOCHS + 1)

    for row, ctrl in enumerate(ctrls):
        for col, (metric, mlabel, logy) in enumerate(zip(metrics, metric_labels, metric_logy)):
            ax = axes[row][col]
            ax.set_title(f'{ctrl_labels[ctrl]} — {mlabel}', fontsize=10)

            for lr in lrs:
                color = LR_COLORS[lr]

                dp_data  = load_run('dp',  ctrl, lr, target=args.target)
                grb_data = load_run('grb', ctrl, lr, target=args.target)

                if dp_data is not None:
                    y = dp_data[metric][:N_EPOCHS]
                    label = f'DP lr={lr} t={dp_data["target"]:.3f}'
                    ax.plot(epochs[:len(y)], y, color=color, ls='-', lw=1.8,
                            label=label, alpha=0.9)

                if grb_data is not None:
                    y = grb_data[metric][:N_EPOCHS]
                    label = f'Grb lr={lr} t={grb_data["target"]:.3f}'
                    ax.plot(epochs[:len(y)], y, color=color, ls='--', lw=1.8,
                            label=label, alpha=0.9)

            if logy:
                ax.set_yscale('log')
            ax.set_xlabel('Epoch', fontsize=10)
            ax.set_ylabel(mlabel, fontsize=10)
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.3, which='both')

    plt.tight_layout()
    out = args.out or os.path.join(BASE, 'fig_kn_dp_vs_grb_trajectories.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    print(f'Saved → {out}')
    plt.close()


if __name__ == '__main__':
    main()
