"""
Trace plots for the best Hamming-decay runs (per LR).

Panels: val_regret | sigma | Hamming rate | FracImproving
Overlays constant-target best run (solid) vs decay best run (dashed) for each LR.
Vertical dashed line at warmup_epochs=100 (where decay begins).

Usage:
    conda run -n pco_bench_rhel7 python rethink_exp/fig_kn_hamming_decay_traces.py
    conda run -n pco_bench_rhel7 python rethink_exp/fig_kn_hamming_decay_traces.py --out my.png
"""

import argparse
import glob
import os

import matplotlib.pyplot as plt
import numpy as np

ADAPTIVE_BASE = 'saved_records/knapsack-gen/perturb_adaptive_sweep'
MSE_BASELINE  = 0.0640

LR_COLORS = {'5e-2': '#e66101', '1e-2': '#5e3c99', '5e-3': '#1a9641'}
WARMUP_EPOCHS = 100


def load_best(lr, decay=False):
    """Return (npz_dict, target) for the best-val run at this LR."""
    if decay:
        prefix = f'kn_bench_hamming_decay_lr{lr}'
        stem   = 'hamd_t'
    else:
        prefix = f'kn_bench_hamming_lr{lr}'
        stem   = 'ham_t'
    pat = os.path.join(ADAPTIVE_BASE, prefix, f'{stem}*_dense.npz')
    files = sorted(glob.glob(pat))
    if not files:
        return None, None
    best_f, best_val, best_t = None, float('inf'), None
    for f in files:
        d = np.load(f, allow_pickle=True)
        v = float(d['val_regret'].min())
        t = float(os.path.basename(f).replace('.npz','').split('_')[1][1:])
        if v < best_val:
            best_val, best_f, best_t = v, f, t
    d = np.load(best_f, allow_pickle=True)
    return dict(d), best_t


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', default=None)
    args = parser.parse_args()

    lrs = ['5e-2', '1e-2', '5e-3']

    fig, axes = plt.subplots(1, 4, figsize=(20, 4))
    ax_val, ax_sig, ax_ham, ax_frac = axes

    fig.suptitle(
        'Best Hamming run traces per LR — solid=constant target, dashed=decay (100ep warmup → 200ep linear decay)\n'
        'Knapsack benchmark scale, DP n=100, 300ep',
        fontsize=11, fontweight='bold',
    )

    for lr in lrs:
        color = LR_COLORS[lr]
        for decay, ls in [(False, '-'), (True, '--')]:
            d, t = load_best(lr, decay=decay)
            if d is None:
                continue
            tag = 'decay' if decay else 'const'
            label = f'lr={lr} {tag} t={t:.3f} val={d["val_regret"].min():.4f}'
            epochs = np.arange(1, len(d['val_regret']) + 1)

            ax_val.plot(epochs, d['val_regret'],    color=color, ls=ls, lw=1.5, label=label)
            ax_sig.plot(epochs, d['adaptive_sigma'], color=color, ls=ls, lw=1.5, label=label)
            ax_ham.plot(epochs, d['hamming'],        color=color, ls=ls, lw=1.5, label=label)
            ax_frac.plot(epochs, d['frac_improving'], color=color, ls=ls, lw=1.5, label=label)
            if decay:
                ax_ham.axhline(t, color=color, lw=0.7, ls=':', alpha=0.4)

    # MSE baseline
    ax_val.axhline(MSE_BASELINE, color='gray', lw=1.5, ls=':', label=f'MSE ({MSE_BASELINE:.4f})')

    # Warmup boundary
    for ax in axes:
        ax.axvline(WARMUP_EPOCHS, color='k', lw=0.8, ls='--', alpha=0.3, label='decay start')

    # Formatting
    titles = ['Val regret', 'Sigma (adaptive)', 'Hamming rate', 'FracImproving']
    ylabels = ['Val regret (norm.)', 'Sigma', 'Hamming rate', 'FracImproving']
    for ax, title, ylabel in zip(axes, titles, ylabels):
        ax.set_xlabel('Epoch', fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    ax_val.set_yscale('log')
    ax_sig.set_yscale('log')
    ax_frac.set_ylim(-0.05, 1.05)
    ax_ham.set_ylim(-0.02, 0.45)

    plt.tight_layout()
    out = args.out or os.path.join(ADAPTIVE_BASE, 'fig_kn_hamming_decay_traces.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    print(f'Saved → {out}')
    plt.close()


if __name__ == '__main__':
    main()
