"""
Figure: Knapsack benchmark-scale Hamming-rate controller sweep.

Grid: 5 targets × 3 LRs = 15 runs (DP, n=100, 300ep, per-instance).
Prefixes: kn_bench_hamming_lr{lr}/  Files: ham_t{target}_s1_dense.npz

Layout: 2 rows × 3 panels
  Row 0 — sweep summary (across targets)
    Panel 0: Hamming target vs best val regret  (MSE baseline as hline)
    Panel 1: Hamming target vs final sigma      (log-scale)
    Panel 2: Hamming target vs test regret      (MSE baseline as hline)

  Row 1 — best-run trajectories per LR
    Panel 3: val_regret over epochs
    Panel 4: achieved Hamming rate over epochs  (dotted hline at each target)
    Panel 5: FracImproving over epochs

Colors by LR.  X marks = stuck (val >= STUCK_THRESH).

Usage:
    conda run -n pco_bench_rhel7 python rethink_exp/fig_kn_bench_hamming.py
    conda run -n pco_bench_rhel7 python rethink_exp/fig_kn_bench_hamming.py --out my_fig.png
"""

import argparse
import glob
import os

import matplotlib.pyplot as plt
import numpy as np

ADAPTIVE_BASE = 'saved_records/knapsack-gen/perturb_adaptive_sweep'
MSE_BASE      = 'saved_records/knapsack-gen/mse'

LR_COLORS  = {'5e-2': '#e66101', '1e-2': '#5e3c99', '5e-3': '#1a9641'}
STUCK_THRESH = 0.40


# ---- Loaders ----

def load_bench_mse():
    best, best_lr = None, None
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
                best, best_lr = regret, lr
        except Exception:
            pass
    return best, best_lr


def load_hamming(lr, decay=False):
    """Load all target runs for a given LR. Returns list of row dicts.

    decay=False → constant-target runs (ham_t* stem, kn_bench_hamming_lr*)
    decay=True  → decay runs (hamd_t* stem, kn_bench_hamming_decay_lr*)
    """
    if decay:
        prefix = f'kn_bench_hamming_decay_lr{lr}'
        stem   = 'hamd_t'
    else:
        prefix = f'kn_bench_hamming_lr{lr}'
        stem   = 'ham_t'
    pat = os.path.join(ADAPTIVE_BASE, prefix, f'{stem}*_dense.npz')
    rows = []
    for f in sorted(glob.glob(pat)):
        d = np.load(f, allow_pickle=True)
        name = os.path.basename(f).replace('.npz', '')
        target = float(name.split('_')[1][1:])
        rows.append(dict(
            target       = target,
            best_val     = float(d['val_regret'].min()),
            test_regret  = float(d['test_regret']) if 'test_regret' in d else float('nan'),
            final_sigma  = float(d['adaptive_sigma'][-1]),
            val_traj     = d['val_regret'],
            hamming_traj = d['hamming'],
            frac_traj    = d['frac_improving'],
            ocv_y_traj   = d['ocv_y'],
            n_epochs     = len(d['val_regret']),
        ))
    rows.sort(key=lambda r: r['target'])
    return rows


# ---- Plotting ----

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', default=None)
    parser.add_argument('--mse_baseline', type=float, default=None)
    args = parser.parse_args()

    lrs = ['5e-2', '1e-2', '5e-3']

    # MSE baseline
    mse_baseline = args.mse_baseline
    if mse_baseline is None:
        mse_val, mse_lr = load_bench_mse()
        if mse_val is not None:
            mse_baseline = mse_val
            print(f'MSE benchmark baseline: {mse_val:.4f} (lr={mse_lr})')
        else:
            mse_baseline = 0.064
            print('MSE baseline not available; using small-scale fallback (0.064)')

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(
        'Knapsack benchmark — per-instance Hamming-rate controller\n'
        'Solid = constant target | Dashed = decay (100ep warmup → 200ep linear decay to 0)\n'
        'DP, n=100, 300ep, 320+80 train/val, 200 test',
        fontsize=12, fontweight='bold',
    )

    ax_val, ax_sigma, ax_test = axes[0]
    ax_traj_val, ax_traj_ham, ax_traj_frac = axes[1]

    has_data = False
    for decay, ls in [(False, '-'), (True, '--')]:
        label_suffix = ' (decay)' if decay else ''
        for lr in lrs:
            rows = load_hamming(lr, decay=decay)
            if not rows:
                continue
            has_data = True
            color = LR_COLORS[lr]

            targets = np.array([r['target']      for r in rows])
            vals    = np.array([r['best_val']    for r in rows])
            tests   = np.array([r['test_regret'] for r in rows])
            sigmas  = np.array([r['final_sigma'] for r in rows])
            stuck   = vals >= STUCK_THRESH

            kw = dict(color=color, lw=1.8, ms=7, ls=ls)
            if (~stuck).any():
                ax_val.plot(targets[~stuck], vals[~stuck], 'o',
                            label=f'lr={lr}{label_suffix}', **kw)
                ax_sigma.semilogy(targets[~stuck], sigmas[~stuck], 'o',
                                  label=f'lr={lr}{label_suffix}', **kw)
                test_valid = ~stuck & ~np.isnan(tests)
                if test_valid.any():
                    ax_test.plot(targets[test_valid], tests[test_valid], 'o',
                                 label=f'lr={lr}{label_suffix}', **kw)
            if stuck.any():
                for ax in [ax_val, ax_test]:
                    ax.scatter(targets[stuck], vals[stuck],
                               marker='x', color=color, s=80, lw=2, zorder=5)

            # Row 1: best-run trajectories (only best LR per decay mode shown)
            best = min(rows, key=lambda r: r['best_val'])
            epochs = np.arange(1, best['n_epochs'] + 1)
            traj_label = f"lr={lr}{label_suffix} t={best['target']:.3f} val={best['best_val']:.4f}"
            ax_traj_val.plot(epochs, best['val_traj'],     color=color, ls=ls, lw=1.5, label=traj_label)
            ax_traj_ham.plot(epochs, best['hamming_traj'], color=color, ls=ls, lw=1.5, label=traj_label)
            ax_traj_ham.axhline(best['target'], color=color, lw=0.8, ls=':', alpha=0.5)
            ax_traj_frac.plot(epochs, best['frac_traj'],   color=color, ls=ls, lw=1.5, label=traj_label)

    # MSE hlines
    for ax in [ax_val, ax_test]:
        ax.axhline(mse_baseline, color='gray', lw=2.0, ls=':',
                   label=f'MSE dense ({mse_baseline:.4f})', zorder=3)

    if not has_data:
        for ax in axes.flat:
            ax.text(0.5, 0.5, 'No results yet', transform=ax.transAxes,
                    ha='center', va='center', fontsize=13, color='gray')

    # Axis labels / formatting — row 0
    for ax, xlabel, ylabel, title in [
        (ax_val,   'Hamming target', 'Best val regret (norm.)', 'Val regret vs Hamming target'),
        (ax_sigma, 'Hamming target', 'Final sigma',             'Final sigma vs Hamming target'),
        (ax_test,  'Hamming target', 'Test regret (norm.)',     'Test regret vs Hamming target'),
    ]:
        ax.set_xlabel(xlabel, fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(title, fontsize=11)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    ax_val.set_yscale('log')
    ax_test.set_yscale('log')

    # Axis labels / formatting — row 1
    ax_traj_val.set_xlabel('Epoch'); ax_traj_val.set_ylabel('Val regret')
    ax_traj_val.set_title('Val regret trajectory (best per LR)')
    ax_traj_val.set_yscale('log')
    ax_traj_val.axhline(mse_baseline, color='gray', lw=1.5, ls=':', label=f'MSE ({mse_baseline:.4f})')

    ax_traj_ham.set_xlabel('Epoch'); ax_traj_ham.set_ylabel('Hamming rate')
    ax_traj_ham.set_title('Achieved Hamming rate (dotted = target)')
    ax_traj_ham.set_ylim(-0.02, 0.55)

    ax_traj_frac.set_xlabel('Epoch'); ax_traj_frac.set_ylabel('FracImproving')
    ax_traj_frac.set_title('FracImproving trajectory (best per LR)')
    ax_traj_frac.set_ylim(-0.05, 1.05)
    ax_traj_frac.axhline(0, color='k', lw=0.8, ls='--', alpha=0.4)

    for ax in [ax_traj_val, ax_traj_ham, ax_traj_frac]:
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = args.out or os.path.join(ADAPTIVE_BASE, 'fig_kn_bench_hamming.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    print(f'Saved → {out}')
    plt.close()

    # ---- Summary table ----
    for decay, tag in [(False, 'constant'), (True, 'decay')]:
        print(f'\n--- Hamming {tag} results (benchmark scale) ---')
        for lr in lrs:
            rows = load_hamming(lr, decay=decay)
            if not rows:
                print(f'  lr={lr}: no results yet')
                continue
            best = min(rows, key=lambda r: r['best_val'])
            test_str = f'  test={best["test_regret"]:.4f}' if not np.isnan(best['test_regret']) else ''
            print(f'  lr={lr}: val={best["best_val"]:.4f}{test_str} at t={best["target"]:.3f} '
                  f'({best["n_epochs"]} epochs)')
    print(f'\n  MSE baseline: {mse_baseline:.4f}')


if __name__ == '__main__':
    main()
