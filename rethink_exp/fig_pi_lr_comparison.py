"""
Figure: Per-instance adaptive sigma — 3 LRs compared.

2-row (dense / poly) × 3-panel layout, mirroring fig_ocv_lr_comparison.py:
  Panel 1 — OCV_Y target vs best val regret (all 3 LRs overlaid)
  Panel 2 — OCV_Y target vs final mean sigma
  Panel 3 — FracImproving over training for the best run per LR

Usage:
    python rethink_exp/fig_pi_lr_comparison.py
"""

import glob
import os

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# ---- Config ----
PROBLEM  = 'cubic'
PROB_VER = 'gen'
BASE_DIR = f'saved_records/{PROBLEM}-{PROB_VER}/perturb_adaptive_sweep'

LR_CONFIGS = [
    ('pi_lr5e-2', '5e-2', '#e66101'),
    ('pi_lr1e-2', '1e-2', '#5e3c99'),
    ('pi_lr5e-3', '5e-3', '#1a9641'),
]

STUCK_THRESH = 0.40
MODELS = ['dense', 'poly']


# ---- Load ----

def load_pi(prefix, model):
    rows = []
    pat = os.path.join(BASE_DIR, prefix, f'pi_t*_{model}.npz')
    for f in sorted(glob.glob(pat)):
        d = np.load(f, allow_pickle=True)
        name = os.path.basename(f).replace('.npz', '')
        # filename: pi_t{target}_s{sigma_init}_{model}
        target = float(name.split('_t')[1].split('_')[0])
        rows.append(dict(
            target       = target,
            best_val     = float(d['val_regret'].min()),
            final_sigma  = float(d['adaptive_sigma'][-1]),
            frac_traj    = d['frac_improving'],
            val_traj     = d['val_regret'],
        ))
    rows.sort(key=lambda r: r['target'])
    return rows


# ---- Figure ----

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle(
    'Per-instance adaptive sigma — LR comparison (cubic, proportional mode)',
    fontsize=13, fontweight='bold',
)

for row_idx, model in enumerate(MODELS):
    ax_val, ax_sigma, ax_frac = axes[row_idx]
    model_label = model.capitalize()

    # ---- Panels 1 & 2: sweep summary ----
    for prefix, lr_label, color in LR_CONFIGS:
        rows = load_pi(prefix, model)
        if not rows:
            continue

        targets = np.array([r['target']     for r in rows])
        vals    = np.array([r['best_val']   for r in rows])
        sigmas  = np.array([r['final_sigma'] for r in rows])
        stuck   = vals >= STUCK_THRESH

        if (~stuck).any():
            ax_val.plot(targets[~stuck], vals[~stuck], 'o-',
                        color=color, lw=1.8, ms=6, label=f'lr={lr_label}')
        if stuck.any():
            ax_val.scatter(targets[stuck], vals[stuck],
                           marker='x', color=color, s=80, lw=2, zorder=5)

        if (~stuck).any():
            ax_sigma.semilogy(targets[~stuck], sigmas[~stuck], 'o-',
                              color=color, lw=1.8, ms=6, label=f'lr={lr_label}')
        if stuck.any():
            ax_sigma.scatter(targets[stuck], sigmas[stuck],
                             marker='x', color=color, s=80, lw=2, zorder=5)

    ax_val.set_xscale('log')
    ax_val.set_yscale('log')
    ax_val.set_xlabel('OCV_Y target', fontsize=11)
    ax_val.set_ylabel('Best val regret', fontsize=11)
    ax_val.set_title(f'{model_label} — val regret vs OCV_Y target', fontsize=11)
    ax_val.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.3f'))
    ax_val.legend(fontsize=9)
    ax_val.grid(True, alpha=0.3, which='both')

    ax_sigma.set_xscale('log')
    ax_sigma.set_xlabel('OCV_Y target', fontsize=11)
    ax_sigma.set_ylabel('Final mean sigma', fontsize=11)
    ax_sigma.set_title(f'{model_label} — final sigma vs OCV_Y target', fontsize=11)
    ax_sigma.legend(fontsize=9)
    ax_sigma.grid(True, alpha=0.3, which='both')

    # ---- Panel 3: FracImproving for best run per LR ----
    for prefix, lr_label, color in LR_CONFIGS:
        rows = load_pi(prefix, model)
        if not rows:
            continue
        best = min(rows, key=lambda r: r['best_val'])
        epochs = np.arange(1, len(best['frac_traj']) + 1)
        ax_frac.plot(epochs, best['frac_traj'], color=color, lw=1.5, alpha=0.85,
                     label=f"lr={lr_label} t={best['target']:.3f} val={best['best_val']:.4f}")

    ax_frac.set_xlabel('Epoch', fontsize=11)
    ax_frac.set_ylabel('FracImproving', fontsize=11)
    ax_frac.set_title(f'{model_label} — FracImproving (best run per LR)', fontsize=11)
    ax_frac.set_ylim(0, 1)
    ax_frac.axhline(0, color='k', lw=0.8, ls='--', alpha=0.4)
    ax_frac.legend(fontsize=8)
    ax_frac.grid(True, alpha=0.3)

plt.tight_layout()
out = os.path.join(BASE_DIR, 'fig_pi_lr_comparison.png')
plt.savefig(out, dpi=150, bbox_inches='tight')
print(f'Saved → {out}')
plt.close()
