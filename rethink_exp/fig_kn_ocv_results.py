"""
Figure: Knapsack OCV_Y adaptive sigma — LR comparison + MSE baseline.

Covers two waves of results:
  Wave 1 (--wave 1): global controller only, 150 epochs
    prefixes: kn_ocv_lr{5e-2,1e-2,5e-3}
  Wave 2 (--wave 2, default): global + per-instance, 300 epochs, poly deg=4 fixed
    prefixes: kn_w2_global_lr{5e-2,1e-2,5e-3}  (solid lines)
              kn_w2_pi_lr{5e-2,1e-2,5e-3}       (dashed lines)

Layout: 2 rows (dense / poly) × 3 panels
  Panel 1 — OCV_Y target vs best val regret
            (global=solid, per-instance=dashed, MSE best=gray hline)
  Panel 2 — OCV_Y target vs final sigma
  Panel 3 — FracImproving for the best run per (LR, controller)

MSE baselines from:
  saved_records/knapsack-gen/mse/kn_mse_{dense,poly}_lr{5e-2,1e-2,5e-3}/log.txt

Usage:
    python rethink_exp/fig_kn_ocv_results.py            # wave 2 (default)
    python rethink_exp/fig_kn_ocv_results.py --wave 1   # wave 1
    python rethink_exp/fig_kn_ocv_results.py --out fig.png
"""

import argparse
import glob
import os

import matplotlib.pyplot as plt
import numpy as np

ADAPTIVE_BASE = 'saved_records/knapsack-gen/perturb_adaptive_sweep'
MSE_BASE      = 'saved_records/knapsack-gen/mse'

STUCK_THRESH = 0.40
MODELS = ['dense', 'poly']

LR_COLORS = {'5e-2': '#e66101', '1e-2': '#5e3c99', '5e-3': '#1a9641'}

# Wave configs: list of (prefix, lr_label, controller_label, linestyle)
WAVE_CONFIGS = {
    1: [
        ('kn_ocv_lr5e-2', '5e-2', 'global', '-',  'prop'),
        ('kn_ocv_lr1e-2', '1e-2', 'global', '-',  'prop'),
        ('kn_ocv_lr5e-3', '5e-3', 'global', '-',  'prop'),
    ],
    2: [
        ('kn_w2_global_lr5e-2', '5e-2', 'global', '-',  'prop'),
        ('kn_w2_global_lr1e-2', '1e-2', 'global', '-',  'prop'),
        ('kn_w2_global_lr5e-3', '5e-3', 'global', '-',  'prop'),
        ('kn_w2_pi_lr5e-2',     '5e-2', 'pi',     '--', 'pi'),
        ('kn_w2_pi_lr1e-2',     '1e-2', 'pi',     '--', 'pi'),
        ('kn_w2_pi_lr5e-3',     '5e-3', 'pi',     '--', 'pi'),
    ],
}


# ---- Loaders ----

def load_adaptive(prefix, model, stem='prop'):
    """Load all runs for a given prefix/model. stem='prop' or 'pi'."""
    rows = []
    pat = os.path.join(ADAPTIVE_BASE, prefix, f'{stem}_*_{model}.npz')
    for f in sorted(glob.glob(pat)):
        d = np.load(f, allow_pickle=True)
        name = os.path.basename(f).replace('.npz', '')
        # filename: prop_tX.XXX_s1_dense  or  pi_tX.XXX_s1_dense
        target = float(name.split('_')[1][1:])
        rows.append(dict(
            target      = target,
            best_val    = float(d['val_regret'].min()),
            final_sigma = float(d['adaptive_sigma'][-1]),
            frac_traj   = d['frac_improving'],
            val_traj    = d['val_regret'],
        ))
    rows.sort(key=lambda r: r['target'])
    return rows


def load_mse_per_lr(model):
    """Return {lr: normalised_test_regret} for each completed MSE run."""
    out = {}
    for lr in ['5e-2', '1e-2', '5e-3']:
        log_path = os.path.join(MSE_BASE, f'kn_mse_{model}_lr{lr}', 'log.txt')
        if not os.path.exists(log_path):
            continue
        try:
            with open(log_path) as f:
                lines = [l.strip() for l in f if l.strip()]
            avg_regret = float(lines[-1].split()[-3])
            opt_line = next(l for l in reversed(lines) if '[Optimal Obj]' in l)
            opt_obj = float(opt_line.split('[Optimal Obj]:')[1].split()[0])
            out[lr] = avg_regret / (opt_obj + 1e-8)
        except Exception:
            pass
    return out


# ---- Main ----

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--wave', type=int, default=2, choices=[1, 2])
    parser.add_argument('--out', type=str, default=None)
    args = parser.parse_args()

    configs = WAVE_CONFIGS[args.wave]
    epoch_label = '150' if args.wave == 1 else '300'
    out_path = args.out or os.path.join(
        ADAPTIVE_BASE, f'fig_kn_w{args.wave}_results.png'
    )

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(
        f'Knapsack — OCV_Y adaptive sigma, wave {args.wave} '
        f'(DP solver, 200 train, {epoch_label} epochs)',
        fontsize=13, fontweight='bold',
    )

    for row_idx, model in enumerate(MODELS):
        ax_val, ax_sigma, ax_frac = axes[row_idx]
        model_label = model.capitalize()
        has_data = False

        # ---- Panels 1 & 2 ----
        for prefix, lr_label, ctrl_label, ls, stem in configs:
            rows = load_adaptive(prefix, model, stem)
            if not rows:
                continue
            has_data = True
            color = LR_COLORS[lr_label]

            targets = np.array([r['target']      for r in rows])
            vals    = np.array([r['best_val']    for r in rows])
            sigmas  = np.array([r['final_sigma'] for r in rows])
            stuck   = vals >= STUCK_THRESH

            label = f'lr={lr_label} ({ctrl_label})'

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

        # ---- MSE baseline hline ----
        mse_per_lr = load_mse_per_lr(model)
        if mse_per_lr:
            best_mse = min(mse_per_lr.values())
            ax_val.axhline(best_mse, color='gray', lw=2.0, ls=':',
                           label=f'MSE best ({best_mse:.4f})', zorder=3)
        elif not has_data:
            ax_val.text(0.5, 0.5, 'No results yet', transform=ax_val.transAxes,
                        ha='center', va='center', fontsize=12, color='gray')

        ax_val.set_xscale('log')
        ax_val.set_yscale('log')
        ax_val.set_xlabel('OCV_Y target', fontsize=11)
        ax_val.set_ylabel('Best val regret', fontsize=11)
        ax_val.set_title(f'{model_label} — val regret vs OCV_Y target', fontsize=11)
        ax_val.legend(fontsize=7, ncol=2)
        ax_val.grid(True, alpha=0.3, which='both')

        ax_sigma.set_xscale('log')
        ax_sigma.set_xlabel('OCV_Y target', fontsize=11)
        ax_sigma.set_ylabel('Final sigma', fontsize=11)
        ax_sigma.set_title(f'{model_label} — final sigma vs OCV_Y target', fontsize=11)
        ax_sigma.legend(fontsize=7, ncol=2)
        ax_sigma.grid(True, alpha=0.3, which='both')

        # ---- Panel 3: FracImproving for best run per (LR, controller) ----
        for prefix, lr_label, ctrl_label, ls, stem in configs:
            rows = load_adaptive(prefix, model, stem)
            if not rows:
                continue
            best = min(rows, key=lambda r: r['best_val'])
            color = LR_COLORS[lr_label]
            epochs = np.arange(1, len(best['frac_traj']) + 1)
            ax_frac.plot(
                epochs, best['frac_traj'],
                color=color, ls=ls, lw=1.5, alpha=0.85,
                label=f"lr={lr_label} {ctrl_label} t={best['target']:.3f} "
                      f"val={best['best_val']:.4f}",
            )

        ax_frac.set_xlabel('Epoch', fontsize=11)
        ax_frac.set_ylabel('FracImproving', fontsize=11)
        ax_frac.set_title(f'{model_label} — FracImproving (best per LR+ctrl)', fontsize=11)
        ax_frac.set_ylim(-0.05, 1.05)
        ax_frac.axhline(0, color='k', lw=0.8, ls='--', alpha=0.4)
        ax_frac.legend(fontsize=7)
        ax_frac.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f'Saved → {out_path}')
    plt.close()

    # ---- Summary table ----
    print(f'\n--- Best val regret summary (wave {args.wave}) ---')
    for prefix, lr_label, ctrl_label, _, stem in configs:
        for model in MODELS:
            rows = load_adaptive(prefix, model, stem)
            if rows:
                best_row = min(rows, key=lambda r: r['best_val'])
                print(f'  {ctrl_label:10s} lr={lr_label} {model}: '
                      f'best={best_row["best_val"]:.4f} at t={best_row["target"]:.3f}')
    print()
    for model in MODELS:
        mse = load_mse_per_lr(model)
        if mse:
            best_lr = min(mse, key=mse.get)
            print(f'  MSE {model}: best={mse[best_lr]:.4f} at lr={best_lr}')


if __name__ == '__main__':
    main()
