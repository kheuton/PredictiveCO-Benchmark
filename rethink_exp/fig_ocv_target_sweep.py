"""
Figure: Proportional adaptive sigma — full OCV_Y target sweep.

3-panel figure analogous to fig_rcr_target_sweep.py, but for OCV_Y-controlled runs:

  Panel 1 — OCV_Y target vs final val regret (log-y, stuck runs marked ×)
  Panel 2 — OCV_Y target vs final sigma (log-y)
  Panel 3 — OCV_Y target vs achieved mean OCV_Y (controller tracking accuracy)

Also adds a 4th panel showing FracImproving over training for the best poly run.

Usage:
    python rethink_exp/fig_ocv_target_sweep.py [--prefix ocv_run1]
"""

import argparse
import glob
import os

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# ---- Args ----
p = argparse.ArgumentParser()
p.add_argument('--prefix', type=str, default='ocv_run1')
p.add_argument('--problem', type=str, default='cubic')
p.add_argument('--prob_version', type=str, default='gen')
p.add_argument('--stuck_thresh', type=float, default=0.40,
               help='val_regret >= this treated as stuck/failed')
args = p.parse_args()

adap_dir = f'saved_records/{args.problem}-{args.prob_version}/perturb_adaptive_sweep/{args.prefix}'

# ---- Colours ----
C_DENSE = '#2166ac'
C_POLY  = '#d6604d'
STUCK   = args.stuck_thresh


# ---- Load ----

def load_prop(model):
    rows = []
    for f in sorted(glob.glob(f'{adap_dir}/prop_*_{model}.npz')):
        d = np.load(f, allow_pickle=True)
        name = os.path.basename(f).replace('.npz', '')
        # filename: prop_tX.XXX_sY_<model>
        target = float(name.split('_')[1][1:])
        rows.append(dict(
            target=target,
            val=float(d['val_regret'][-20:].mean()),
            best_val=float(d['val_regret'].min()),
            mean_ocv=float(d['ocv_y'].mean()),
            final_ocv=float(d['ocv_y'][-1]),
            final_sigma=float(d['adaptive_sigma'][-1]),
            frac_traj=d['frac_improving'],
            val_traj=d['val_regret'],
            ocv_traj=d['ocv_y'],
            sigma_traj=d['adaptive_sigma'],
        ))
    rows.sort(key=lambda r: r['target'])
    return rows


dense_rows = load_prop('dense')
poly_rows  = load_prop('poly')

if not dense_rows and not poly_rows:
    raise FileNotFoundError(f'No prop_* files found in {adap_dir}')

all_targets = sorted(set(r['target'] for r in dense_rows + poly_rows))
print(f'Loaded {len(dense_rows)} dense + {len(poly_rows)} poly runs '
      f'({len(all_targets)} unique targets)')


# ---- Figure: 1 row × 4 panels ----
fig, axes = plt.subplots(1, 4, figsize=(20, 5))
fig.suptitle(
    f'Proportional adaptive sigma — OCV_Y target sweep ({args.problem}, {args.prefix})',
    fontsize=13, fontweight='bold',
)
ax_val, ax_sigma, ax_track, ax_frac = axes


# ============================================================
# Panel 1: OCV_Y target vs val regret
# ============================================================
for rows, color, label in [
    (dense_rows, C_DENSE, 'dense'),
    (poly_rows,  C_POLY,  'poly'),
]:
    targets = np.array([r['target']   for r in rows])
    vals    = np.array([r['best_val'] for r in rows])
    stuck   = vals >= STUCK

    if (~stuck).any():
        ax_val.plot(targets[~stuck], vals[~stuck], 'o-', color=color,
                    lw=1.8, ms=6, label=label)
    if stuck.any():
        ax_val.scatter(targets[stuck], vals[stuck],
                       marker='x', color=color, s=80, lw=2, zorder=5,
                       label=f'{label} (stuck/failed)')

ax_val.set_xscale('log')
ax_val.set_yscale('log')
ax_val.set_xlabel('OCV_Y target', fontsize=11)
ax_val.set_ylabel('Best val regret (lower = better)', fontsize=11)
ax_val.set_title('Val regret vs OCV_Y target', fontsize=11)
ax_val.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.3f'))
ax_val.legend(fontsize=9)
ax_val.grid(True, alpha=0.3, which='both')


# ============================================================
# Panel 2: OCV_Y target vs final sigma
# ============================================================
for rows, color, label in [
    (dense_rows, C_DENSE, 'dense'),
    (poly_rows,  C_POLY,  'poly'),
]:
    targets = np.array([r['target']      for r in rows])
    sigmas  = np.array([r['final_sigma'] for r in rows])
    vals    = np.array([r['best_val']    for r in rows])
    stuck   = vals >= STUCK

    if (~stuck).any():
        ax_sigma.semilogy(targets[~stuck], sigmas[~stuck], 'o-',
                          color=color, lw=1.8, ms=6, label=label)
    if stuck.any():
        ax_sigma.scatter(targets[stuck], sigmas[stuck],
                         marker='x', color=color, s=80, lw=2, zorder=5)

ax_sigma.set_xscale('log')
ax_sigma.set_xlabel('OCV_Y target', fontsize=11)
ax_sigma.set_ylabel('Final sigma (log scale)', fontsize=11)
ax_sigma.set_title('Converged sigma vs OCV_Y target', fontsize=11)
ax_sigma.legend(fontsize=9)
ax_sigma.grid(True, alpha=0.3, which='both')


# ============================================================
# Panel 3: Controller tracking accuracy (target vs achieved OCV_Y)
# ============================================================
for rows, color, label in [
    (dense_rows, C_DENSE, 'dense'),
    (poly_rows,  C_POLY,  'poly'),
]:
    targets  = np.array([r['target']   for r in rows])
    achieved = np.array([r['mean_ocv'] for r in rows])
    ax_track.loglog(targets, achieved, 'o-', color=color, lw=1.8, ms=5, label=label)

# Perfect tracking diagonal
lo = min(all_targets); hi = max(all_targets)
ax_track.loglog([lo, hi], [lo, hi], 'k--', lw=1, alpha=0.4, label='perfect tracking')
ax_track.set_xlabel('OCV_Y target', fontsize=11)
ax_track.set_ylabel('Mean achieved OCV_Y', fontsize=11)
ax_track.set_title('Controller tracking accuracy', fontsize=11)
ax_track.legend(fontsize=9)
ax_track.grid(True, alpha=0.3, which='both')


# ============================================================
# Panel 4: FracImproving over training for best dense run
# ============================================================
# Find best dense run (lowest best_val)
best_dense = min(dense_rows, key=lambda r: r['best_val'])
epochs = np.arange(1, len(best_dense['frac_traj']) + 1)

ax_frac.plot(epochs, best_dense['frac_traj'], color=C_DENSE, lw=1.5,
             label=f"dense t={best_dense['target']:.3f}")
ax_frac.set_xlabel('Epoch', fontsize=11)
ax_frac.set_ylabel('FracImproving', fontsize=11)
ax_frac.set_title(
    f"FracImproving over training\n(best dense: t={best_dense['target']:.3f}, "
    f"val={best_dense['best_val']:.4f})",
    fontsize=10,
)
ax_frac.set_ylim(0, 1)
ax_frac.axhline(0, color='k', lw=0.8, ls='--', alpha=0.4)
ax_frac.legend(fontsize=9)
ax_frac.grid(True, alpha=0.3)

# Also overlay val regret on a twin axis
ax_frac2 = ax_frac.twinx()
ax_frac2.plot(epochs, best_dense['val_traj'], color='grey', lw=1.2,
              ls='--', alpha=0.7, label='val regret')
ax_frac2.set_ylabel('Val regret', fontsize=10, color='grey')
ax_frac2.tick_params(axis='y', colors='grey')
ax_frac2.set_yscale('log')


plt.tight_layout()
out = os.path.join(adap_dir, 'fig_ocv_target_sweep.png')
plt.savefig(out, dpi=150, bbox_inches='tight')
print(f'Saved → {out}')
plt.close()
