"""
Figure: Proportional adaptive sigma — full RCR target sweep.

Shows val regret vs RCR target for dense and poly models across all
completed proportional runs, with the fixed-sigma baseline for reference.

Saves to saved_records/cubic-gen/perturb_adaptive_sweep/adaptive_run1/fig_rcr_target_sweep.png
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import glob, os

# ---- Load data ----
adap_dir = 'saved_records/cubic-gen/perturb_adaptive_sweep/adaptive_run1'
diag_dir = 'saved_records/cubic-gen/perturb_sigma_sweep/diag_run1'

def load_prop(model):
    rows = []
    for f in sorted(glob.glob(f'{adap_dir}/prop_*_{model}.npz')):
        d = np.load(f, allow_pickle=True)
        name = os.path.basename(f).replace('.npz', '')
        target = float(name.split('_')[1][1:])   # prop_tX.XXX_sY → X.XXX
        rows.append(dict(
            target=target,
            val=float(d['val_regret'][-20:].mean()),
            mean_rcr=float(d['rank_change_rate'].mean()),
            final_sigma=float(d['adaptive_sigma'][-1]),
            sigma_traj=d['adaptive_sigma'],
            val_traj=d['val_regret'],
        ))
    rows.sort(key=lambda r: r['target'])
    return rows

dense_rows = load_prop('dense')
poly_rows  = load_prop('poly')

# Fixed-sigma bests
def fixed_best(model):
    vals = [np.load(f, allow_pickle=True)['val_regret'][-20:].mean()
            for f in glob.glob(f'{diag_dir}/*_{model}.npz')]
    return min(vals)

fixed_dense = fixed_best('dense')
fixed_poly  = fixed_best('poly')

# ---- Colours ----
C_DENSE = '#2166ac'
C_POLY  = '#d6604d'
STUCK   = 0.40   # val_regret threshold for "stuck"

# ---- Figure: 2 rows × 2 cols ----
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle('Proportional adaptive sigma — RCR target sweep (cubic)',
             fontsize=13, fontweight='bold')
ax_val, ax_sigma, ax_rcr = axes

# ============================================================
# Panel 1: RCR target vs final val regret
# ============================================================
for rows, color, label, fixed in [
    (dense_rows, C_DENSE, 'dense', fixed_dense),
    (poly_rows,  C_POLY,  'poly',  fixed_poly),
]:
    targets = np.array([r['target'] for r in rows])
    vals    = np.array([r['val']    for r in rows])
    stuck   = vals >= STUCK

    # Converging runs — solid line + markers
    ax_val.plot(targets[~stuck], vals[~stuck], 'o-', color=color,
                lw=1.8, ms=6, label=label)
    # Stuck runs — open markers
    if stuck.any():
        ax_val.scatter(targets[stuck], vals[stuck],
                       marker='x', color=color, s=80, lw=2, zorder=5,
                       label=f'{label} (stuck)')
    # Baseline
    ax_val.axhline(fixed, color=color, ls=':', lw=1.5,
                   label=f'fixed-σ best {label} ({fixed:.3f})')

ax_val.set_xlabel('RCR target', fontsize=11)
ax_val.set_ylabel('Final val regret (lower = better)', fontsize=11)
ax_val.set_title('Val regret vs RCR target', fontsize=11)
ax_val.set_yscale('log')
ax_val.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.3f'))
ax_val.legend(fontsize=8, loc='upper right')
ax_val.grid(True, alpha=0.3)

# ============================================================
# Panel 2: RCR target vs final sigma (log scale)
# ============================================================
for rows, color, label in [(dense_rows, C_DENSE, 'dense'),
                            (poly_rows,  C_POLY,  'poly')]:
    targets = [r['target']      for r in rows]
    sigmas  = [r['final_sigma'] for r in rows]
    vals    = [r['val']         for r in rows]
    stuck   = [v >= STUCK       for v in vals]

    conv_t = [t for t, s in zip(targets, stuck) if not s]
    conv_s = [sig for sig, s in zip(sigmas, stuck) if not s]
    stk_t  = [t for t, s in zip(targets, stuck) if s]
    stk_s  = [sig for sig, s in zip(sigmas, stuck) if s]

    ax_sigma.semilogy(conv_t, conv_s, 'o-', color=color, lw=1.8, ms=6, label=label)
    if stk_t:
        ax_sigma.scatter(stk_t, stk_s, marker='x', color=color, s=80, lw=2, zorder=5)

ax_sigma.set_xlabel('RCR target', fontsize=11)
ax_sigma.set_ylabel('Final sigma (log scale)', fontsize=11)
ax_sigma.set_title('Converged sigma vs RCR target', fontsize=11)
ax_sigma.legend(fontsize=9)
ax_sigma.grid(True, alpha=0.3)

# ============================================================
# Panel 3: RCR target vs achieved mean RCR (controller accuracy)
# ============================================================
for rows, color, label in [(dense_rows, C_DENSE, 'dense'),
                            (poly_rows,  C_POLY,  'poly')]:
    targets  = [r['target']   for r in rows]
    mean_rcr = [r['mean_rcr'] for r in rows]
    ax_rcr.plot(targets, mean_rcr, 'o-', color=color, lw=1.8, ms=5, label=label)

ax_rcr.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.4, label='perfect tracking')
ax_rcr.set_xlabel('RCR target', fontsize=11)
ax_rcr.set_ylabel('Mean achieved RCR', fontsize=11)
ax_rcr.set_title('Controller tracking accuracy', fontsize=11)
ax_rcr.legend(fontsize=9)
ax_rcr.grid(True, alpha=0.3)
ax_rcr.set_xlim(0, 1); ax_rcr.set_ylim(0, 1)

plt.tight_layout()
out = f'{adap_dir}/fig_rcr_target_sweep.png'
plt.savefig(out, dpi=150, bbox_inches='tight')
print(f'Saved → {out}')
plt.close()
