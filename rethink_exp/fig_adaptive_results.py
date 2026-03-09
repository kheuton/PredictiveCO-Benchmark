"""
Figure: Adaptive sigma sweep results vs fixed-sigma baseline.

Four panels:
  A  Target RCR vs final val regret — proportional mode (dense + poly)
  B  Same for feedback mode
  C  Sigma trajectory for selected runs (stable vs blowup)
  D  Summary bar: best val regret per method

Saves to saved_records/cubic-gen/perturb_adaptive_sweep/adaptive_run1/fig_adaptive_results.png
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import glob, os

# ---- Load data ----
adap_dir = 'saved_records/cubic-gen/perturb_adaptive_sweep/adaptive_run1'
diag_dir = 'saved_records/cubic-gen/perturb_sigma_sweep/diag_run1'

def load_runs(pattern):
    rows = []
    for f in sorted(glob.glob(pattern)):
        d = np.load(f, allow_pickle=True)
        name = os.path.basename(f).replace('.npz', '')
        parts = name.split('_')
        mode  = 'prop' if name.startswith('prop') else 'fb'
        model = parts[-1]
        target  = float(parts[1][1:])
        s_init  = float(parts[2][1:])
        rows.append(dict(
            name=name, mode=mode, model=model, target=target, s_init=s_init,
            val=float(d['val_regret'][-20:].mean()),
            mean_rcr=float(d['rank_change_rate'].mean()),
            final_sigma=float(d['adaptive_sigma'][-1]),
            sigma_traj=d['adaptive_sigma'],
            val_traj=d['val_regret'],
            rcr_traj=d['rank_change_rate'],
            epochs=d['epoch'],
        ))
    return rows

adap = load_runs(f'{adap_dir}/*.npz')
prop_dense = [r for r in adap if r['mode']=='prop' and r['model']=='dense']
prop_poly  = [r for r in adap if r['mode']=='prop' and r['model']=='poly']
fb_dense   = [r for r in adap if r['mode']=='fb'   and r['model']=='dense']
fb_poly    = [r for r in adap if r['mode']=='fb'   and r['model']=='poly']

# Fixed-sigma baselines
def load_fixed(pattern):
    rows = []
    for f in sorted(glob.glob(pattern)):
        d = np.load(f, allow_pickle=True)
        name = os.path.basename(f).replace('.npz','')
        sigma = float(name.split('_')[1])
        model = name.split('_')[-1]
        rows.append(dict(sigma=sigma, model=model,
                         val=float(d['val_regret'][-20:].mean())))
    return rows

fixed = load_fixed(f'{diag_dir}/*.npz')
fixed_dense_best = min(r['val'] for r in fixed if r['model']=='dense')
fixed_poly_best  = min(r['val'] for r in fixed if r['model']=='poly')

# ---- Colour / marker helpers ----
model_color = {'dense': '#2166ac', 'poly': '#d6604d'}
model_marker = {'dense': 'o', 'poly': 's'}
s_init_style = {0.1: '-', 1.0: '--'}   # linestyle for feedback sigma_init

def outcome_alpha(val):
    if val < 0.02: return 1.0
    if val < 0.15: return 0.85
    if val < 0.40: return 0.5
    return 0.25

# ---- Figure ----
fig, axes = plt.subplots(2, 2, figsize=(13, 9))
fig.suptitle('Adaptive sigma controller — cubic benchmark\n'
             'proportional (top) vs feedback α=0.05 (bottom)',
             fontsize=12, fontweight='bold')
(ax_A, ax_B), (ax_C, ax_D) = axes

# ============================================================
# Panel A: proportional — target vs val regret
# ============================================================
for r in prop_dense + prop_poly:
    ax_A.scatter(r['target'], r['val'],
                 color=model_color[r['model']],
                 marker=model_marker[r['model']],
                 s=80, alpha=outcome_alpha(r['val']),
                 edgecolors='k', linewidths=0.5, zorder=3)

ax_A.axhline(fixed_dense_best, color=model_color['dense'], ls=':', lw=1.5,
             label=f'fixed-σ best dense ({fixed_dense_best:.3f})')
ax_A.axhline(fixed_poly_best,  color=model_color['poly'],  ls=':', lw=1.5,
             label=f'fixed-σ best poly ({fixed_poly_best:.3f})')
ax_A.set_xlabel('RCR target', fontsize=10)
ax_A.set_ylabel('Final val regret', fontsize=10)
ax_A.set_title('Proportional: target RCR vs val regret', fontsize=10)
ax_A.set_ylim(-0.05, 1.7)
handles = [mpatches.Patch(color=model_color[m], label=m) for m in ('dense','poly')]
ax_A.legend(handles=handles + ax_A.get_lines(), fontsize=8, loc='upper right')

# ============================================================
# Panel B: feedback — target vs val regret, split by s_init
# ============================================================
for r in fb_dense + fb_poly:
    ls = 'o' if r['s_init'] == 0.1 else '^'
    ax_B.scatter(r['target'] + (0.005 if r['s_init']==1.0 else -0.005),
                 r['val'],
                 color=model_color[r['model']],
                 marker=ls, s=80,
                 alpha=outcome_alpha(r['val']),
                 edgecolors='k', linewidths=0.5, zorder=3)

ax_B.axhline(fixed_dense_best, color=model_color['dense'], ls=':', lw=1.5)
ax_B.axhline(fixed_poly_best,  color=model_color['poly'],  ls=':', lw=1.5)
# Annotate sigma blowup
for r in fb_poly:
    if r['final_sigma'] > 100:
        ax_B.annotate(f'σ→{r["final_sigma"]:.0f}',
                      (r['target'] + (0.005 if r['s_init']==1.0 else -0.005), r['val']),
                      textcoords='offset points', xytext=(4, 4), fontsize=6, color='gray')

ax_B.set_xlabel('RCR target', fontsize=10)
ax_B.set_ylabel('Final val regret', fontsize=10)
ax_B.set_title('Feedback: target RCR vs val regret', fontsize=10)
ax_B.set_ylim(-0.05, 1.7)
s_handles = [plt.scatter([], [], color='gray', marker='o', s=50, label='σ_init=0.1'),
             plt.scatter([], [], color='gray', marker='^', s=50, label='σ_init=1.0')]
ax_B.legend(handles=handles + s_handles, fontsize=8, loc='upper right')

# ============================================================
# Panel C: sigma trajectory — prop stable vs fb blowup
# ============================================================
highlight = [
    # (run_name, label, color, linestyle)
    ('prop_t0.300_s1_poly',   'prop poly t=0.30 (val=0.008)',  '#d6604d', '-'),
    ('prop_t0.450_s1_poly',   'prop poly t=0.45 (blowup)',     '#f4a582', '--'),
    ('fb_t0.300_s1_poly',     'fb   poly t=0.30 s=1.0 (σ→∞)', '#b2182b', ':'),
    ('fb_t0.400_s0.1_poly',   'fb   poly t=0.40 s=0.1 (val=0.112)', '#92c5de', '-'),
    ('prop_t0.300_s1_dense',  'prop dense t=0.30 (val=0.115)', '#2166ac', '-'),
]
for rname, label, color, ls in highlight:
    r = next((x for x in adap if x['name'] == rname), None)
    if r is None:
        continue
    ax_C.semilogy(r['epochs'], r['sigma_traj'], color=color, ls=ls, lw=1.8, label=label)

ax_C.set_xlabel('Epoch', fontsize=10)
ax_C.set_ylabel('Sigma (log scale)', fontsize=10)
ax_C.set_title('Sigma trajectory over training', fontsize=10)
ax_C.legend(fontsize=7.5, loc='upper left')
ax_C.set_ylim(1e-4, 2e3)

# ============================================================
# Panel D: summary bar chart — best per method
# ============================================================
methods = [
    ('Fixed σ\ndense', fixed_dense_best, model_color['dense'], '//'),
    ('Fixed σ\npoly',  fixed_poly_best,  model_color['poly'],  '//'),
    ('Prop\ndense',    min(r['val'] for r in prop_dense), model_color['dense'], ''),
    ('Prop\npoly',     min(r['val'] for r in prop_poly),  model_color['poly'],  ''),
    ('FB\ndense',      min(r['val'] for r in fb_dense),   model_color['dense'], 'xx'),
    ('FB\npoly',       min(r['val'] for r in fb_poly),    model_color['poly'],  'xx'),
]
xs = np.arange(len(methods))
for i, (label, val, color, hatch) in enumerate(methods):
    ax_D.bar(i, val, color=color, hatch=hatch, edgecolor='k', linewidth=0.8, alpha=0.85)
    ax_D.text(i, val + 0.003, f'{val:.4f}', ha='center', va='bottom', fontsize=8)

ax_D.set_xticks(xs)
ax_D.set_xticklabels([m[0] for m in methods], fontsize=9)
ax_D.set_ylabel('Best val regret (lower = better)', fontsize=10)
ax_D.set_title('Best result per method (no sigma tuning for adaptive)', fontsize=10)
ax_D.set_ylim(0, 0.19)

plt.tight_layout()
out = f'{adap_dir}/fig_adaptive_results.png'
plt.savefig(out, dpi=150, bbox_inches='tight')
print(f'Saved → {out}')
plt.close()
