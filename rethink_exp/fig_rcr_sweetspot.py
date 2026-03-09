"""
Figure: RCR sweet spot and comparison with Entropy / Dist-to-Binary.

Saves to saved_records/cubic-gen/perturb_sigma_sweep/diag_run1/fig_rcr_sweetspot.png
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import glob, os

# ---- Load all runs ----
base = 'saved_records/cubic-gen/perturb_sigma_sweep/diag_run1'
files = sorted(glob.glob(f'{base}/*.npz'))

runs = []
for f in files:
    d = np.load(f, allow_pickle=True)
    name = os.path.basename(f).replace('.npz', '')
    parts = name.split('_')
    sigma = float(parts[1])
    model = parts[2]           # 'dense' or 'poly'
    final_val = float(d['val_regret'][-20:].mean())
    runs.append(dict(
        name=name, sigma=sigma, model=model, final_val=final_val,
        rcr=d['rank_change_rate'],
        entropy=d['entropy'],
        dist=d['dist_binary'],
        val=d['val_regret'],
        epochs=d['epoch'],
    ))

# ---- Colour helpers ----
sigma_vals = sorted(set(r['sigma'] for r in runs))
log_sigma = np.log10(sigma_vals)
log_min, log_max = log_sigma.min(), log_sigma.max()
cmap_sigma = plt.cm.plasma

def sigma_color(s):
    t = (np.log10(s) - log_min) / (log_max - log_min)
    return cmap_sigma(t)

# Outcome labels
def outcome(final_val):
    if final_val < 0.05:
        return 'excellent'
    elif final_val < 0.15:
        return 'good'
    elif final_val < 0.40:
        return 'partial'
    else:
        return 'stuck'

outcome_color = {'excellent': '#1a9641', 'good': '#a6d96a',
                 'partial': '#fdae61', 'stuck': '#d7191c'}
outcome_marker = {'excellent': '*', 'good': 'o', 'partial': 's', 'stuck': 'X'}
outcome_size   = {'excellent': 180, 'good': 80, 'partial': 80, 'stuck': 80}

# ---- Pearson r helper ----
def pearson(a, b):
    a = a - a.mean(); b = b - b.mean()
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return np.dot(a, b) / denom if denom > 0 else 0.0

# ---- Figure layout ----
fig = plt.figure(figsize=(14, 10))
gs = fig.add_gridspec(2, 3, hspace=0.45, wspace=0.35,
                      left=0.07, right=0.97, top=0.93, bottom=0.07)

# Row 0: per-run mean-metric vs final val-regret (three metrics side by side)
ax_r = [fig.add_subplot(gs[0, i]) for i in range(3)]
# Row 1, col 0-1: RCR & Entropy trajectory over epochs for key runs
ax_traj_rcr = fig.add_subplot(gs[1, 0])
ax_traj_ent = fig.add_subplot(gs[1, 1])
# Row 1, col 2: scatter cosine-sim vs each metric to show which is more predictive
ax_scatter = fig.add_subplot(gs[1, 2])

metric_keys = ['rcr', 'entropy', 'dist']
metric_labels = ['Rank Change Rate (RCR)', 'Entropy H(z̄)', 'Dist to Binary ‖z̄−round(z̄)‖']
metric_short  = ['RCR', 'Entropy', 'Dist2Bin']

# ---- Row 0: per-run mean metric vs final val regret ----
for col, (key, label, short) in enumerate(zip(metric_keys, metric_labels, metric_short)):
    ax = ax_r[col]
    for r in runs:
        oc = outcome(r['final_val'])
        mean_metric = r[key].mean()
        ax.scatter(mean_metric, r['final_val'],
                   color=outcome_color[oc],
                   marker=outcome_marker[oc],
                   s=outcome_size[oc],
                   edgecolors='k', linewidths=0.5, zorder=3,
                   label=oc)

    # Compute Pearson r over per-run means
    ms = np.array([r[key].mean() for r in runs])
    vs = np.array([r['final_val'] for r in runs])
    r_val = pearson(ms, vs)
    ax.set_xlabel(f'Mean {short}', fontsize=10)
    ax.set_ylabel('Final val regret', fontsize=10) if col == 0 else None
    ax.set_title(f'{label}\n(Pearson r={r_val:.2f} with val regret)', fontsize=9)

    # Reference threshold line for RCR
    if col == 0:
        ax.axvline(0.15, color='gray', ls='--', lw=1.2, label='RCR threshold ≈0.15')
        ax.text(0.15, ax.get_ylim()[1] * 0.95 if ax.get_ylim()[1] > 0 else 0.4,
                ' 0.15', fontsize=8, color='gray', va='top')

# Shared legend for outcome colours
handles = [plt.scatter([], [], color=c, marker=outcome_marker[k],
                       s=outcome_size[k], edgecolors='k', lw=0.5, label=k)
           for k, c in outcome_color.items()]
ax_r[2].legend(handles=handles, title='Outcome', fontsize=8, title_fontsize=8,
               loc='upper right')

# ---- Row 1, col 0-1: epoch trajectories for selected runs ----
# Pick 4 runs that show divergent trajectories
highlight_runs = [
    ('sigma_0.0976859_poly', 'poly σ=0.098 → val=0.010', 'excellent'),
    ('sigma_0.307107_poly',  'poly σ=0.307 → val=0.455', 'stuck'),
    ('sigma_30_dense',       'dense σ=30 → val=0.114',   'good'),
    ('sigma_0.307107_dense', 'dense σ=0.307 → val=0.127','good'),
]

linestyles = ['-', '--', '-', ':']
for ax_traj, key, label_y in [(ax_traj_rcr, 'rcr', 'Rank Change Rate'),
                               (ax_traj_ent, 'entropy', 'Entropy H(z̄)')]:
    for (rname, rlabel, oc), ls in zip(highlight_runs, linestyles):
        r = next(x for x in runs if x['name'] == rname)
        ax_traj.plot(r['epochs'], r[key], ls=ls,
                     color=outcome_color[oc], lw=1.8, label=rlabel)

    if key == 'rcr':
        ax_traj.axhline(0.15, color='gray', ls='--', lw=1.2, label='threshold ≈0.15')
    ax_traj.set_xlabel('Epoch', fontsize=10)
    ax_traj.set_ylabel(label_y, fontsize=10)
    ax_traj.set_title(f'{label_y} over training', fontsize=9)
    ax_traj.legend(fontsize=7.5, loc='upper right')

# ---- Row 1, col 2: RCR vs Entropy scatter to show near-identity ----
rcr_pool  = np.concatenate([r['rcr']     for r in runs])
ent_pool  = np.concatenate([r['entropy'] for r in runs])
dist_pool = np.concatenate([r['dist']    for r in runs])
fr_pool   = np.concatenate([np.full(300, r['final_val']) for r in runs])

# Downsample for readability
rng = np.random.default_rng(42)
idx = rng.choice(len(rcr_pool), size=1500, replace=False)

sc = ax_scatter.scatter(rcr_pool[idx], ent_pool[idx],
                        c=dist_pool[idx], cmap='viridis', s=6, alpha=0.6,
                        vmin=0, vmax=dist_pool.max())
plt.colorbar(sc, ax=ax_scatter, label='Dist to Binary', shrink=0.8)
r_re = pearson(rcr_pool, ent_pool)
ax_scatter.set_xlabel('Rank Change Rate (RCR)', fontsize=10)
ax_scatter.set_ylabel('Entropy H(z̄)', fontsize=10)
ax_scatter.set_title(f'RCR vs Entropy (r={r_re:.3f})\ncoloured by Dist-to-Binary', fontsize=9)

# Annotate correlations with dist
r_rd = pearson(rcr_pool, dist_pool)
r_ed = pearson(ent_pool, dist_pool)
ax_scatter.text(0.04, 0.95,
                f'RCR–Dist r={r_rd:.2f}\nEnt–Dist r={r_ed:.2f}',
                transform=ax_scatter.transAxes, fontsize=8,
                va='top', bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.8))

fig.suptitle('Sigma sweep diagnostics — cubic (20 runs × 300 epochs)',
             fontsize=12, fontweight='bold')

# ---- Save ----
out = f'{base}/fig_rcr_sweetspot.png'
plt.savefig(out, dpi=150, bbox_inches='tight')
print(f'Saved → {out}')
plt.close()
