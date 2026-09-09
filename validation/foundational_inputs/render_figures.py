from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / 'results'
FIGURES = ROOT / 'figures'
FIGURES.mkdir(parents=True, exist_ok=True)

eta = pd.read_csv(RESULTS / 'eta_flow_label_sensitivity.csv')
ont = pd.read_csv(RESULTS / 'ontology_fixed_prediction_robustness.csv')
seed = pd.read_csv(RESULTS / 'ontology_resampling_seed_robustness.csv')

# Eta: keep labels away from data and mark the frozen coefficient explicitly.
fig, ax = plt.subplots(figsize=(7.1, 3.7))
for dataset, g in eta.groupby('dataset', sort=False):
    g = g.sort_values('eta')
    ax.plot(g['eta'], g['max_flow_abs_diff_vs_primary_eta'], marker='o', linewidth=1.6, label=dataset)
ax.axvline(1e-6, linestyle='--', linewidth=1.1, label=r'frozen $\eta=10^{-6}$')
ax.set_xscale('log')
ax.set_xlabel(r'GraphFlow scalarization coefficient $\eta$')
ax.set_ylabel(r'Maximum $|f_{\eta}-f_{10^{-6}}|$')
ax.legend(frameon=False, fontsize=8, ncol=2, loc='upper right')
ax.grid(axis='y', alpha=0.2)
fig.tight_layout(pad=0.8)
fig.savefig(FIGURES / 'supp_eta_robustness.pdf', bbox_inches='tight')
fig.savefig(FIGURES / 'supp_eta_robustness.png', dpi=260, bbox_inches='tight')
plt.close(fig)

order = list(ont['ontology'])
pretty = {
    'O9_fine': 'O9 fine',
    'O6_adjacent': 'O6 adjacent',
    'O5_canonical': 'O5 canonical',
    'O5_activation_merged': 'O5 activation\nmerged',
    'O4_coarse': 'O4 coarse',
    'O3_coarse': 'O3 coarse',
}
x = np.arange(len(order))

# Ontology movement gets its own scale.
fig, ax = plt.subplots(figsize=(7.1, 3.15))
ax.plot(x, ont['movement_lcb'], marker='o', linewidth=1.6)
ax.axhline(0, linewidth=0.9)
ax.set_xticks(x)
ax.set_xticklabels([pretty[o] for o in order], fontsize=8)
ax.set_ylabel('Movement lower confidence bound')
ax.set_xlabel('Outcome-blind ontology re-expression')
ax.grid(axis='y', alpha=0.2)
fig.tight_layout(pad=0.8)
fig.savefig(FIGURES / 'supp_ontology_movement.pdf', bbox_inches='tight')
fig.savefig(FIGURES / 'supp_ontology_movement.png', dpi=260, bbox_inches='tight')
plt.close(fig)

# Utility is near the decision boundary; show primary point and resampling-seed range.
grp = seed.groupby('ontology', sort=False)['utility_lcb'].agg(['min','mean','max']).reindex(order)
primary = ont.set_index('ontology').reindex(order)
y = grp['mean'].to_numpy()
yerr = np.vstack([y-grp['min'].to_numpy(), grp['max'].to_numpy()-y])
fig, ax = plt.subplots(figsize=(7.1, 3.25))
ax.errorbar(x, y, yerr=yerr, fmt='o', capsize=4, linewidth=1.3, label='Utility LCB (mean and range across 4 seeds)')
ax.plot(x, primary['deployment_retention_lcb'].to_numpy(), marker='s', linewidth=1.3, label='Deployment Retention LCB')
ax.axhline(0, linewidth=0.9)
ax.set_xticks(x)
ax.set_xticklabels([pretty[o] for o in order], fontsize=8)
ax.set_ylabel('Lower confidence bound')
ax.set_xlabel('Outcome-blind ontology re-expression')
ax.legend(frameon=False, fontsize=7.5, loc='lower center', bbox_to_anchor=(0.5, 1.01), ncol=2)
ax.grid(axis='y', alpha=0.2)
fig.tight_layout(pad=0.8)
fig.savefig(FIGURES / 'supp_ontology_utility_retention.pdf', bbox_inches='tight')
fig.savefig(FIGURES / 'supp_ontology_utility_retention.png', dpi=260, bbox_inches='tight')
plt.close(fig)
