from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / 'results' / 'pseudoreplication'
out = R / 'pseudoreplication_patient_unit_main'
null = pd.read_csv(R / 'pseudoreplication_cell_count_null.csv')
power = pd.read_csv(R / 'pseudoreplication_patient_count_power.csv')
dup = pd.read_csv(R / 'pseudoreplication_mechanical_duplication.csv')

fig = plt.figure(figsize=(11.4, 3.55))
gs = fig.add_gridspec(1, 3, wspace=0.36)

ax = fig.add_subplot(gs[0,0])
ax.plot(null['cells_per_patient'], null['naive_cell_type1_one_sided'], marker='o', label='Naive cell-level')
ax.plot(null['cells_per_patient'], null['patient_type1_one_sided'], marker='o', label='Patient-level')
ax.axhline(0.05, linestyle='--', linewidth=1, label='Nominal 0.05')
ax.set_xscale('log')
ax.set_ylim(0, 0.54)
ax.set_xlabel('Cells per physical patient')
ax.set_ylabel('One-sided Type-I error')
ax.legend(frameon=False, fontsize=8)
ax.text(-0.17, 1.05, 'a', transform=ax.transAxes, fontweight='bold', fontsize=12)

ax = fig.add_subplot(gs[0,1])
ax.plot(null['cells_per_patient'], null['mean_naive_95ci_halfwidth'], marker='o', label='Naive cell-level')
ax.plot(null['cells_per_patient'], null['mean_patient_95ci_halfwidth'], marker='o', label='Patient-level')
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel('Cells per physical patient')
ax.set_ylabel('Mean 95% CI half-width')
ax.legend(frameon=False, fontsize=8)
ax.text(-0.17, 1.05, 'b', transform=ax.transAxes, fontweight='bold', fontsize=12)

ax = fig.add_subplot(gs[0,2])
ax.plot(dup['duplication_factor'], dup['naive_t_ratio_vs_k1'], marker='o', label='Naive cell t ratio')
ax.plot(dup['duplication_factor'], dup['theoretical_sqrt_k'], linestyle='--', label=r'$\sqrt{k}$')
ax.plot(dup['duplication_factor'], [1.0]*len(dup), linestyle=':', label='Patient t ratio')
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel('Mechanical row-duplication factor $k$')
ax.set_ylabel('t-statistic ratio vs. $k=1$')
ax.legend(frameon=False, fontsize=8)
ax.text(-0.17, 1.05, 'c', transform=ax.transAxes, fontweight='bold', fontsize=12)

fig.savefig(out.with_suffix('.pdf'), bbox_inches='tight')
fig.savefig(out.with_suffix('.png'), dpi=300, bbox_inches='tight')
plt.close(fig)
print(out.with_suffix('.pdf'))
