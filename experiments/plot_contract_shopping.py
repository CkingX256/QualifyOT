from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

src = Path('results/contract_shopping/contract_shopping_null.csv')
out = Path('results/contract_shopping/contract_shopping_stress.pdf')
df = pd.read_csv(src)
fig, ax = plt.subplots(figsize=(6.8,4.3))
for rho, g in df.groupby('equicorrelation_rho'):
    ax.plot(g['contracts_K'], g['posthoc_any_contract_false_qualification'], marker='o', label=f'post-hoc, rho={rho:g}')
# frozen nominal reference
ax.axhline(0.05, linestyle='--', linewidth=1.2, label='nominal 0.05')
# family-aware values for middle rho to avoid clutter; its control is conservative under positive corr
mid = df[df['equicorrelation_rho']==0.5]
ax.plot(mid['contracts_K'], mid['family_aware_holm_first_step_false_qualification'], marker='s', linestyle=':', label='family-aware, rho=0.5')
ax.set_xscale('log', base=3)
ax.set_xticks([1,3,9,27]); ax.set_xticklabels(['1','3','9','27'])
ax.set_ylim(0,0.82)
ax.set_xlabel('Number of alternative complete contracts')
ax.set_ylabel('False Qualification probability')
ax.legend(frameon=False, fontsize=8)
ax.grid(alpha=0.2)
fig.tight_layout()
fig.savefig(out, bbox_inches='tight')
fig.savefig(out.with_suffix('.png'), dpi=220, bbox_inches='tight')
print(out)
