from __future__ import annotations
from pathlib import Path
import pandas as pd

def combine(results_root:Path):
    mets=[]; audits=[]
    for p in results_root.iterdir() if results_root.exists() else []:
        if p.is_dir() and (p/'model_metrics.csv').exists(): mets.append(pd.read_csv(p/'model_metrics.csv'))
        if p.is_dir() and (p/'prior_audit.json').exists(): audits.append(pd.read_json(p/'prior_audit.json',typ='series').to_dict())
    mdf=pd.concat(mets,ignore_index=True) if mets else pd.DataFrame()
    adf=pd.DataFrame(audits) if audits else pd.DataFrame()
    if len(mdf): mdf.to_csv(results_root/'ALL_COHORT_MODEL_METRICS.csv',index=False)
    if len(adf): adf.to_csv(results_root/'ALL_COHORT_PRIOR_AUDITS.csv',index=False)
    lines=['# QualifyOT reproducibility summary','']
    if len(mdf):
        lines += ['## Model metrics','',mdf.to_markdown(index=False),'']
    if len(adf):
        cols=[c for c in ['cohort','PDR_Hellinger','PDR_CI2.5','PDR_CI97.5','PUC_reference_minus_prior','PUC_CI2.5','PUC_CI97.5','NPI_reference_minus_qualified','NPI_CI2.5','NPI_CI97.5','positive_weight_fold_fraction','qualified_prior_case','decision','GCF'] if c in adf]
        lines += ['## Prior qualification','',adf[cols].to_markdown(index=False),'']
        pos=adf[adf.get('qualified_prior_case',False)==True] if 'qualified_prior_case' in adf else adf.iloc[0:0]
        lines += ['## Pre-specified qualification criterion', '', f'Positive cohorts: {len(pos)} / {len(adf)}', '']
    (results_root/'QUALIFICATION_SUMMARY.md').write_text('\n'.join(lines),encoding='utf-8')
