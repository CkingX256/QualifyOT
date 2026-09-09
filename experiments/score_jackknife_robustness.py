from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
from qualifyot.inference import one_sided_lcb
from scipy.stats import t

DATASETS=['GSE123813','GSE272993_9state','GSE236581','GSE120575','GSE179994','GSE175522','HCC','AML']

def t_interval(x, alpha=.05):
    x=np.asarray(x,float); n=len(x); mu=float(x.mean())
    if n<2: return mu, float('nan'), float('nan')
    s=float(x.std(ddof=1)); q=float(t.ppf(1-alpha,n-1)); se=s/np.sqrt(n)
    return mu, mu-q*se, mu+q*se

def label(glo, ulo, uhi, nlo):
    if glo<=0: return 'Negligible'
    if uhi<0: return 'Adverse'
    if ulo>0 and nlo>0: return 'Qualified'
    if ulo>0: return 'Promising'
    return 'Equivocal'

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--indir',required=True); ap.add_argument('--outdir',required=True)
    a=ap.parse_args(); indir=Path(a.indir); outdir=Path(a.outdir); outdir.mkdir(parents=True,exist_ok=True)
    rows=[]; summ=[]
    for ds in DATASETS:
        f=indir/f'{ds}__nested_patient_influence.csv'; df=pd.read_csv(f)
        pids=df.patient_id.astype(str).to_numpy(); g=df.movement_margin_contribution.to_numpy(float); u=df.PUC_contribution.to_numpy(float); n=df.NPI_contribution.to_numpy(float)
        bg,bglo,bghi=t_interval(g); bu,bulo,buhi=t_interval(u); bn,bnlo,bnhi=t_interval(n); base=label(bglo,bulo,buhi,bnlo)
        states=[]; q=[]
        for j,pid in enumerate(pids):
            keep=np.arange(len(df))!=j
            gm,glo,ghi=t_interval(g[keep]); um,ulo,uhi=t_interval(u[keep]); nm,nlo,nhi=t_interval(n[keep]); st=label(glo,ulo,uhi,nlo)
            rows.append(dict(dataset=ds,deleted_patient=pid,n_remaining=int(keep.sum()),movement=float(gm),movement_lcb_t=float(glo),utility=float(um),utility_lcb_t=float(ulo),utility_ucb_t=float(uhi),retention=float(nm),retention_lcb_t=float(nlo),retention_ucb_t=float(nhi),state=st,qualified=st=='Qualified'))
            states.append(st); q.append(st=='Qualified')
        summ.append(dict(dataset=ds,n_patients=len(df),base_fixed_score_t_state=base,delete_one_unique_states='|'.join(sorted(set(states))),delete_one_state_change_fraction=float(np.mean(np.asarray(states)!=base)),delete_one_qualified_fraction=float(np.mean(q)),any_delete_one_qualified=bool(any(q)),all_delete_one_same_state=bool(all(s==states[0] for s in states))))
    rdf=pd.DataFrame(rows); sdf=pd.DataFrame(summ)
    rdf.to_csv(outdir/'R6_fixed_score_delete_one_patient_details.csv',index=False); sdf.to_csv(outdir/'R6_fixed_score_delete_one_patient_summary.csv',index=False)
    print(sdf.to_string(index=False)); print(json.dumps({'datasets':len(sdf),'datasets_with_any_state_change':int((sdf.delete_one_state_change_fraction>0).sum()),'datasets_with_any_delete_one_qualified':int(sdf.any_delete_one_qualified.sum())},indent=2))

if __name__=='__main__': main()
