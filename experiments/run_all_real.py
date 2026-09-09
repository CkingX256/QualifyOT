from __future__ import annotations
import argparse, json
from pathlib import Path
import pandas as pd

HERE=Path(__file__).resolve().parents[1]
from qualifyot.candidates import RobustBlendCandidate
from qualifyot.generic_engine import run_generic_lopo

DATA={
 'GSE272993_9state':('GSE272993_9state_pairs.csv',['SCM','CM','Early Activated','IFN','Activated','Early Effector','Effector','NK-like','Exhausted'],'LOPO'),
 'GSE236581':('GSE236581_pairs.csv',['Memory','Effector','Exhausted','Regulatory','Other_T'],'LOPO'),
 'GSE123813':('GSE123813_pairs.csv',['Memory','Effector','Exhausted','Regulatory','Other_T'],'LOPO'),
 'GSE120575':('GSE120575_pairs.csv',['Memory','Effector','Exhausted','Regulatory','Other_T'],'LOPO'),
 'GSE179994_refit':('GSE179994_pairs.csv',['Memory','Effector','Exhausted','Regulatory','Other_T'],'exploratory target-refit'),
 'GSE235063_AML':('GSE235063_AML_primary_pairs.csv',['HSC','Progenitor','GMP','Monocytic','Other'],'retrospective'),
 'GSE229772_HCC':('GSE229772_HCC_primary_pairs.csv',['Tumor','Cytotoxic','Memory_Lymphoid','Immunoregulatory','Other_TME'],'retrospective'),
 'GSE175522_Bcell_n6':('GSE175522_D0_D7_Bcell_pairs.csv',['Memory','Effector','Exhausted','Regulatory','Other_T'],'small-N stress'),
}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--bootstrap',type=int,default=5000)
    ap.add_argument('--seed',type=int,default=20260824)
    ap.add_argument('--dataset',default='all',choices=['all',*DATA.keys()])
    ap.add_argument('--out',default=str(HERE/'results'/'new_run'))
    a=ap.parse_args()
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    rows=[]
    todo=DATA.items() if a.dataset=='all' else [(a.dataset,DATA[a.dataset])]
    for i,(ds,(fn,states,role)) in enumerate(todo):
        p=pd.read_csv(HERE/'data'/'processed_pairs'/fn)
        c=RobustBlendCandidate(states=states)
        r=run_generic_lopo(p,c,bootstrap=a.bootstrap,seed=a.seed+i*10000,states=states)
        folds=r.pop('folds'); gates=r.pop('gates')
        folds.to_csv(out/f'{ds}__RobustBlend__folds.csv',index=False)
        rows.append({**r,'dataset':ds,'role':role,'gates_json':json.dumps(gates,sort_keys=True)})
        pd.DataFrame(rows).to_csv(out/'summary.csv',index=False)
        print(ds,'PUC',r['PUC'],'NPI',r['NPI'],'positive_fold_fraction',r['positive_weight_fold_fraction'],'qualified',r['qualified'],flush=True)
    print('GSE197268 is intentionally pending: see data/pending/README_REQUIRED_DATA.md')
if __name__=='__main__': main()
