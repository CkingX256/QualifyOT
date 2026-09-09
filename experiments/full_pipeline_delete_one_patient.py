from __future__ import annotations
import argparse, hashlib, json, time
from pathlib import Path
import pandas as pd
from qualifyot.candidates import RobustBlendCandidate
from qualifyot.candidate_api import GraphFlowCandidate
from qualifyot.generic_engine import run_generic_lopo

ROOT=Path(__file__).resolve().parents[1]; DATA=ROOT/'data'/'processed_pairs'
SPECS={
 'GSE123813':('GSE123813_pairs.csv','robust',None,0),
 'GSE272993_9state':('GSE272993_9state_pairs.csv','robust',None,1),
 'GSE236581':('GSE236581_pairs.csv','robust',None,2),
 'AML':('GSE235063_AML_primary_pairs.csv','graph',[('Monocytic','GMP'),('GMP','Progenitor'),('Progenitor','HSC')],7),
}
def states_of(df): return [c.split('__',1)[1] for c in df.columns if c.startswith('source__')]
def cand(kind,states,edges):
    return RobustBlendCandidate(states) if kind=='robust' else GraphFlowCandidate(states=states,edges=edges,alpha=1.0,patient_balanced=False,name='GraphFlow')
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--dataset',required=True,choices=SPECS); ap.add_argument('--max-new',type=int,default=0); ap.add_argument('--bootstrap',type=int,default=1000); ap.add_argument('--seed',type=int,default=20260901); ap.add_argument('--outdir',required=True)
    a=ap.parse_args(); fn,kind,edges,gidx=SPECS[a.dataset]; df=pd.read_csv(DATA/fn); states=states_of(df); pids=sorted(df.patient_id.astype(str).unique()); outdir=Path(a.outdir); outdir.mkdir(parents=True,exist_ok=True)
    outfile=outdir/f'{a.dataset}__full_pipeline_delete_one.csv'; rows=pd.read_csv(outfile).to_dict('records') if outfile.exists() else []; done={str(r['deleted_patient']) for r in rows}
    new_count=0
    for j,pid in enumerate(pids):
        if pid in done: print(json.dumps({'dataset':a.dataset,'deleted_patient':pid,'status':'SKIP_ALREADY_COMPLETE'}),flush=True); continue
        sub=df[df.patient_id.astype(str)!=pid].reset_index(drop=True)
        # Seed schedule is deterministic from the frozen dataset index and sorted deletion index; no outcome-adaptive reruns.
        runseed=int(a.seed + gidx*10000 + 500000 + j*1000)
        t=time.time(); o=run_generic_lopo(sub,cand(kind,states,edges),bootstrap=a.bootstrap,seed=runseed,states=states,reference_rule='nested',movement_contrast_method='t',use_movement_contrast_for_core=True,orthogonal_profiles=False)
        row=dict(dataset=a.dataset,deleted_patient=pid,n_patients=int(o['patients']),n_pairs=int(o['pairs']),candidate=o['candidate'],state=o['core_evidence_state'],qualified=bool(o['core_qualified']),movement_contrast=float(o['movement_margin_evidence']['contrast']),movement_lcb=float(o['movement_margin_evidence']['contrast_lcb']),utility=float(o['PUC']),utility_lcb=float(o['PUC_lo']),utility_ucb=float(o['PUC_hi']),retention=float(o['NPI']),retention_lcb=float(o['NPI_lo']),retention_ucb=float(o['NPI_hi']),retention_stability=float(o['retention_stability']),reference_risk=float(o['reference_risk']),retained_risk=float(o['qualified_risk']),bootstrap=a.bootstrap,seed=runseed,seconds=time.time()-t)
        rows.append(row); pd.DataFrame(rows).sort_values('deleted_patient').to_csv(outfile,index=False); print(json.dumps(row,sort_keys=True),flush=True)
        new_count += 1
        if a.max_new and new_count >= a.max_new:
            break
if __name__=='__main__': main()
