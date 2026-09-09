from __future__ import annotations
import argparse, json, time
from pathlib import Path
import pandas as pd
from qualifyot.candidates import RobustBlendCandidate
from qualifyot.candidate_api import GraphFlowCandidate
from qualifyot.generic_engine import run_generic_lopo

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'data'/'processed_pairs'
SPECS=[
 ('GSE123813','GSE123813_pairs.csv','robust',None),
 ('GSE272993_9state','GSE272993_9state_pairs.csv','robust',None),
 ('GSE236581','GSE236581_pairs.csv','robust',None),
 ('GSE120575','GSE120575_pairs.csv','robust',None),
 ('GSE179994','GSE179994_pairs.csv','robust',None),
 ('GSE175522','GSE175522_D0_D7_Bcell_pairs.csv','robust',None),
 ('HCC','GSE229772_HCC_primary_pairs.csv','robust',None),
 ('AML','GSE235063_AML_primary_pairs.csv','graph',[('Monocytic','GMP'),('GMP','Progenitor'),('Progenitor','HSC')]),
]
REFS=['nested','Persistence','CohortMean','MeanDelta']
def states_of(df): return [c.split('__',1)[1] for c in df.columns if c.startswith('source__')]
def candidate_of(kind,states,edges):
    if kind=='robust': return RobustBlendCandidate(states=states)
    return GraphFlowCandidate(states=states,edges=edges,alpha=1.0,patient_balanced=False,name='GraphFlow')

ap=argparse.ArgumentParser(); ap.add_argument('--dataset',required=True); ap.add_argument('--reference',choices=REFS); ap.add_argument('--bootstrap',type=int,default=1000); ap.add_argument('--seed',type=int,default=20260901); ap.add_argument('--outdir',required=True)
a=ap.parse_args()
idx={x[0]:i for i,x in enumerate(SPECS)}[a.dataset]
name,fn,kind,edges=SPECS[idx]
outdir=Path(a.outdir); outdir.mkdir(parents=True,exist_ok=True)
df=pd.read_csv(DATA/fn); states=states_of(df)
outfile=outdir/f'{name}__reference_matrix.csv'
rows=pd.read_csv(outfile).to_dict('records') if outfile.exists() else []
done_refs={str(r['reference']) for r in rows}
refs_to_run=[a.reference] if a.reference else REFS
for ref in refs_to_run:
    if ref in done_refs:
        print(json.dumps({'dataset':name,'reference':ref,'status':'SKIP_ALREADY_COMPLETE'},sort_keys=True),flush=True)
        continue
    k=REFS.index(ref)
    cand=candidate_of(kind,states,edges); runseed=a.seed+idx*10000+k*1000; t=time.time()
    out=run_generic_lopo(df,cand,bootstrap=a.bootstrap,seed=runseed,states=states,reference_rule=ref,
                         movement_contrast_method='t',use_movement_contrast_for_core=True,
                         return_predictions=(ref=='nested'),orthogonal_profiles=False)
    row=dict(dataset=name,reference=ref,candidate=out['candidate'],patients=out['patients'],pairs=out['pairs'],
             state=out['core_evidence_state'],qualified=bool(out['core_qualified']),
             movement_contrast=float(out['movement_margin_evidence']['contrast']),
             movement_contrast_lcb=float(out['movement_margin_evidence']['contrast_lcb']),
             movement_ratio=float(out['PDR']),utility=float(out['PUC']),utility_lo=float(out['PUC_lo']),utility_hi=float(out['PUC_hi']),
             retention=float(out['NPI']),retention_lo=float(out['NPI_lo']),retention_hi=float(out['NPI_hi']),
             retention_stability=float(out['retention_stability']),reference_risk=float(out['reference_risk']),retained_risk=float(out['qualified_risk']),
             seconds=time.time()-t,bootstrap=a.bootstrap,seed=runseed)
    rows.append(row); pd.DataFrame(rows).sort_values('seed').to_csv(outfile,index=False)
    if ref=='nested':
        out['patient_influence'].to_csv(outdir/f'{name}__nested_patient_influence.csv',index=False)
        # prediction vectors need a safe serialized representation for fixed-prediction loss audit
        pr=out['prediction_rows'].copy()
        for c in ['target','source','reference','candidate','retained']:
            pr[c]=pr[c].map(lambda x: json.dumps([float(v) for v in x],separators=(',',':')))
        pr.to_csv(outdir/f'{name}__nested_prediction_rows.csv',index=False)
    print(json.dumps(row,sort_keys=True),flush=True)
