from __future__ import annotations
from pathlib import Path
import json, time
import numpy as np
import pandas as pd

from qualifyot.generic_engine import run_generic_lopo
from qualifyot.candidates import RobustBlendCandidate, DirectDeltaRidgeCandidate
from qualifyot.adaptive_candidates import PatientMultiScaleDeltaCandidate
from qualifyot.comparison_candidates import CompositionEntropicOTCandidate
from qualifyot.decision import IUTConfig

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'data'/'processed_pairs'
OUT=ROOT/'results'/'predictor_benchmark'
OUT.mkdir(parents=True,exist_ok=True)
FILES=sorted(DATA.glob('*.csv'))

def states_of(df): return [c.split('__',1)[1] for c in df.columns if c.startswith('source__')]

def scalarize(out):
    return {k:v for k,v in out.items() if k not in {'folds','patient_influence','prediction_rows','orthogonal_profiles'} and not isinstance(v,(pd.DataFrame,dict,list,np.ndarray))}

existing_path=OUT/'head_to_head_300.csv'
rows=pd.read_csv(existing_path).to_dict('records') if existing_path.exists() else []
done={(r['dataset'],r['candidate']) for r in rows}
for f in FILES:
    df=pd.read_csv(f); states=states_of(df)
    cands=[
      RobustBlendCandidate(states=states,alpha=10.0,blend=.5,name='RobustBlend'),
      DirectDeltaRidgeCandidate(states=states,alpha=10.0,patient_balanced=True,name='DirectDeltaRidge_a10'),
      PatientMultiScaleDeltaCandidate(states=states,name='PatientMultiScaleDelta'),
      CompositionEntropicOTCandidate(states=states,epsilon=.25,patient_balanced=True,name='CompositionEntropicOT_e0.25'),
    ]
    for ci,cand in enumerate(cands):
        if (f.stem,cand.name) in done: continue
        t=time.time()
        out=run_generic_lopo(df,cand,bootstrap=300,seed=20260902+ci*1000,
            patient_balanced=True,states=states,reference_rule='MeanDelta',
            iut_cfg=IUTConfig(),orthogonal_profiles=False,return_predictions=False,
            movement_contrast_method='t',use_movement_contrast_for_core=True)
        r={'dataset':f.stem,'candidate':cand.name,'n_patients':df.patient_id.nunique(),'n_pairs':len(df),'K':len(states),
           'candidate_risk': float(out['reference_risk']-out['PUC']),
           'reference_risk':float(out['reference_risk']),
           'utility':float(out['PUC']),'utility_lo':float(out['PUC_lo']),'utility_hi':float(out['PUC_hi']),
           'retention':float(out['NPI']),'retention_lo':float(out['NPI_lo']),'retention_hi':float(out['NPI_hi']),
           'movement_contrast':float(out['movement_margin_evidence']['contrast']),
           'movement_contrast_lcb':float(out['movement_margin_evidence']['contrast_lcb']),
           'retention_stability':float(out['retention_stability']),
           'state':out['core_evidence_state'],'qualified':bool(out['core_qualified']),
           'runtime_s':time.time()-t}
        rows.append(r); print(r,flush=True)
        pd.DataFrame(rows).to_csv(OUT/'head_to_head_300.csv',index=False)

res=pd.DataFrame(rows)
# Cross-dataset paired summaries: risk rank and number of supported axes/qualifications.
res['risk_rank']=res.groupby('dataset')['candidate_risk'].rank(method='average')
summary=res.groupby('candidate').agg(
    datasets=('dataset','nunique'),mean_risk_rank=('risk_rank','mean'),median_risk_rank=('risk_rank','median'),
    mean_candidate_risk=('candidate_risk','mean'),mean_utility=('utility','mean'),
    utility_supported=('utility_lo',lambda x:int((x>0).sum())),
    retention_supported=('retention_lo',lambda x:int((x>0).sum())),
    qualified_count=('qualified','sum')).reset_index()
summary.to_csv(OUT/'head_to_head_summary_300.csv',index=False)
# Winner/evidence discordance
wins=[]
for ds,g in res.groupby('dataset'):
    g=g.sort_values('candidate_risk')
    winner=g.iloc[0]
    qual=g[g.qualified]
    wins.append({'dataset':ds,'risk_winner':winner.candidate,'risk_winner_risk':winner.candidate_risk,
                 'risk_winner_state':winner.state,'risk_winner_qualified':winner.qualified,
                 'qualified_candidates':'|'.join(qual.candidate.tolist()) if len(qual) else '',
                 'n_qualified':len(qual)})
pd.DataFrame(wins).to_csv(OUT/'risk_vs_evidence_discordance_300.csv',index=False)
print('\nSUMMARY\n',summary.to_string(index=False))
