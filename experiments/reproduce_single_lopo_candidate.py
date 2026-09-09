from __future__ import annotations
import argparse, json, time
from pathlib import Path
import numpy as np, pandas as pd
from qualifyot.generic_engine import run_generic_lopo
from qualifyot.candidates import RobustBlendCandidate, DirectDeltaRidgeCandidate
from qualifyot.adaptive_candidates import PatientMultiScaleDeltaCandidate
from qualifyot.comparison_candidates import CompositionEntropicOTCandidate
from qualifyot.decision import IUTConfig

ROOT=Path(__file__).resolve().parents[1]

def states_of(df): return [c.split('__',1)[1] for c in df.columns if c.startswith('source__')]

def make_candidate(name, states):
    if name=='RobustBlend': return RobustBlendCandidate(states=states,alpha=10.0,blend=.5,name=name)
    if name=='DirectDeltaRidge_a10': return DirectDeltaRidgeCandidate(states=states,alpha=10.0,patient_balanced=True,name=name)
    if name=='PatientMultiScaleDelta': return PatientMultiScaleDeltaCandidate(states=states,name=name)
    if name=='CompositionEntropicOT_e0.25': return CompositionEntropicOTCandidate(states=states,epsilon=.25,patient_balanced=True,name=name)
    raise ValueError(name)

ap=argparse.ArgumentParser()
ap.add_argument('--dataset', required=True)
ap.add_argument('--candidate', required=True)
ap.add_argument('--bootstrap', type=int, default=2000)
ap.add_argument('--seed', type=int, required=True)
ap.add_argument('--out', required=True)
a=ap.parse_args()
path=ROOT/'data'/'processed_pairs'/f'{a.dataset}.csv'
df=pd.read_csv(path); states=states_of(df); cand=make_candidate(a.candidate, states)
t0=time.time()
out=run_generic_lopo(df,cand,bootstrap=a.bootstrap,seed=a.seed,patient_balanced=True,states=states,
                     reference_rule='MeanDelta',iut_cfg=IUTConfig(),orthogonal_profiles=False,return_predictions=False,
                     movement_contrast_method='t',use_movement_contrast_for_core=True)
r={
 'dataset':a.dataset,'candidate':a.candidate,'n_patients':int(df.patient_id.astype(str).nunique()),'n_pairs':int(len(df)),'K':len(states),
 'candidate_risk':float(out['reference_risk']-out['PUC']),'reference_risk':float(out['reference_risk']),
 'utility':float(out['PUC']),'utility_lcb':float(out['PUC_lo']),'utility_ucb':float(out['PUC_hi']),
 'retention':float(out['NPI']),'retention_lcb':float(out['NPI_lo']),'retention_ucb':float(out['NPI_hi']),
 'movement':float(out['movement_margin_evidence']['contrast']),'movement_lcb':float(out['movement_margin_evidence']['contrast_lcb']),
 'movement_ucb':float(out['movement_margin_evidence'].get('contrast_ucb', np.nan)),
 'retention_stability':float(out['retention_stability']),'state':out['core_evidence_state'],'qualified':bool(out['core_qualified']),
 'runtime_s':time.time()-t0,'seed':a.seed,'bootstrap':a.bootstrap
}
Path(a.out).parent.mkdir(parents=True,exist_ok=True)
Path(a.out).write_text(json.dumps(r,indent=2,sort_keys=True)+'\n',encoding='utf-8')
print(json.dumps(r,sort_keys=True),flush=True)
