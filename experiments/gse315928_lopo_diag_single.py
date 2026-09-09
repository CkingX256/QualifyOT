from __future__ import annotations
import argparse, json, time
from pathlib import Path
import numpy as np, pandas as pd
from qualifyot.generic_engine import run_generic_lopo
from qualifyot.candidates import RobustBlendCandidate, DirectDeltaRidgeCandidate
from qualifyot.adaptive_candidates import PatientMultiScaleDeltaCandidate
from qualifyot.comparison_candidates import CompositionEntropicOTCandidate
from qualifyot.candidate_family import one_sided_t_pvalue, iut_pvalue

ROOT=Path(__file__).resolve().parents[1]

def states_of(df): return [c.split('__',1)[1] for c in df.columns if c.startswith('source__')]
def make(name, states):
    if name=='RobustBlend': return RobustBlendCandidate(states=states,alpha=10.0,blend=.5,name=name)
    if name=='DirectDeltaRidge_a10': return DirectDeltaRidgeCandidate(states=states,alpha=10.0,patient_balanced=True,name=name)
    if name=='PatientMultiScaleDelta': return PatientMultiScaleDeltaCandidate(states=states,name=name)
    if name=='CompositionEntropicOT_e0.25': return CompositionEntropicOTCandidate(states=states,epsilon=.25,patient_balanced=True,name=name)
    raise KeyError(name)

ap=argparse.ArgumentParser(); ap.add_argument('--candidate',required=True); ap.add_argument('--dataset',default='GSE315928_B_F1_pairs.csv'); ap.add_argument('--seed',type=int,required=True); ap.add_argument('--bootstrap',type=int,default=2000); ap.add_argument('--out',required=True)
a=ap.parse_args()
df=pd.read_csv(ROOT/'data/processed_pairs'/a.dataset); states=states_of(df)
t0=time.time(); out=run_generic_lopo(df,make(a.candidate,states),bootstrap=a.bootstrap,seed=a.seed,patient_balanced=True,states=states,reference_rule='MeanDelta',orthogonal_profiles=False,return_predictions=False,movement_contrast_method='t',use_movement_contrast_for_core=True)
pi=out['patient_influence'].copy()
pm=one_sided_t_pvalue(pi.movement_margin_contribution.to_numpy(float)); pu=one_sided_t_pvalue(pi.PUC_contribution.to_numpy(float)); pn=one_sided_t_pvalue(pi.NPI_contribution.to_numpy(float)); pc=iut_pvalue((pm,pu,pn))
r={'candidate':a.candidate,'seed':a.seed,'bootstrap':a.bootstrap,'n_patients':int(out['patients']),'candidate_risk':float(out['reference_risk']-out['PUC']),'reference_risk':float(out['reference_risk']),'utility':float(out['PUC']),'utility_lcb':float(out['PUC_lo']),'retention':float(out['NPI']),'retention_lcb':float(out['NPI_lo']),'movement':float(out['movement_margin_evidence']['contrast']),'movement_lcb':float(out['movement_margin_evidence']['contrast_lcb']),'state':str(out['core_evidence_state']),'core_qualified':bool(out['core_qualified']),'p_m_tdiag':pm,'p_u_tdiag':pu,'p_n_tdiag':pn,'p_candidate_iut_tdiag':pc,'runtime_s':time.time()-t0}
p=Path(a.out); p.parent.mkdir(parents=True,exist_ok=True); p.write_text(json.dumps(r,indent=2,sort_keys=True)+'\n'); pi.to_csv(p.with_suffix('.patients.csv'),index=False); print(json.dumps(r,sort_keys=True))
