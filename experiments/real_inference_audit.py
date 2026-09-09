from pathlib import Path
import json, time
import pandas as pd, numpy as np
from qualifyot.candidates import RobustBlendCandidate
from qualifyot.candidate_api import GraphFlowCandidate
from qualifyot.generic_engine import run_generic_lopo
from qualifyot.inference import one_sided_lcb

ROOT=Path(__file__).resolve().parents[1]; DATA=ROOT/'data'/'processed_pairs'; OUT=ROOT/'results'/'inference'/'real_data'; OUT.mkdir(parents=True,exist_ok=True)

def states_of(df): return [c.split('__',1)[1] for c in df.columns if c.startswith('source__')]

specs=[
 ('GSE123813','GSE123813_pairs.csv','robust',None),
 ('GSE272993_9state','GSE272993_9state_pairs.csv','robust',None),
 ('GSE236581','GSE236581_pairs.csv','robust',None),
 ('GSE120575','GSE120575_pairs.csv','robust',None),
 ('GSE179994','GSE179994_pairs.csv','robust',None),
 ('GSE175522','GSE175522_D0_D7_Bcell_pairs.csv','robust',None),
 ('HCC','GSE229772_HCC_primary_pairs.csv','robust',None),
 ('AML','GSE235063_AML_primary_pairs.csv','graph',[('Monocytic','GMP'),('GMP','Progenitor'),('Progenitor','HSC')]),
]
rows=[]
for j,(name,fn,kind,edges) in enumerate(specs):
    df=pd.read_csv(DATA/fn); states=states_of(df)
    cand=RobustBlendCandidate(states=states) if kind=='robust' else GraphFlowCandidate(states=states,edges=edges,alpha=1.0,patient_balanced=False,name='GraphFlow')
    t=time.time(); out=run_generic_lopo(df,cand,bootstrap=200,seed=20260920+j,states=states,reference_rule='nested',orthogonal_profiles=False,return_predictions=False,movement_contrast_method='t')
    pi=out['patient_influence']; pi.to_csv(OUT/f'{name}_patient_influence.csv',index=False)
    # Additional one-sided diagnostics on already-computed patient contributions.
    u=pi.PUC_contribution.to_numpy(float); n=pi.NPI_contribution.to_numpy(float); gm=pi.movement_margin_contribution.to_numpy(float)
    row=dict(dataset=name,candidate=out['candidate'],patients=out['patients'],pairs=out['pairs'],seconds=time.time()-t,
             PDR=out['PDR'],PDR_lo_2s_pct=out['PDR_lo'],PUC=out['PUC'],PUC_lo_2s_pct=out['PUC_lo'],NPI=out['NPI'],NPI_lo_2s_pct=out['NPI_lo'],
             movement_denominator=out['movement_margin_evidence']['denominator'],movement_contrast=out['movement_margin_evidence']['contrast'],movement_contrast_lcb_t_1s=out['movement_margin_evidence']['contrast_lcb'],
             utility_lcb_t_1s=one_sided_lcb(u,method='t',alpha=.05),retention_lcb_t_1s=one_sided_lcb(n,method='t',alpha=.05),
             legacy_core_state=out['core_evidence_state'],retention_stability=out['retention_stability'])
    rows.append(row); print(name,row,flush=True)
pd.DataFrame(rows).to_csv(OUT/'real_inference_audit.csv',index=False)
