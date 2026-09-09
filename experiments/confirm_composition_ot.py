from pathlib import Path
import pandas as pd, time
from qualifyot.generic_engine import run_generic_lopo
from qualifyot.comparison_candidates import CompositionEntropicOTCandidate
from qualifyot.decision import IUTConfig
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'results'/'predictor_benchmark'; rows=[]
for name in ['GSE235063_AML_primary_pairs.csv','GSE236581_pairs.csv']:
 d=pd.read_csv(ROOT/'data'/'processed_pairs'/name); st=[c.split('__',1)[1] for c in d if c.startswith('source__')]
 t=time.time(); o=run_generic_lopo(d,CompositionEntropicOTCandidate(st),bootstrap=1000,seed=20260902,patient_balanced=True,states=st,reference_rule='MeanDelta',iut_cfg=IUTConfig(),orthogonal_profiles=True,profile_draws=5000,movement_contrast_method='t',use_movement_contrast_for_core=True)
 rows.append({'dataset':name.replace('.csv',''),'candidate':o['candidate'],'patients':o['patients'],'reference':'MeanDelta','PUC':o['PUC'],'PUC_lo':o['PUC_lo'],'PUC_hi':o['PUC_hi'],'NPI':o['NPI'],'NPI_lo':o['NPI_lo'],'NPI_hi':o['NPI_hi'],'GM':o['movement_margin_evidence']['contrast'],'GM_lcb':o['movement_margin_evidence']['contrast_lcb'],'retention_stability':o['retention_stability'],'state':o['core_evidence_state'],'qualified':o['core_qualified'],'runtime_s':time.time()-t})
 print(rows[-1],flush=True)
pd.DataFrame(rows).to_csv(OUT/'composition_ot_targeted_1000.csv',index=False)
