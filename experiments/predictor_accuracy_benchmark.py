from pathlib import Path
import time
import numpy as np
import pandas as pd
from qualifyot.candidates import RobustBlendCandidate, DirectDeltaRidgeCandidate
from qualifyot.adaptive_candidates import PatientMultiScaleDeltaCandidate
from qualifyot.comparison_candidates import CompositionEntropicOTCandidate
from qualifyot.candidate_api import MeanDeltaCandidate
from qualifyot.model import feature_matrix, mae
from qualifyot.utils import patient_equal

ROOT=Path(__file__).resolve().parents[1]; DATA=ROOT/'data'/'processed_pairs'; OUT=ROOT/'results'/'predictor_benchmark'; OUT.mkdir(parents=True,exist_ok=True)
rows=[]
for f in sorted(DATA.glob('*.csv')):
    df=pd.read_csv(f).reset_index(drop=True); st=[c.split('__',1)[1] for c in df if c.startswith('source__')]
    _,S,T=feature_matrix(df,states=st); P=df.patient_id.astype(str).to_numpy()
    candidates=[
      MeanDeltaCandidate(states=st,patient_balanced=True,name='MeanDelta'),
      RobustBlendCandidate(states=st,alpha=10,blend=.5,name='RobustBlend'),
      DirectDeltaRidgeCandidate(states=st,alpha=10,patient_balanced=True,name='DirectDeltaRidge_a10'),
      PatientMultiScaleDeltaCandidate(states=st,name='PatientMultiScaleDelta'),
      CompositionEntropicOTCandidate(states=st,epsilon=.25,patient_balanced=True,name='CompositionEntropicOT_e0.25')]
    for c in candidates:
        pred=np.zeros_like(T); t0=time.time()
        for h in np.unique(P):
            te=P==h; tr=~te
            m=c.fresh().fit(df.loc[tr].reset_index(drop=True)); pred[te]=m.predict(df.loc[te].reset_index(drop=True))
        pl=mae(T,pred); risk=patient_equal(pl,P)
        rows.append({'dataset':f.stem,'candidate':c.name,'n_patients':len(np.unique(P)),'n_pairs':len(df),'K':len(st),'patient_equal_MAE':risk,'runtime_s':time.time()-t0})
        print(rows[-1],flush=True)
res=pd.DataFrame(rows); res['risk_rank']=res.groupby('dataset')['patient_equal_MAE'].rank(method='average')
res.to_csv(OUT/'accuracy_head_to_head_9datasets.csv',index=False)
s=res.groupby('candidate').agg(datasets=('dataset','nunique'),mean_rank=('risk_rank','mean'),median_rank=('risk_rank','median'),wins=('risk_rank',lambda x:int((x==1).sum())),mean_MAE=('patient_equal_MAE','mean')).reset_index().sort_values('mean_rank')
s.to_csv(OUT/'accuracy_head_to_head_summary.csv',index=False)
print('\n',s.to_string(index=False))
