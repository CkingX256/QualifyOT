from __future__ import annotations
from pathlib import Path
import sys,math
import numpy as np,pandas as pd
from synthetic_overcomplete_refinement import states,true_graph,graphs,gkey,flow_labels,cv_predictions
from synthetic_subgroup_refinement import make_mix
from qualifyot.model import feature_matrix,incidence,fit_graphflow,mae
from qualifyot.weight_selection import patient_curves,one_se_msw
from sklearn.metrics import roc_auc_score

def tail_score(per,q=.2,gamma=.7):
    per=np.asarray(per,float); k=max(1,int(np.ceil(q*len(per))))
    tail=np.sort(per)[-k:].mean()
    return (1-gamma)*per.mean()+gamma*tail

def select(design,mode='mean',q=.2,gamma=.7):
    X,S,T=feature_matrix(design,states=states);P=design.patient_id.astype(str).to_numpy();rows=[];c={}
    for g in graphs:
      F=flow_labels(design,g);pr=cv_predictions(design,g,F,k=5);c[g]=(F,pr);loss=mae(T,pr);per=np.array([loss[P==p].mean() for p in np.unique(P)])
      score=per.mean() if mode=='mean' else tail_score(per,q,gamma)
      # SE for conservative parsimony around chosen objective: bootstrap-like empirical sd/sqrt(n) on patient losses
      se=per.std(ddof=1)/np.sqrt(len(per)) if len(per)>1 else 0
      rows.append({'g':g,'score':float(score),'se':float(se),'n_edges':len(g)})
    best=min(rows,key=lambda r:(r['score'],r['n_edges'],gkey(r['g'])));thr=best['score']+best['se'];elig=[r for r in rows if r['score']<=thr+1e-15];ch=min(elig,key=lambda r:(r['n_edges'],r['score'],gkey(r['g'])))
    return ch['g'],c

def eval(seed,frac,mode):
    rng=np.random.default_rng(seed);design=make_mix(rng,60,frac);confirm=make_mix(rng,60,frac)
    g,c=select(design,mode);Xd,Sd,Td=feature_matrix(design,states=states);Pd=design.patient_id.astype(str).to_numpy();F,po=c[g]
    grid=np.round(np.arange(0,1.0001,.02),2);_,D,R=patient_curves(Td,Sd,po,Pd,grid);w=one_se_msw(D,R,grid,.05,0,200,seed+44,1)['weight']
    B=incidence(states,g);gm=fit_graphflow(Xd,Sd,Td,alpha=1.,flow_labels=F,Bmat=B,edges=g,patients=Pd,patient_balanced=True)
    Xc,Sc,Tc=feature_matrix(confirm,states=states);C=gm.predict(Xc,Sc);A=(1-w)*Sc+w*C;npi=mae(Tc,Sc)-mae(Tc,A);sub=confirm.reverse_subgroup.to_numpy(int);harm=npi<0
    auc=np.nan
    if len(np.unique(sub))==2:
      try:auc=roc_auc_score(sub,-npi)
      except:pass
    return {'seed':seed,'frac_reverse':frac,'mode':mode,'graph':gkey(g),'n_edges':len(g),'weight':w,'mean_npi':npi.mean(),'harm_frac':harm.mean(),'harm_reverse':harm[sub==1].mean(),'harm_majority':harm[sub==0].mean(),'mean_npi_reverse':npi[sub==1].mean(),'mean_npi_majority':npi[sub==0].mean(),'auc':auc}
if __name__=='__main__':
 out=[]
 for frac in [0.2,0.3,0.4]:
  for mode in ['mean','tail']:
   for rep in range(8):out.append(eval(20260825+int(frac*1000)*10000+(0 if mode=='mean' else 5000)+rep*101,frac,mode))
   d=pd.DataFrame(out);sub=d[(d.frac_reverse==frac)&(d['mode']==mode)];print(frac,mode,sub[['n_edges','weight','mean_npi','harm_frac','harm_reverse','harm_majority','mean_npi_reverse','mean_npi_majority','auc']].mean(numeric_only=True).to_dict(),flush=True)
 pd.DataFrame(out).to_csv(Path(__file__).resolve().parents[1]/'results'/'tail_aware_runs.csv',index=False)
 s=pd.DataFrame(out).groupby(['frac_reverse','mode']).agg(reps=('seed','count'),mean_edges=('n_edges','mean'),mean_weight=('weight','mean'),mean_npi=('mean_npi','mean'),harm_frac=('harm_frac','mean'),harm_reverse=('harm_reverse','mean'),harm_majority=('harm_majority','mean'),mean_npi_reverse=('mean_npi_reverse','mean'),mean_npi_majority=('mean_npi_majority','mean'),auc=('auc','mean')).reset_index();s.to_csv(Path(__file__).resolve().parents[1]/'results'/'tail_aware_summary_recomputed.csv',index=False);print(s.to_string(index=False))
