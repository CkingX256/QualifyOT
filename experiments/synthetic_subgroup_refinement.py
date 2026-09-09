from __future__ import annotations
from pathlib import Path
import sys,math,hashlib
import numpy as np,pandas as pd
from sklearn.metrics import roc_auc_score
from synthetic_overcomplete_refinement import states,true_graph,overcomplete,graphs,gkey,select_graph,flow_labels,cv_predictions
from qualifyot import model,experiment
from qualifyot.model import feature_matrix,incidence,fit_graphflow,mae,hellinger
from qualifyot.weight_selection import patient_curves,one_se_msw

def make_mix(rng,n,frac,conc=450):
    S=rng.dirichlet([4,3,2,2.5,2.5],size=n);M=S.copy();sub=np.zeros(n,dtype=bool);m=int(round(frac*n));
    if m>0: sub[rng.choice(n,m,replace=False)]=True
    # majority true: M->E, E->X, R->O
    idx=~sub
    f1=0.28*S[idx,0];f2=0.24*S[idx,1];f3=0.22*S[idx,3]
    M[idx,0]-=f1;M[idx,1]+=f1;M[idx,1]-=f2;M[idx,2]+=f2;M[idx,3]-=f3;M[idx,4]+=f3
    # subgroup exact reverse directions
    idx=sub
    if idx.any():
      f1=0.22*S[idx,1];f2=0.20*S[idx,2];f3=0.18*S[idx,4]
      M[idx,1]-=f1;M[idx,0]+=f1;M[idx,2]-=f2;M[idx,1]+=f2;M[idx,4]-=f3;M[idx,3]+=f3
    M=np.clip(M,1e-8,None);M/=M.sum(1,keepdims=True);T=np.vstack([rng.dirichlet(np.maximum(x*conc,0.05)) for x in M])
    d={'patient_id':[f'P{i:03d}' for i in range(n)],'pair_id':[f'P{i:03d}:0->1' for i in range(n)],'source_time':np.zeros(n),'target_time':np.ones(n),'reverse_subgroup':sub.astype(int)}
    for j,s in enumerate(states):d[f'source__{s}']=S[:,j];d[f'target__{s}']=T[:,j]
    return pd.DataFrame(d)

def eval_rep(seed,frac,n=60,boot=300):
    rng=np.random.default_rng(seed);design=make_mix(rng,n,frac);confirm=make_mix(rng,n,frac)
    selected,best,rows,caches=select_graph(design)
    Xd,Sd,Td=feature_matrix(design,states=states);Pd=design.patient_id.astype(str).to_numpy();F,po=caches[selected];grid=np.round(np.arange(0,1.00001,.01),2)
    _,D,R=patient_curves(Td,Sd,po,Pd,grid);w=one_se_msw(D,R,grid,.05,0,300,seed+501,1)['weight']
    B=incidence(states,selected);gm=fit_graphflow(Xd,Sd,Td,alpha=1.0,flow_labels=F,Bmat=B,edges=selected,patients=Pd,patient_balanced=True)
    Xc,Sc,Tc=feature_matrix(confirm,states=states);Pc=confirm.patient_id.astype(str).to_numpy();C=gm.predict(Xc,Sc);A=(1-w)*Sc+w*C
    ref_loss=mae(Tc,Sc);ret_loss=mae(Tc,A);npi_i=ref_loss-ret_loss;sub=confirm.reverse_subgroup.to_numpy(int)
    pdr=experiment.bootstrap_ratio(hellinger(C,Sc),hellinger(Tc,Sc),Pc,boot,seed+601);puc=experiment.bootstrap_contrast(ref_loss-mae(Tc,C),Pc,boot,seed+602);npi=experiment.bootstrap_contrast(npi_i,Pc,boot,seed+603)
    retained_pass=bool(pdr[0]>.05 and pdr[1]>.01 and puc[1]>0 and w>0 and npi[1]>0 and ret_loss.mean()<ref_loss.mean())
    harm=(npi_i<0);auc=np.nan
    if len(np.unique(sub))==2:
      try:auc=roc_auc_score(sub,-npi_i)
      except:pass
    return {'seed':seed,'frac_reverse':frac,'n':n,'selected_graph':gkey(selected),'exact_true':selected==true_graph,'weight':w,'PUC':puc[0],'PUC_lo':puc[1],'NPI':npi[0],'NPI_lo':npi[1],'retained_pass':retained_pass,'harm_frac':harm.mean(),'harm_frac_reverse':harm[sub==1].mean() if sub.sum() else np.nan,'harm_frac_majority':harm[sub==0].mean(),'mean_npi_reverse':npi_i[sub==1].mean() if sub.sum() else np.nan,'mean_npi_majority':npi_i[sub==0].mean(),'discordance_auc_neg_npi':auc}

if __name__=='__main__':
  out=[]
  for fi,frac in enumerate([0,0.1,0.2,0.3,0.5]):
    for rep in range(20):out.append(eval_rep(20260825+fi*100000+rep*101,frac,n=60,boot=200))
    print('frac',frac,pd.DataFrame(out)[pd.DataFrame(out).frac_reverse==frac][['exact_true','retained_pass','harm_frac','harm_frac_reverse','harm_frac_majority','discordance_auc_neg_npi']].mean(numeric_only=True).to_dict(),flush=True)
  d=pd.DataFrame(out);d.to_csv(Path(__file__).resolve().parents[1]/'results'/'mechanism_mixture_runs.csv',index=False)
  s=d.groupby('frac_reverse').agg(reps=('seed','count'),exact_true_rate=('exact_true','mean'),retained_pass_rate=('retained_pass','mean'),mean_PUC=('PUC','mean'),mean_NPI=('NPI','mean'),harm_frac=('harm_frac','mean'),harm_frac_reverse=('harm_frac_reverse','mean'),harm_frac_majority=('harm_frac_majority','mean'),mean_npi_reverse=('mean_npi_reverse','mean'),mean_npi_majority=('mean_npi_majority','mean'),discordance_auc=('discordance_auc_neg_npi','mean')).reset_index();s.to_csv(Path(__file__).resolve().parents[1]/'results'/'mechanism_mixture_summary_recomputed.csv',index=False);print(s.to_string(index=False))
