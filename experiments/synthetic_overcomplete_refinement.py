from __future__ import annotations
import sys, hashlib, math, json, time
from pathlib import Path
import numpy as np, pandas as pd
from qualifyot import model, experiment
from qualifyot.model import feature_matrix, incidence, residual_flow, fit_graphflow, mae, hellinger, patient_sample_weights
from qualifyot.weight_selection import patient_curves, one_se_msw
from qualifyot.utils import normalize_rows

states=['Memory','Effector','Exhausted','Regulatory','Other_T']
T1=('Memory','Effector'); T2=('Effector','Exhausted'); T3=('Regulatory','Other_T')
D1=('Effector','Memory'); D2=('Exhausted','Effector'); D3=('Other_T','Regulatory')
true_graph=(T1,T2,T3); overcomplete=(T1,T2,T3,D1,D2,D3)
# 16 graph finite library: exact truth, all supersets of truth with decoys, one-true-edge omissions, and wrong sparse controls.
graphs=[]
def add(g):
    g=tuple(g)
    if g not in graphs: graphs.append(g)
add(true_graph)
for mask in range(1,8):
    add(tuple(true_graph)+tuple([d for j,d in enumerate([D1,D2,D3]) if mask&(1<<j)]))
for drop in range(3): add(tuple(e for j,e in enumerate(true_graph) if j!=drop))
add((D1,D2,D3)); add((T1,D2,T3)); add((D1,T2,T3)); add((T1,T2,D3)); add((T1,D1,T2,D2))
assert len(graphs)==16

def gkey(g): return '|'.join(f'{a}->{b}' for a,b in g)

def make_df(rng,n,regime='true',conc=450):
    S=rng.dirichlet([4,3,2,2.5,2.5],size=n)
    M=S.copy()
    if regime=='true':
        # nonredundant true flows, exactly representable by true_graph in expectation
        f1=0.28*S[:,0]; f2=0.24*S[:,1]; f3=0.22*S[:,3]
        M[:,0]-=f1; M[:,1]+=f1
        M[:,1]-=f2; M[:,2]+=f2
        M[:,3]-=f3; M[:,4]+=f3
    elif regime=='null':
        pass
    elif regime=='adverse':
        # dynamics outside the refinement library, so graph editing cannot legitimately repair the candidate
        f1=0.26*S[:,0] # Memory -> Regulatory
        f2=0.23*S[:,1] # Effector -> Other_T
        f3=0.20*S[:,2] # Exhausted -> Other_T
        M[:,0]-=f1; M[:,3]+=f1
        M[:,1]-=f2; M[:,4]+=f2
        M[:,2]-=f3; M[:,4]+=f3
    else: raise KeyError(regime)
    M=np.clip(M,1e-8,None); M=M/M.sum(1,keepdims=True)
    T=np.vstack([rng.dirichlet(np.maximum(m*conc,0.05)) for m in M])
    d={'patient_id':[f'P{i:03d}' for i in range(n)], 'pair_id':[f'P{i:03d}:0->1' for i in range(n)], 'source_time':np.zeros(n), 'target_time':np.ones(n)}
    for j,s in enumerate(states): d[f'source__{s}']=S[:,j]; d[f'target__{s}']=T[:,j]
    return pd.DataFrame(d)

def flow_labels(df,g):
    _,S,T=feature_matrix(df,states=states); B=incidence(states,g)
    return np.vstack([residual_flow(s,t,Bmat=B)[0] for s,t in zip(S,T)])

def cv_predictions(df,g,F,k=5):
    X,S,T=feature_matrix(df,states=states); P=df.patient_id.astype(str).to_numpy(); u=sorted(np.unique(P));
    uh=sorted(u,key=lambda x:hashlib.sha256(x.encode()).hexdigest()); fmap={p:j%k for j,p in enumerate(uh)}
    pred=np.zeros_like(T)
    B=incidence(states,g)
    for fold in range(k):
        te=np.array([fmap[x]==fold for x in P]); tr=~te
        gm=fit_graphflow(X[tr],S[tr],T[tr],alpha=1.0,flow_labels=F[tr],Bmat=B,edges=g,patients=P[tr],patient_balanced=True)
        pred[te]=gm.predict(X[te],S[te])
    return pred

def select_graph(design):
    X,S,T=feature_matrix(design,states=states); P=design.patient_id.astype(str).to_numpy()
    rows=[]; caches={}
    for g in graphs:
        F=flow_labels(design,g); pr=cv_predictions(design,g,F,k=5); caches[g]=(F,pr)
        per=[]
        loss=mae(T,pr)
        for p in np.unique(P): per.append(float(loss[P==p].mean()))
        per=np.asarray(per); rows.append({'g':g,'risk':float(per.mean()),'se':float(per.std(ddof=1)/math.sqrt(len(per))),'n_edges':len(g)})
    best=min(rows,key=lambda r:(r['risk'],r['n_edges'],gkey(r['g'])))
    threshold=best['risk']+best['se']
    eligible=[r for r in rows if r['risk']<=threshold+1e-15]
    chosen=min(eligible,key=lambda r:(r['n_edges'],r['risk'],gkey(r['g'])))
    return chosen['g'],best,rows,caches

def eval_rep(seed,regime,n_design=45,n_confirm=35,boot=800):
    rng=np.random.default_rng(seed); design=make_df(rng,n_design,regime); confirm=make_df(rng,n_confirm,regime)
    selected,best,rows,caches=select_graph(design)
    Xd,Sd,Td=feature_matrix(design,states=states); Pd=design.patient_id.astype(str).to_numpy()
    F,po=caches[selected]
    # Fixed reference for this controlled experiment: Persistence. Weight chosen on design OOF only.
    grid=np.round(np.arange(0,1.00001,.01),2); ref_oof=Sd.copy()
    _,D,R=patient_curves(Td,ref_oof,po,Pd,grid)
    w=one_se_msw(D,R,grid,.05,0,300,seed+501,1)['weight']
    B=incidence(states,selected); gm=fit_graphflow(Xd,Sd,Td,alpha=1.0,flow_labels=F,Bmat=B,edges=selected,patients=Pd,patient_balanced=True)
    Xc,Sc,Tc=feature_matrix(confirm,states=states); Pc=confirm.patient_id.astype(str).to_numpy(); C=gm.predict(Xc,Sc); A=(1-w)*Sc+w*C
    pdr=experiment.bootstrap_ratio(hellinger(C,Sc),hellinger(Tc,Sc),Pc,boot,seed+601)
    puc=experiment.bootstrap_contrast(mae(Tc,Sc)-mae(Tc,C),Pc,boot,seed+602)
    npi=experiment.bootstrap_contrast(mae(Tc,Sc)-mae(Tc,A),Pc,boot,seed+603)
    ref=float(mae(Tc,Sc).mean()); rr=float(mae(Tc,A).mean())
    retained_pass=bool(pdr[0]>.05 and pdr[1]>.01 and puc[1]>0 and w>0 and npi[1]>0 and rr<ref)
    # Original over-complete comparator, same design/confirmation split and independent scoring.
    Fo=flow_labels(design,overcomplete); pro=cv_predictions(design,overcomplete,Fo,k=5); _,Do,Ro=patient_curves(Td,Sd,pro,Pd,grid); wo=one_se_msw(Do,Ro,grid,.05,0,300,seed+701,1)['weight']
    Bo=incidence(states,overcomplete); gmo=fit_graphflow(Xd,Sd,Td,alpha=1.0,flow_labels=Fo,Bmat=Bo,edges=overcomplete,patients=Pd,patient_balanced=True); Co=gmo.predict(Xc,Sc); Ao=(1-wo)*Sc+wo*Co
    return {
      'seed':seed,'regime':regime,'selected_graph':gkey(selected),'selected_n_edges':len(selected),'exact_true':selected==true_graph,
      'best_graph':gkey(best['g']),'best_cv_risk':best['risk'],'selected_cv_risk':next(r['risk'] for r in rows if r['g']==selected),
      'weight':w,'PDR':pdr[0],'PDR_lo':pdr[1],'PUC':puc[0],'PUC_lo':puc[1],'NPI':npi[0],'NPI_lo':npi[1], 'retained_pass':retained_pass,
      'confirm_ref_mae':ref,'confirm_refined_mae':rr,'overcomplete_weight':wo,'overcomplete_mae':float(mae(Tc,Ao).mean()),
      'delta_mae_refined_minus_overcomplete':float(mae(Tc,A).mean()-mae(Tc,Ao).mean())}

def main():
    import argparse
    ap=argparse.ArgumentParser(); ap.add_argument('--reps',type=int,default=50); ap.add_argument('--boot',type=int,default=800); ap.add_argument('--out',default=str(Path(__file__).resolve().parents[1]/'results'/'synthetic_refinement_runs.csv')); a=ap.parse_args()
    out=[]
    for ri,reg in enumerate(['true','null','adverse']):
        for rep in range(a.reps):
            row=eval_rep(20260825+ri*100000+rep*101,reg,boot=a.boot); out.append(row)
            print(reg,rep,row['selected_graph'],row['exact_true'],row['PUC'],row['NPI'],row['retained_pass'],flush=True)
        pd.DataFrame(out).to_csv(a.out,index=False)
    df=pd.DataFrame(out)
    summ=df.groupby('regime').agg(reps=('seed','count'),exact_true_rate=('exact_true','mean'),retained_pass_rate=('retained_pass','mean'),mean_PUC=('PUC','mean'),mean_NPI=('NPI','mean'),mean_delta_mae_vs_overcomplete=('delta_mae_refined_minus_overcomplete','mean')).reset_index()
    summ.to_csv(a.out.replace('.csv','_summary.csv'),index=False); print('\nSUMMARY\n',summ.to_string(index=False))
if __name__=='__main__': main()