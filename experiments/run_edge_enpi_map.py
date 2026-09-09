from __future__ import annotations
import json,math
from pathlib import Path
import numpy as np,pandas as pd
from qualifyot import model
from qualifyot.model import feature_matrix,choose_null_loocv,fit_null,predict_null,mae
from qualifyot.weight_selection import patient_curves,one_se_msw
from refinement_candidate import GraphRefinementCandidate

JOBS=[
 ('GSE123813','GSE123813_pairs.csv',['Memory','Effector','Exhausted','Regulatory','Other_T'],[('Memory','Effector'),('Effector','Exhausted'),('Memory','Exhausted')]),
 ('GSE120575','GSE120575_pairs.csv',['Memory','Effector','Exhausted','Regulatory','Other_T'],[('Memory','Effector'),('Effector','Exhausted'),('Memory','Exhausted')]),
 ('GSE236581','GSE236581_pairs.csv',['Memory','Effector','Exhausted','Regulatory','Other_T'],[('Memory','Effector'),('Effector','Exhausted'),('Memory','Exhausted')]),
 ('GSE179994','GSE179994_pairs.csv',['Memory','Effector','Exhausted','Regulatory','Other_T'],[('Memory','Effector'),('Effector','Exhausted'),('Memory','Exhausted')]),
 ('GSE175522','GSE175522_D0_D7_Bcell_pairs.csv',['Memory','Effector','Exhausted','Regulatory','Other_T'],[('Memory','Effector'),('Effector','Exhausted'),('Memory','Exhausted')]),
 ('AML','GSE235063_AML_primary_pairs.csv',['HSC','Progenitor','GMP','Monocytic','Other'],[('Monocytic','GMP'),('GMP','Progenitor'),('Progenitor','HSC')]),
 ('HCC','GSE229772_HCC_primary_pairs.csv',['Tumor','Cytotoxic','Memory_Lymphoid','Immunoregulatory','Other_TME'],[('Memory_Lymphoid','Cytotoxic'),('Immunoregulatory','Cytotoxic'),('Tumor','Cytotoxic')]),
]

def fixed_graph_patient_losses(pairs,states,edges,seed=1,boot=500):
    pairs=pairs.reset_index(drop=True).copy();X,S,T=feature_matrix(pairs,states=states);P=pairs.patient_id.astype(str).to_numpy();grid=np.round(np.arange(0,1.00001,.01),2)
    shared={'flow':{},'log':[]};cand0=GraphRefinementCandidate(states,[edges],model,base_name='Fixed',alpha=1.0,selector='best',cv_folds=3,all_pairs=pairs,shared=shared)
    rec=[];fold=[]
    for j,h in enumerate(np.unique(P)):
        te=P==h;tr=~te;train=pairs.loc[tr].reset_index(drop=True);test=pairs.loc[te].reset_index(drop=True)
        c=cand0.fresh().fit(train);pr=c.predict(test)
        nk,_=choose_null_loocv(S[tr],T[tr],P[tr],patient_balanced=True);nm=fit_null(nk,S[tr],T[tr],patients=P[tr],patient_balanced=True);nu=predict_null(nm,S[te])
        # inner patient OOF candidate/reference
        Tin=T[tr];Pin=P[tr];po=np.zeros_like(Tin);no=np.zeros_like(Tin);trainP=train.patient_id.astype(str).to_numpy()
        for hh in np.unique(trainP):
            ite=trainP==hh;itr=~ite
            cc=cand0.fresh().fit(train.loc[itr].reset_index(drop=True));po[ite]=cc.predict(train.loc[ite].reset_index(drop=True))
            # map train matrices directly to inner mask
            Xtr,Str,Ttr=feature_matrix(train,states=states)
            nk2,_=choose_null_loocv(Str[itr],Ttr[itr],trainP[itr],patient_balanced=True) if len(np.unique(trainP[itr]))>=4 else ('Persistence',{})
            nm2=fit_null(nk2,Str[itr],Ttr[itr],patients=trainP[itr],patient_balanced=True);no[ite]=predict_null(nm2,Str[ite])
        _,D,R=patient_curves(Tin,no,po,Pin,grid);g=one_se_msw(D,R,grid,.05,0,boot,seed+j*7919,1)
        final=(1-g['weight'])*nu+g['weight']*pr
        for loc,ix in enumerate(np.flatnonzero(te)):
            rec.append({'patient_id':P[ix],'pair_id':pairs.pair_id.iloc[ix],'ref_loss':float(mae(T[ix:ix+1],nu[loc:loc+1])[0]),'cand_loss':float(mae(T[ix:ix+1],pr[loc:loc+1])[0]),'ret_loss':float(mae(T[ix:ix+1],final[loc:loc+1])[0]),'weight':g['weight']})
        fold.append({'patient_id':h,'weight':g['weight'],'reference_kind':nk})
    d=pd.DataFrame(rec)
    per=d.groupby('patient_id').agg(ref_loss=('ref_loss','mean'),cand_loss=('cand_loss','mean'),ret_loss=('ret_loss','mean')).reset_index()
    return per,pd.DataFrame(fold)

def simultaneous_max_t(C,B=5000,seed=1,alpha=.05):
    C=np.asarray(C,float);n,m=C.shape;theta=C.mean(0);se=C.std(0,ddof=1)/np.sqrt(n);rng=np.random.default_rng(seed);maxabs=[]
    for _ in range(B):
        z=C[rng.integers(0,n,size=n)];mb=z.mean(0);seb=z.std(0,ddof=1)/np.sqrt(n);t=np.zeros(m)
        ok=seb>1e-12;t[ok]=(mb[ok]-theta[ok])/seb[ok]
        maxabs.append(np.max(np.abs(t[ok])) if ok.any() else 0.0)
    q=float(np.quantile(maxabs,1-alpha));lo=theta-q*se;hi=theta+q*se
    return theta,se,q,lo,hi

allrows=[];patrows=[]
for di,(ds,fn,states,E) in enumerate(JOBS):
    repo=Path(__file__).resolve().parents[1]; p=pd.read_csv(repo/'data'/'processed_pairs'/fn)
    full,ff=fixed_graph_patient_losses(p,states,E,seed=20260825+di*10000,boot=500)
    full=full.set_index('patient_id')
    cols=[];names=[]
    for ei,e in enumerate(E):
        gm=[x for x in E if x!=e]
        alt,af=fixed_graph_patient_losses(p,states,gm,seed=20260825+di*10000+(ei+1)*1000,boot=500);alt=alt.set_index('patient_id').loc[full.index]
        # positive contribution means retaining e lowers retained loss vs deletion
        c=(alt.ret_loss-full.ret_loss).to_numpy(float);cols.append(c);names.append(f'{e[0]}->{e[1]}')
        for pid,val in zip(full.index,c):patrows.append({'dataset':ds,'patient_id':pid,'edge':names[-1],'ENPI_patient':val})
    C=np.column_stack(cols);theta,se,q,lo,hi=simultaneous_max_t(C,B=5000,seed=20260825+di*777)
    for j,name in enumerate(names):
        status='retention-supported' if lo[j]>0 else ('deletion-supported' if hi[j]<0 else 'unresolved')
        allrows.append({'dataset':ds,'n_patients':len(full),'edge':name,'ENPI_mean':theta[j],'ENPI_se':se[j],'sim_max_t_q95':q,'ENPI_sim_lo':lo[j],'ENPI_sim_hi':hi[j],'status':status,'full_mean_ret_loss':full.ret_loss.mean()})
    print('\n',ds);print(pd.DataFrame(allrows)[pd.DataFrame(allrows).dataset==ds].to_string(index=False),flush=True)
pd.DataFrame(allrows).to_csv(Path(__file__).resolve().parents[1]/'results'/'edge_enpi_summary_recomputed.csv',index=False)
pd.DataFrame(patrows).to_csv(Path(__file__).resolve().parents[1]/'results'/'edge_enpi_patient_contributions.csv',index=False)
