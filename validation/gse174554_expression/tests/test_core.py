import numpy as np
from qualifyot.model import project_flow,B,residual_flow,feature_matrix,fit_graphflow
from qualifyot.weight_selection import patient_curves,one_se_msw
from qualifyot.ontology import canonical_state
from qualifyot.robustness import exact_downsample_counts
from qualifyot.adapters import gse236581,gse123813

def test_simplex_projection():
    s=np.array([.6,.2,.1,.1,0]); f=np.array([1.,0.,0.]); _,p=project_flow(s,f); assert p.min()>=-1e-9 and abs(p.sum()-1)<1e-9

def test_residual_gcf_bounds():
    s=np.array([.5,.2,.1,.1,.1]); t=np.array([.4,.25,.15,.1,.1]); _,r,c=residual_flow(s,t); assert -1e-9<=c<=1+1e-9 and abs(r.sum())<1e-8

def test_one_se_zero_feasible():
    y=np.array([[.6,.2,.1,.1,0],[.5,.2,.1,.2,0]]); n=y.copy(); pr=np.roll(y,1,axis=1); pats=np.array(['a','b']); grid=np.array([0.,.5,1.]); _,D,R=patient_curves(y,n,pr,pats,grid); g=one_se_msw(D,R,grid,B=200,seed=1); assert g['weight']==0

def test_mapping():
    assert canonical_state('CD8 exhausted T')=='Exhausted'; assert canonical_state('Treg')=='Regulatory'; assert canonical_state('CD8 memory T')=='Memory'


def test_feature_matrix_time_parsing():
    import pandas as pd
    row={'source_time':'Baseline','target_time':'Follow Up 1'}
    for s in ['Memory','Effector','Exhausted','Regulatory','Other_T']:
        row[f'source__{s}']=0.2; row[f'target__{s}']=0.2
    X,_,_=feature_matrix(pd.DataFrame([row])); assert X.shape==(1,6) and abs(X[0,-1]-1.0)<1e-12
    row['source_time']='Follow Up 5'; row['target_time']='Follow Up 6'
    X,_,_=feature_matrix(pd.DataFrame([row])); assert abs(X[0,-1]-1.0)<1e-12


def test_cell_depth_100_percent_is_exact_no_smoothing():
    c=np.array([100,20,0,3,7]); out=exact_downsample_counts(c,1.0,np.random.default_rng(7))
    assert np.array_equal(out,c)
    assert np.allclose(out/out.sum(),c/c.sum())


def test_heldout_target_mutation_does_not_change_fit_prediction():
    # A held-out target may be present in the dataset object, but training uses only training rows.
    X=np.array([[.5,.2,.1,.1,.1,1.],[.4,.3,.1,.1,.1,1.],[.45,.25,.1,.1,.1,1.]],float)
    S=X[:,:5].copy()
    T=np.array([[.4,.3,.1,.1,.1],[.35,.35,.1,.1,.1],[.1,.1,.6,.1,.1]],float)
    flow=np.vstack([residual_flow(s,t)[0] for s,t in zip(S,T)])
    m1=fit_graphflow(X[:2],S[:2],T[:2],flow_labels=flow[:2]); q1=m1.predict(X[2:],S[2:])
    T2=T.copy(); T2[2]=np.array([.7,.05,.05,.1,.1])
    flow2=np.vstack([residual_flow(s,t)[0] for s,t in zip(S,T2)])
    m2=fit_graphflow(X[:2],S[:2],T2[:2],flow_labels=flow2[:2]); q2=m2.predict(X[2:],S[2:])
    assert np.allclose(q1,q2,atol=1e-12)


def test_gse236581_adapter_fixture(tmp_path):
    rows=[]
    for time in ['I','II']:
        for lab,n in [('Memory T',70),('Effector T',30),('Exhausted T',20)]:
            for i in range(n): rows.append({'barcode':f'{time}-{lab}-{i}','Patient':'P01','Treatment':time,'Tissue':'Tumor','MajorCellType':'T','SubCellType':lab,'Ident':f'P01-T-{time}'})
    path=tmp_path/'meta.tsv.gz'; __import__('pandas').DataFrame(rows).to_csv(path,sep='\t',index=False,compression='gzip')
    pairs,counts,audit=gse236581({'metadata':path},min_cells=100)
    assert pairs.patient_id.nunique()==1 and len(pairs)==1 and abs(pairs.iloc[0][[c for c in pairs if c.startswith('source__')]].sum()-1)<1e-12


def test_gse123813_adapter_fixture(tmp_path):
    rows=[]
    for sample in ['su001_pre','su001_post']:
        for lab,n in [('CD8 memory T',80),('CD8 exhausted T',40)]:
            for i in range(n): rows.append({'barcode':f'{sample}-{i}-{lab}','sample':sample,'cluster':lab})
    path=tmp_path/'bcc.tsv.gz'; __import__('pandas').DataFrame(rows).to_csv(path,sep='\t',index=False,compression='gzip')
    pairs,counts,audit=gse123813({'metadata':path},min_cells=100)
    assert pairs.patient_id.nunique()==1 and len(pairs)==1 and pairs.iloc[0].source_time=='pre' and pairs.iloc[0].target_time=='post'
