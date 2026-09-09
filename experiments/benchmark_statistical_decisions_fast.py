from __future__ import annotations
import argparse,sys
from pathlib import Path
import numpy as np,pandas as pd
from scipy.stats import t as student_t
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/'src'))
from benchmark_statistical_decisions import draw_effects,SEED
OUT=ROOT/'results'/'robust_extensions'; OUT.mkdir(parents=True,exist_ok=True)

def batch_metrics(X, *, boot=300, signs=500, bayes=500, alpha=.05, seed=0):
    # X: reps x n patient effects
    r,n=X.shape; mu=X.mean(1); sd=X.std(1,ddof=1); se=sd/np.sqrt(n)
    tstat=np.divide(mu,se,out=np.zeros_like(mu),where=se>1e-15); tpass=student_t.sf(tstat,n-1)<alpha
    rng=np.random.default_rng(seed)
    # sign-flip MC with +1 correction
    ge=np.zeros(r,int)
    for _ in range(signs):
        sg=rng.choice(np.array([-1.,1.]),size=(r,n)); ge += ((sg*X).mean(1)>=mu-1e-15)
    sfpass=((1+ge)/(1+signs))<alpha
    # bootstrap means/studentized statistics; keep only needed quantiles
    Mb=np.empty((boot,r)); Tb=np.empty((boot,r))
    for b in range(boot):
        idx=rng.integers(0,n,size=(r,n)); xb=np.take_along_axis(X,idx,axis=1); mb=xb.mean(1); seb=xb.std(1,ddof=1)/np.sqrt(n)
        Mb[b]=mb; Tb[b]=np.divide(mb-mu,seb,out=np.zeros_like(mu),where=seb>1e-15)
    pct=np.quantile(Mb,alpha,axis=0)>0
    q=np.quantile(Tb,1-alpha,axis=0); btlcb=mu-q*se; bt=btlcb>0
    # Bayesian bootstrap posterior Pr(mean>0)
    pos=np.zeros(r,int)
    for _ in range(bayes):
        g=rng.exponential(1.0,size=(r,n)); w=g/g.sum(1,keepdims=True); pos += ((w*X).sum(1)>0)
    bb=(pos/bayes)>=.95
    return tpass,sfpass,pct,bt,bb

def run(reps=1000,boot=300,signs=500,bayes=500,batch=100):
    scenarios=['null','adverse','weak','moderate','strong','skew_null','heavy_tail_null','contaminated_null']; ns=[10,15,20,30,50,100]
    rows=[]
    for si,sc in enumerate(scenarios):
      for n in ns:
        counts=np.zeros(5,int); total=0; effects=[]
        for st in range(0,reps,batch):
          b=min(batch,reps-st); X=np.vstack([draw_effects(np.random.default_rng(SEED+si*10_000_000+n*10_000+st+j),n,sc) for j in range(b)])
          m=batch_metrics(X,boot=boot,signs=signs,bayes=bayes,seed=SEED+si*100000+n*1000+st)
          counts += np.array([np.sum(z) for z in m]); total+=b; effects.append(X.mean(1))
        eff=float(np.concatenate(effects).mean())
        rows.append({'scenario':('null_effect' if sc=='null' else sc),'n':n,'reps':total,'mean_effect':eff,'t_rate':counts[0]/total,'signflip_rate':counts[1]/total,
                     'percentile_rate':counts[2]/total,'bootstrap_t_rate':counts[3]/total,'bayesian_bootstrap_rate':counts[4]/total})
        print(rows[-1],flush=True)
    d=pd.DataFrame(rows); d.to_csv(OUT/'statistical_benchmark_highrep_summary.csv',index=False); return d

if __name__=='__main__':
 ap=argparse.ArgumentParser(); ap.add_argument('--reps',type=int,default=1000); ap.add_argument('--bootstrap',type=int,default=300); ap.add_argument('--signs',type=int,default=500); ap.add_argument('--bayes',type=int,default=500); ap.add_argument('--batch',type=int,default=100); a=ap.parse_args(); run(a.reps,a.bootstrap,a.signs,a.bayes,a.batch)
