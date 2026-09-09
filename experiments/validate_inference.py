from __future__ import annotations
import argparse, json, math, time
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import t as student_t, norm

OUT=Path(__file__).resolve().parents[1]/'results'/'inference'
OUT.mkdir(parents=True,exist_ok=True)
ALPHA=.05
DELTA=.01


def _t_lcb_batch(x,alpha=.05):
    n=x.shape[1]; mu=x.mean(1); sd=x.std(1,ddof=1); se=sd/np.sqrt(n)
    return mu-student_t.ppf(1-alpha,n-1)*se

def _t_ci_batch(x,alpha=.05):
    n=x.shape[1]; mu=x.mean(1); sd=x.std(1,ddof=1); se=sd/np.sqrt(n); q=student_t.ppf(1-alpha/2,n-1)
    return mu-q*se,mu+q*se

def _draw_regime(rng, reps, n, regime, mu=0.0):
    if regime=='normal': x=rng.normal(size=(reps,n))
    elif regime=='heavy_t': x=rng.standard_t(5,size=(reps,n))/math.sqrt(5/3)
    elif regime=='skew':
        z=rng.lognormal(mean=0,sigma=.7,size=(reps,n)); x=(z-np.exp(.7**2/2))/math.sqrt((np.exp(.7**2)-1)*np.exp(.7**2))
    elif regime=='contamination':
        x=rng.normal(size=(reps,n)); m=rng.random((reps,n))<.05; x[m]=rng.normal(scale=5,size=m.sum())
        x=x/np.sqrt(.95+ .05*25)
    elif regime=='heteroskedastic':
        scale=np.where(rng.random((reps,n))<.5,.5,1.5); x=rng.normal(size=(reps,n))*scale/np.sqrt((.25+2.25)/2)
    else: raise ValueError(regime)
    return x+mu


def e1_iut_geometry(reps=10000,seed=20260901):
    rng=np.random.default_rng(seed); rows=[]
    ns=[8,9,12,20,40,80]; rhos=[-.4,0,.5,.9]; effects=[.25,.5,1.0]
    masks=[(0,1,1),(1,0,1),(1,1,0),(0,0,1),(0,1,0),(1,0,0),(0,0,0)]
    for n in ns:
      for rho in rhos:
        C=np.full((3,3),rho); np.fill_diagonal(C,1.0); L=np.linalg.cholesky(C)
        for eff in effects:
          for mask in masks:
            # mask 0=null boundary, 1=favorable alternative
            mu=np.array([eff if x else 0.0 for x in mask])
            hit=bonf=0; axis=np.zeros(3,int)
            done=0
            while done<reps:
                b=min(2000,reps-done); z=rng.normal(size=(b,n,3))@L.T+mu
                m=z.mean(1); sd=z.std(1,ddof=1); t=m/(sd/np.sqrt(n)); p=student_t.sf(t,n-1)
                pa=p<ALPHA; pb=p<ALPHA/3
                hit+=np.all(pa,1).sum(); bonf+=np.all(pb,1).sum(); axis+=pa.sum(0); done+=b
            rows.append(dict(n=n,rho=rho,effect=eff,null_face=''.join(map(str,mask)),reps=reps,
                             iut_pass=hit/reps,bonf_pass=bonf/reps,movement_pass=axis[0]/reps,utility_pass=axis[1]/reps,retention_pass=axis[2]/reps))
    d=pd.DataFrame(rows); d.to_csv(OUT/'E1_iut_complete_null_geometry.csv',index=False)
    return d


def _bca_lcb_from_boot(x,bootmeans,alpha=.05):
    n=len(x); theta=x.mean(); prop=(np.sum(bootmeans<theta)+.5*np.sum(bootmeans==theta))/len(bootmeans); eps=.5/len(bootmeans); prop=np.clip(prop,eps,1-eps); z0=norm.ppf(prop)
    jk=np.array([(x.sum()-x[i])/(n-1) for i in range(n)]); jb=jk.mean(); den=6*(np.sum((jb-jk)**2)**1.5); acc=0 if den<1e-30 else np.sum((jb-jk)**3)/den
    za=norm.ppf(alpha); adj=norm.cdf(z0+(z0+za)/(1-acc*(z0+za)+1e-15)); return float(np.quantile(bootmeans,np.clip(adj,0,1)))

def _row_quantile_nearest(a,q):
    a=np.asarray(a,float); s=np.sort(a,axis=1)
    if np.ndim(q)==0:
        return np.quantile(a,float(q),axis=1)
    q=np.asarray(q,float); idx=np.clip(np.rint(q*(a.shape[1]-1)).astype(int),0,a.shape[1]-1)
    return s[np.arange(len(s)),idx]

def _bca_lcb_batch(x,bootmeans,alpha=.05):
    reps,n=x.shape; B=bootmeans.shape[1]; theta=x.mean(1)
    prop=((bootmeans<theta[:,None]).sum(1)+.5*(bootmeans==theta[:,None]).sum(1))/B; eps=.5/B; prop=np.clip(prop,eps,1-eps); z0=norm.ppf(prop)
    jk=(x.sum(1)[:,None]-x)/(n-1); jb=jk.mean(1)[:,None]; num=np.sum((jb-jk)**3,axis=1); den=6*np.sum((jb-jk)**2,axis=1)**1.5; acc=np.divide(num,den,out=np.zeros_like(num),where=den>1e-30)
    za=norm.ppf(alpha); adj=norm.cdf(z0+(z0+za)/(1-acc*(z0+za)+1e-15)); return _row_quantile_nearest(bootmeans,np.clip(adj,0,1))

def e2_movement_stress(reps=500,boot=500,seed=20260902):
    rng=np.random.default_rng(seed); rows=[]
    ns=[9,20,40,80]; bmeans=[1e-4,5e-4,.001,.002,.005,.01,.05]
    for obs in ['clean','fixed_measurement_noise']:
      for truth,ratio_true in [('boundary',DELTA),('alternative_2x',2*DELTA)]:
       for n in ns:
        for b0 in bmeans:
          counts={k:0 for k in ['ratio_2s','ratio_1s','contrast_t','contrast_pct','contrast_bca']}; den_tiny=0; done=0
          while done<reps:
            q=min(100,reps-done)
            btrue=b0*2*rng.beta(8,8,size=(q,n)); atrue=ratio_true*btrue*2*rng.beta(8,8,size=(q,n))
            if obs=='clean': a,b=atrue,btrue
            else:
                a=np.clip(atrue+rng.normal(0,5e-4,size=(q,n)),0,1); b=np.clip(btrue+rng.normal(0,5e-4,size=(q,n)),0,1)
            den_tiny += np.sum(b.mean(1)<1e-5); g=a-DELTA*b
            idx=rng.integers(0,n,size=(q,boot,n)); ab=np.take_along_axis(a[:,None,:],idx,axis=2).mean(2); bb=np.take_along_axis(b[:,None,:],idx,axis=2).mean(2); gb=np.take_along_axis(g[:,None,:],idx,axis=2).mean(2)
            rat=ab/np.maximum(bb,1e-15)
            counts['ratio_2s'] += np.sum(np.quantile(rat,.025,axis=1)>DELTA)
            counts['ratio_1s'] += np.sum(np.quantile(rat,.05,axis=1)>DELTA)
            counts['contrast_pct'] += np.sum(np.quantile(gb,.05,axis=1)>0)
            counts['contrast_bca'] += np.sum(_bca_lcb_batch(g,gb,.05)>0)
            counts['contrast_t'] += np.sum(_t_lcb_batch(g,.05)>0)
            done+=q
          for method,v in counts.items(): rows.append(dict(observation=obs,truth=truth,n=n,denominator_mean=b0,reps=reps,bootstrap=boot,method=method,pass_rate=v/reps,tiny_denominator_rate=den_tiny/reps))
    d=pd.DataFrame(rows); d.to_csv(OUT/'E2_movement_denominator_stress.csv',index=False); return d

def e3_dependent_lopo(reps=1000,boot=400,seed=20260903):
    rng=np.random.default_rng(seed); rows=[]
    ns=[9,20,40,80]; regimes=['normal','heavy_t','skew','contamination','heteroskedastic']; gammas=[.5,1.0]
    for mu,label in [(0.0,'null'),(.15,'alternative')]:
      for n in ns:
       for regime in regimes:
        for gamma in gammas:
          acc={m:[0,0] for m in ['fixed_t','fixed_percentile','fixed_bootstrap_t','fixed_bca','full_pipeline_percentile','oracle_raw_t']}; theta=(1+gamma)*mu; done=0
          while done<reps:
            q=min(100,reps-done); x=_draw_regime(rng,q,n,regime,mu=mu); sm=x.sum(1)[:,None]; loo=(sm-x)/(n-1); psi=x+gamma*loo
            idx=rng.integers(0,n,size=(q,boot,n)); psib=np.take_along_axis(psi[:,None,:],idx,axis=2); pb=psib.mean(2)
            # fixed t
            lo,hi=_t_ci_batch(psi,.05); lcb=_t_lcb_batch(psi,.05); acc['fixed_t'][0]+=np.sum((lo<=theta)&(theta<=hi)); acc['fixed_t'][1]+=np.sum(lcb>0)
            # percentile
            qlo,qhi=np.quantile(pb,[.025,.975],axis=1); pl=np.quantile(pb,.05,axis=1); acc['fixed_percentile'][0]+=np.sum((qlo<=theta)&(theta<=qhi)); acc['fixed_percentile'][1]+=np.sum(pl>0)
            # bootstrap-t
            muhat=psi.mean(1); sd=psi.std(1,ddof=1); seb=psib.std(2,ddof=1)/np.sqrt(n); valid=seb>1e-12; ts=np.where(valid,(pb-muhat[:,None])/np.where(valid,seb,1),np.nan)
            # row quantiles ignoring vanishingly rare degenerate draws
            ts2=np.where(np.isfinite(ts),ts,np.nan)
            tqlo=np.nanquantile(ts2,.025,axis=1); tqhi=np.nanquantile(ts2,.975,axis=1); tq95=np.nanquantile(ts2,.95,axis=1); se=sd/np.sqrt(n); blo=muhat-tqhi*se; bhi=muhat-tqlo*se; bl=muhat-tq95*se
            acc['fixed_bootstrap_t'][0]+=np.sum((blo<=theta)&(theta<=bhi)); acc['fixed_bootstrap_t'][1]+=np.sum(bl>0)
            # BCa fixed scores
            b2lo=_bca_lcb_batch(psi,pb,.025); b2hi=-_bca_lcb_batch(-psi,-pb,.025); b2l=_bca_lcb_batch(psi,pb,.05); acc['fixed_bca'][0]+=np.sum((b2lo<=theta)&(theta<=b2hi)); acc['fixed_bca'][1]+=np.sum(b2l>0)
            # full-pipeline percentile: resample raw patients then rebuild shared-training functional; mean functional simplifies algebraically
            xb=np.take_along_axis(x[:,None,:],idx,axis=2); mb=(1+gamma)*xb.mean(2); flo,fhi=np.quantile(mb,[.025,.975],axis=1); fl=np.quantile(mb,.05,axis=1); acc['full_pipeline_percentile'][0]+=np.sum((flo<=theta)&(theta<=fhi)); acc['full_pipeline_percentile'][1]+=np.sum(fl>0)
            # stability-aware algebraic t benchmark
            qv=student_t.ppf(.975,n-1); est=(1+gamma)*x.mean(1); se2=(1+gamma)*x.std(1,ddof=1)/np.sqrt(n); olo=est-qv*se2; ohi=est+qv*se2; ol=est-student_t.ppf(.95,n-1)*se2; acc['oracle_raw_t'][0]+=np.sum((olo<=theta)&(theta<=ohi)); acc['oracle_raw_t'][1]+=np.sum(ol>0)
            done+=q
          for method,(cov,pos) in acc.items(): rows.append(dict(truth=label,n=n,regime=regime,gamma=gamma,reps=reps,bootstrap=boot,method=method,coverage=cov/reps,positive_rate=pos/reps,true_mean=theta))
    d=pd.DataFrame(rows); d.to_csv(OUT/'E3_dependent_lopo_inference.csv',index=False); return d

def _hetero_lcb_batch(x,width,alpha=.05):
    # x reps x n, width reps x n or scalar
    n=x.shape[1]
    if np.ndim(width)==0: s=n*float(width)**2
    else: s=np.sum(np.asarray(width)**2,axis=1)
    rad=np.sqrt(s*np.log(1/alpha)/(2*n*n)); return x.mean(1)-rad

def e4_honest_confirmation(reps=10000,seed=20260904):
    rng=np.random.default_rng(seed); rows=[]; ns=[10,20,30,50,100,200]
    # Independent confirmation-patient contributions with envelopes frozen
    # before outcome inspection.  Movement's envelope is [A-delta,A].
    for truth in ['all_null','movement_null','utility_null','retention_null','all_alternative']:
      for n in ns:
        movement_is_null=truth in {'all_null','movement_null'}
        A=.005 if movement_is_null else .03
        gm_mean=0.0 if movement_is_null else .025
        u_mean=0.0 if truth in {'all_null','utility_null'} else .03
        r_mean=0.0 if truth in {'all_null','retention_null'} else .025
        cu=.08; cr=.08
        def twopoint(lo,hi,mean):
            if not (lo-1e-12 <= mean <= hi+1e-12): raise RuntimeError((lo,mean,hi))
            p=(mean-lo)/(hi-lo); return np.where(rng.random((reps,n))<p,hi,lo)
        gm=twopoint(A-DELTA,A,gm_mean); u=twopoint(-cu,cu,u_mean); r=twopoint(-cr,cr,r_mean)
        ml=_hetero_lcb_batch(gm,DELTA); ul=_hetero_lcb_batch(u,2*cu); rl=_hetero_lcb_batch(r,2*cr)
        q=(ml>0)&(ul>0)&(rl>0)
        # Broad-support Hoeffding comparator using only universal loss ranges.
        mb=_hetero_lcb_batch(gm,1+DELTA); ub=_hetero_lcb_batch(u,2.0); rb=_hetero_lcb_batch(r,2.0); qb=(mb>0)&(ub>0)&(rb>0)
        rows.append(dict(truth=truth,n=n,reps=reps,localized_pass=q.mean(),broad_support_pass=qb.mean(),movement_pass=(ml>0).mean(),utility_pass=(ul>0).mean(),retention_pass=(rl>0).mean(),movement_A=A))
    d=pd.DataFrame(rows); d.to_csv(OUT/'E4_honest_confirmation_finite_sample.csv',index=False); return d

def summarize(e1,e2,e3,e4):
    lines=['# Margin-contrast inference executed results','',f'Generated: {pd.Timestamp.utcnow().isoformat()}','']
    null=e1[e1.null_face!='111']; lines += [f"E1 maximum IUT pass over expanded global-null geometry: {null.iut_pass.max():.4f}.",f"E1 maximum component-Bonferroni pass over same null geometry: {null.bonf_pass.max():.4f}.",'']
    e2b=e2[(e2.truth=='boundary')&(e2.observation=='clean')]; lines.append('E2 clean latent-boundary maximum false-pass rates by method:')
    for m,g in e2b.groupby('method'): lines.append(f'- {m}: {g.pass_rate.max():.4f}')
    lines.append('')
    e3n=e3[e3.truth=='null']; lines.append('E3 mean null positive rates across dependent-LOPO stress cells:')
    for m,g in e3n.groupby('method'): lines.append(f'- {m}: {g.positive_rate.mean():.4f}; mean 95% coverage {g.coverage.mean():.4f}')
    lines.append('')
    e4n=e4[e4.truth!='all_alternative']; lines += [f"E4 localized honest-confirmation maximum false qualification: {e4n.localized_pass.max():.4f}.",f"E4 broad-support Hoeffding maximum false qualification: {e4n.broad_support_pass.max():.4f}.",f"E4 localized power at n=100: {e4[(e4.truth=='all_alternative')&(e4.n==100)].localized_pass.iloc[0]:.4f}.",'']
    (OUT/'README_EXECUTED_RESULTS.md').write_text('\n'.join(lines),encoding='utf-8')


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--quick',action='store_true'); args=ap.parse_args()
    t=time.time()
    e1=e1_iut_geometry(reps=2000 if args.quick else 10000)
    e2=e2_movement_stress(reps=150 if args.quick else 500,boot=200 if args.quick else 500)
    e3=e3_dependent_lopo(reps=250 if args.quick else 1000,boot=200 if args.quick else 400)
    e4=e4_honest_confirmation(reps=2000 if args.quick else 10000)
    summarize(e1,e2,e3,e4)
    meta={'seconds':time.time()-t,'quick':args.quick,'seed_base':20260901}
    (OUT/'execution_metadata.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    print(json.dumps(meta,indent=2))

if __name__=='__main__': main()
