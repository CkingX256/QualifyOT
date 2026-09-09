from __future__ import annotations

import argparse
from pathlib import Path
import sys
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from qualifyot.benchmarks import benchmark_patient_effects

OUT=ROOT/'results'/'robust_extensions'; OUT.mkdir(parents=True,exist_ok=True)
SEED=20260830


def draw_effects(rng,n,scenario):
    sd=.02
    if scenario=='null': return rng.normal(0,sd,n)
    if scenario=='adverse': return rng.normal(-.01,sd,n)
    if scenario=='weak': return rng.normal(.004,sd,n)
    if scenario=='moderate': return rng.normal(.010,sd,n)
    if scenario=='strong': return rng.normal(.020,sd,n)
    if scenario=='skew_null':
        x=rng.exponential(sd,n)-sd; return x
    if scenario=='heavy_tail_null':
        x=rng.standard_t(3,n); return x/np.sqrt(3)*sd
    if scenario=='contaminated_null':
        x=rng.normal(0,sd,n); u=rng.random(n); x[u<.025]+=0.08; x[(u>=.025)&(u<.05)]-=0.08; return x
    raise KeyError(scenario)


def run(reps=200,bootstrap=300,permutations=500,bayes_draws=500):
    scenarios=['null','adverse','weak','moderate','strong','skew_null','heavy_tail_null','contaminated_null']
    ns=[10,15,20,30,50,100]
    rows=[]
    for si,scenario in enumerate(scenarios):
        for n in ns:
            for rep in range(reps):
                rng=np.random.default_rng(SEED+si*10_000_000+n*10_000+rep)
                x=draw_effects(rng,n,scenario)
                b=benchmark_patient_effects(x,bootstrap=bootstrap,permutations=permutations,bayes_draws=bayes_draws,
                    signflip_exact_max_n=0,seed=SEED+rep+n*1000+si*100000)
                rows.append({'scenario':('null_effect' if scenario=='null' else scenario),'n':n,'rep':rep,'mean_effect':x.mean(),'t_pass':b.t_pass,'signflip_pass':b.signflip_pass,
                    'percentile_pass':b.percentile_pass,'bootstrap_t_pass':b.bootstrap_t_pass,'bayesian_bootstrap_pass':b.bayesian_bootstrap_pass})
    d=pd.DataFrame(rows); d.to_csv(OUT/'statistical_benchmark_runs.csv',index=False)
    s=d.groupby(['scenario','n']).agg(reps=('rep','count'),mean_effect=('mean_effect','mean'),t_rate=('t_pass','mean'),signflip_rate=('signflip_pass','mean'),
        percentile_rate=('percentile_pass','mean'),bootstrap_t_rate=('bootstrap_t_pass','mean'),bayesian_bootstrap_rate=('bayesian_bootstrap_pass','mean')).reset_index()
    s.to_csv(OUT/'statistical_benchmark_summary.csv',index=False)
    return s

if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--reps',type=int,default=200); ap.add_argument('--bootstrap',type=int,default=300); ap.add_argument('--permutations',type=int,default=500); ap.add_argument('--bayes-draws',type=int,default=500)
    a=ap.parse_args(); print(run(a.reps,a.bootstrap,a.permutations,a.bayes_draws).to_string(index=False))
