from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
sys.path.insert(0,str(Path(__file__).resolve().parent))

from qualifyot.model import feature_matrix, incidence, fit_graphflow, mae
from qualifyot.safe_averaging import safe_graph_average
from qualifyot.robust_averaging import robust_graph_average
from qualifyot.edge_shrinkage import ContinuousEdgeShrinkageCandidate
from qualifyot.bayesian_evidence import bayesian_bootstrap_graph_profile
from qualifyot.partial_identification import graph_confidence_set

import synthetic_overcomplete_refinement as sor
import synthetic_subgroup_refinement as ssr

OUT=ROOT/'results'/'robust_extensions'
OUT.mkdir(parents=True,exist_ok=True)
SEED=20260830


def _fit_graphs_full(design, confirm, graphs):
    Xd,Sd,Td=feature_matrix(design,states=sor.states); Pd=design.patient_id.astype(str).to_numpy()
    Xc,Sc,Tc=feature_matrix(confirm,states=sor.states)
    preds=[]
    for g in graphs:
        F=sor.flow_labels(design,g); B=incidence(sor.states,g)
        m=fit_graphflow(Xd,Sd,Td,alpha=1.0,flow_labels=F,Bmat=B,edges=g,patients=Pd,patient_balanced=True)
        preds.append(m.predict(Xc,Sc))
    return Tc,Sc,np.stack(preds,axis=1)


def _patient_losses(target,pred,patients):
    l=mae(target,pred); p=np.asarray(patients).astype(str)
    return np.asarray([l[p==x].mean() for x in np.unique(p)],float)


def experiment_robust_sga(reps=50, bootstrap=500):
    graphs=[sor.overcomplete,sor.true_graph,(sor.D1,sor.D2,sor.D3),tuple(e for i,e in enumerate(sor.true_graph) if i!=1)]
    frozen_index=0
    rows=[]
    for frac_i,frac in enumerate([0.0,0.2,0.3]):
        for rep in range(reps):
            rng=np.random.default_rng(SEED+frac_i*100000+rep)
            design=ssr.make_mix(rng,60,frac,conc=450); confirm=ssr.make_mix(rng,80,frac,conc=450)
            _,_,Td=feature_matrix(design,states=sor.states); Pd=design.patient_id.astype(str).to_numpy()
            oof=[]
            for g in graphs:
                F=sor.flow_labels(design,g); oof.append(sor.cv_predictions(design,g,F,k=5))
            oof=np.stack(oof,axis=1)
            normal=safe_graph_average(Td,oof,Pd,frozen_index=frozen_index,step=.25,bootstrap=bootstrap,seed=SEED+rep+11,one_se=True)
            robust=robust_graph_average(Td,oof,Pd,frozen_index=frozen_index,step=.25,bootstrap=bootstrap,seed=SEED+rep+23,
                                        tail_fraction=.20,tail_harm_margin=0.0,one_se=True)
            Tc,Sc,pred=_fit_graphs_full(design,confirm,graphs)
            sub=confirm.reverse_subgroup.to_numpy(int); Pc=confirm.patient_id.astype(str).to_numpy()
            for method,sel in [('SGA',normal),('RobustSGA',robust)]:
                q=np.einsum('m,rms->rs',sel.weights,pred)
                risk=float(mae(Tc,q).mean()); fr=float(mae(Tc,pred[:,frozen_index]).mean())
                h=mae(Tc,q)-mae(Tc,pred[:,frozen_index]); n_tail=max(1,int(np.ceil(.20*len(h))))
                rows.append({'frac_reverse':frac,'rep':rep,'method':method,'risk':risk,'frozen_risk':fr,'delta_risk':risk-fr,
                             'fallback':bool(sel.fallback_used),'frozen_weight':float(sel.weights[frozen_index]),
                             'tail_cvar20_harm_vs_frozen':float(np.sort(h)[-n_tail:].mean()),
                             'harmed_fraction_vs_frozen':float(np.mean(h>0)),
                             'reverse_harmed_fraction_vs_frozen':float(np.mean(h[sub==1]>0)) if sub.sum() else np.nan,
                             'reverse_mean_harm_vs_frozen':float(np.mean(h[sub==1])) if sub.sum() else np.nan,
                             'majority_mean_harm_vs_frozen':float(np.mean(h[sub==0])),
                             'selection_mean_ucb':float(sel.safety_ucb if method=='SGA' else sel.mean_harm_ucb),
                             'selection_tail_ucb':np.nan if method=='SGA' else float(sel.tail_harm_ucb)})
    d=pd.DataFrame(rows); d.to_csv(OUT/'robust_sga_subgroup_runs.csv',index=False)
    s=d.groupby(['frac_reverse','method']).agg(reps=('rep','count'),mean_risk=('risk','mean'),mean_delta_risk=('delta_risk','mean'),
        fallback_rate=('fallback','mean'),mean_frozen_weight=('frozen_weight','mean'),mean_cvar20=('tail_cvar20_harm_vs_frozen','mean'),
        harmed_fraction=('harmed_fraction_vs_frozen','mean'),reverse_harmed_fraction=('reverse_harmed_fraction_vs_frozen','mean'),
        reverse_mean_harm=('reverse_mean_harm_vs_frozen','mean')).reset_index()
    s.to_csv(OUT/'robust_sga_subgroup_summary.csv',index=False)
    return s


def experiment_edge_shrinkage(reps=30):
    rows=[]
    for regime_i,regime in enumerate(['true','null','adverse']):
        for rep in range(reps):
            rng=np.random.default_rng(SEED+500000+regime_i*10000+rep)
            design=sor.make_df(rng,45,regime,conc=450); confirm=sor.make_df(rng,60,regime,conc=450)
            cand=ContinuousEdgeShrinkageCandidate(sor.states,sor.overcomplete,gamma_grid=(0,.001,.003,.01,.03),cv_folds=5,seed=SEED+rep)
            cand.fit(design); q=cand.predict(confirm)
            # frozen over-complete comparator fitted to the same design patients
            Xd,Sd,Td=feature_matrix(design,states=sor.states); Pd=design.patient_id.astype(str).to_numpy(); F=sor.flow_labels(design,sor.overcomplete)
            gm=fit_graphflow(Xd,Sd,Td,alpha=1.0,flow_labels=F,Bmat=incidence(sor.states,sor.overcomplete),edges=sor.overcomplete,patients=Pd,patient_balanced=True)
            Xc,Sc,Tc=feature_matrix(confirm,states=sor.states); f=gm.predict(Xc,Sc)
            w=cand.edge_weights()
            rows.append({'regime':regime,'rep':rep,'shrink_risk':float(mae(Tc,q).mean()),'frozen_risk':float(mae(Tc,f).mean()),
                         'delta_risk':float(mae(Tc,q).mean()-mae(Tc,f).mean()),'selected_gamma':cand.selected_gamma,
                         'active_edges':int(sum(v>.05 for v in w.values())),'edge_weight_sum':float(sum(w.values())),**{f'w__{k}':v for k,v in w.items()}})
    d=pd.DataFrame(rows); d.to_csv(OUT/'continuous_edge_shrinkage_runs.csv',index=False)
    s=d.groupby('regime').agg(reps=('rep','count'),mean_shrink_risk=('shrink_risk','mean'),mean_frozen_risk=('frozen_risk','mean'),
                              mean_delta_risk=('delta_risk','mean'),mean_active_edges=('active_edges','mean'),mean_edge_weight_sum=('edge_weight_sum','mean')).reset_index()
    s.to_csv(OUT/'continuous_edge_shrinkage_summary.csv',index=False)
    return s


def experiment_graph_bayesian_profile(reps=40,bootstrap=500,draws=5000):
    rows=[]
    graph_names=[sor.gkey(g) for g in sor.graphs]
    graph_edges={sor.gkey(g):g for g in sor.graphs}
    for n in [20,40,80]:
        for rep in range(reps):
            rng=np.random.default_rng(SEED+800000+n*1000+rep)
            d=sor.make_df(rng,n,'true',conc=450)
            _,_,T=feature_matrix(d,states=sor.states); P=d.patient_id.astype(str).to_numpy()
            losses=[]
            for g in sor.graphs:
                F=sor.flow_labels(d,g); pr=sor.cv_predictions(d,g,F,k=5); losses.append(mae(T,pr))
            L=np.column_stack(losses)
            cs=graph_confidence_set(L,P,graph_names,graph_edges=graph_edges,bootstrap=bootstrap,seed=SEED+rep)
            bb=bayesian_bootstrap_graph_profile(L,P,graph_names,graph_edges=graph_edges,draws=draws,seed=SEED+rep+1)
            true_name=sor.gkey(sor.true_graph); mt=bb.model_table.set_index('model')
            rows.append({'n':n,'rep':rep,'truth_in_confidence_set':true_name in cs.members,'set_size':len(cs.members),
                         'bb_truth_probability_best':float(mt.loc[true_name,'bb_probability_best']),
                         'bb_top_model':str(bb.model_table.iloc[0]['model']),'bb_truth_top':str(bb.model_table.iloc[0]['model'])==true_name,
                         'core_edges':len(cs.core_edges),'optional_edges':len(cs.optional_edges),'excluded_edges':len(cs.excluded_edges)})
    d=pd.DataFrame(rows); d.to_csv(OUT/'graph_bayesian_profile_runs.csv',index=False)
    s=d.groupby('n').agg(reps=('rep','count'),truth_coverage=('truth_in_confidence_set','mean'),median_set_size=('set_size','median'),
                         mean_bb_truth_probability=('bb_truth_probability_best','mean'),bb_truth_top_rate=('bb_truth_top','mean'),
                         mean_core_edges=('core_edges','mean'),mean_excluded_edges=('excluded_edges','mean')).reset_index()
    s.to_csv(OUT/'graph_bayesian_profile_summary.csv',index=False)
    return s


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--reps',type=int,default=30); ap.add_argument('--bootstrap',type=int,default=500); ap.add_argument('--draws',type=int,default=5000)
    ap.add_argument('--suite',choices=['robust-sga','edge-shrink','graph-bayes','all'],default='all'); a=ap.parse_args()
    if a.suite in ('robust-sga','all'):
        print(experiment_robust_sga(a.reps,a.bootstrap).to_string(index=False),flush=True)
    if a.suite in ('edge-shrink','all'):
        print(experiment_edge_shrinkage(max(10,a.reps//2)).to_string(index=False),flush=True)
    if a.suite in ('graph-bayes','all'):
        print(experiment_graph_bayesian_profile(max(10,a.reps//2),a.bootstrap,a.draws).to_string(index=False),flush=True)

if __name__=='__main__': main()
