from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t as student_t

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "principled_validation"
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(Path(__file__).resolve().parent))

from qualifyot.decision import classify_iut
from qualifyot.partial_identification import graph_confidence_set
from qualifyot.safe_averaging import safe_graph_average
from qualifyot.tail_safety import tail_safety_profile
from qualifyot.replication_planning import empirical_t_lcb_power, bayesian_bootstrap_assurance
from qualifyot.generic_engine import run_generic_lopo
from qualifyot.candidates import RobustBlendCandidate
from qualifyot.candidate_api import GraphFlowCandidate
from qualifyot.graph_design import SafeGraphAveragingCandidate
from qualifyot.model import feature_matrix, incidence, residual_flow, fit_graphflow, mae

import synthetic_overcomplete_refinement as sor
import synthetic_subgroup_refinement as ssr

SEED = 20260830


def _t_lcb(x, margin=0.0, alpha=.05):
    x=np.asarray(x,float); n=len(x)
    if n<2: return -np.inf
    return float(x.mean()-student_t.ppf(1-alpha,n-1)*x.std(ddof=1)/math.sqrt(n)-margin)


def experiment_iut_calibration(reps=4000):
    rows=[]
    ns=[10,20,50]; rhos=[0.0,.5,.9]
    # Each union-null scenario has exactly one true component null at its margin;
    # the other two components have positive margins.
    means_map={
        'movement_null':np.array([0.0,.6,.6]),
        'utility_null':np.array([.6,0.0,.6]),
        'retention_null':np.array([.6,.6,0.0]),
        'all_alternative':np.array([.45,.45,.45]),
    }
    for n in ns:
        for rho in rhos:
            cov=(1-rho)*np.eye(3)+rho*np.ones((3,3))
            L=np.linalg.cholesky(cov)
            for scenario,mu in means_map.items():
                rng=np.random.default_rng(SEED+n*10000+int(rho*100)*100+sum(map(ord,scenario)))
                pass_iut=0; pass_bon=0
                for rep in range(reps):
                    X=rng.normal(size=(n,3))@L.T+mu
                    se=X.std(0,ddof=1)/math.sqrt(n)
                    m=X.mean(0)
                    crit=student_t.ppf(.95,n-1)  # one-sided 5%
                    lo=m-crit*se
                    iut=bool(np.all(lo>0))
                    crit_b=student_t.ppf(1-.05/3,n-1)
                    lo_b=m-crit_b*se
                    bon=bool(np.all(lo_b>0))
                    pass_iut+=iut; pass_bon+=bon
                rows.append({'n':n,'rho':rho,'scenario':scenario,'reps':reps,
                             'iut_pass_rate':pass_iut/reps,'bonferroni_component_pass_rate':pass_bon/reps})
    d=pd.DataFrame(rows); d.to_csv(OUT/'iut_calibration.csv',index=False)
    return d


def _graph_oof_losses(df):
    _,_,T=feature_matrix(df,states=sor.states); P=df.patient_id.astype(str).to_numpy()
    names=[]; edge_map={}; cols=[]; preds=[]
    for g in sor.graphs:
        F=sor.flow_labels(df,g); pr=sor.cv_predictions(df,g,F,k=5)
        names.append(sor.gkey(g)); edge_map[names[-1]]=g; preds.append(pr)
        loss=mae(T,pr)
        cols.append(np.array([loss[P==p].mean() for p in np.unique(P)],float))
    return np.column_stack(cols), np.unique(P), names, edge_map, np.stack(preds,axis=1)


def experiment_graph_confidence(reps=12):
    rows=[]
    for n in [20,40,80]:
        for rep in range(reps):
            rng=np.random.default_rng(SEED+100000+n*100+rep)
            df=sor.make_df(rng,n,'true',conc=450)
            L,P,names,edge_map,_=_graph_oof_losses(df)
            # L is already patient-level: use one row per patient.
            out=graph_confidence_set(L,P,names,graph_edges=edge_map,bootstrap=500,seed=SEED+n*1000+rep)
            truth=sor.gkey(sor.true_graph)
            core=set(out.core_edges); true_edges=set(sor.true_graph)
            rows.append({'n':n,'rep':rep,'set_size':len(out.members),'truth_in_set':truth in out.members,
                         'truth_unique':out.members==(truth,),
                         'core_true_edges':len(core & true_edges),'core_false_edges':len(core-true_edges),
                         'optional_edges':len(out.optional_edges),'excluded_edges':len(out.excluded_edges),
                         'critical_value':out.critical_value})
    d=pd.DataFrame(rows); d.to_csv(OUT/'graph_confidence_set_runs.csv',index=False)
    s=d.groupby('n').agg(reps=('rep','count'),truth_coverage=('truth_in_set','mean'),unique_truth_rate=('truth_unique','mean'),
                         median_set_size=('set_size','median'),mean_set_size=('set_size','mean'),
                         mean_core_true_edges=('core_true_edges','mean'),mean_core_false_edges=('core_false_edges','mean')).reset_index()
    s.to_csv(OUT/'graph_confidence_set_summary.csv',index=False)
    return s


def _fit_graphs_full(design,confirm,graphs):
    Xd,Sd,Td=feature_matrix(design,states=sor.states); Pd=design.patient_id.astype(str).to_numpy()
    Xc,Sc,Tc=feature_matrix(confirm,states=sor.states)
    preds=[]
    for g in graphs:
        B=incidence(sor.states,g)
        F=np.vstack([residual_flow(s,t,Bmat=B)[0] for s,t in zip(Sd,Td)])
        m=fit_graphflow(Xd,Sd,Td,alpha=1.0,flow_labels=F,Bmat=B,edges=g,patients=Pd,patient_balanced=True)
        preds.append(m.predict(Xc,Sc))
    return Tc,Sc,np.stack(preds,axis=1)


def experiment_safe_averaging(reps=20):
    # Small pre-specified library: frozen over-complete graph, exact graph,
    # reverse-decoy graph, and a one-edge-omission graph.
    graphs=[sor.overcomplete,sor.true_graph,(sor.D1,sor.D2,sor.D3),tuple(e for i,e in enumerate(sor.true_graph) if i!=1)]
    frozen_index=0
    rows=[]
    for regime_i,regime in enumerate(['true','null','adverse']):
        for rep in range(reps):
            rng=np.random.default_rng(SEED+200000+regime_i*10000+rep)
            design=sor.make_df(rng,40,regime,conc=450); confirm=sor.make_df(rng,40,regime,conc=450)
            _,_,Td=feature_matrix(design,states=sor.states); Pd=design.patient_id.astype(str).to_numpy()
            oof=[]
            for g in graphs:
                F=sor.flow_labels(design,g); oof.append(sor.cv_predictions(design,g,F,k=5))
            oof=np.stack(oof,axis=1)
            sel=safe_graph_average(Td,oof,Pd,frozen_index=frozen_index,step=.25,bootstrap=500,seed=SEED+rep+regime_i*1000,one_se=True)
            Tc,Sc,pred=_fit_graphs_full(design,confirm,graphs)
            q=np.einsum('m,rms->rs',sel.weights,pred)
            frozen=pred[:,frozen_index]
            risk_q=float(mae(Tc,q).mean()); risk_f=float(mae(Tc,frozen).mean())
            rows.append({'regime':regime,'rep':rep,'frozen_weight':sel.weights[frozen_index],
                         'true_graph_weight':sel.weights[1],'n_active':int(np.count_nonzero(sel.weights>1e-12)),
                         'design_safety_ucb':sel.safety_ucb,'confirm_delta_vs_frozen':risk_q-risk_f,
                         'confirm_improved':risk_q<risk_f,'confirm_harm_gt_0p002':(risk_q-risk_f)>.002,
                         'confirm_risk_sga':risk_q,'confirm_risk_frozen':risk_f})
    d=pd.DataFrame(rows); d.to_csv(OUT/'safe_graph_averaging_runs.csv',index=False)
    s=d.groupby('regime').agg(reps=('rep','count'),mean_frozen_weight=('frozen_weight','mean'),mean_true_weight=('true_graph_weight','mean'),
                              improvement_rate=('confirm_improved','mean'),harm_gt_0p002_rate=('confirm_harm_gt_0p002','mean'),
                              mean_delta_vs_frozen=('confirm_delta_vs_frozen','mean')).reset_index()
    s.to_csv(OUT/'safe_graph_averaging_summary.csv',index=False)
    return s


def experiment_tail_safety(reps=20):
    rows=[]
    # Evaluate the frozen true graph under mechanism mixtures; use independent
    # design and confirmation so the safety diagnostic is not fitted on the
    # evaluated patients.
    for frac in [0,.1,.2,.3]:
        for rep in range(reps):
            rng=np.random.default_rng(SEED+300000+int(frac*100)*1000+rep)
            design=ssr.make_mix(rng,60,frac,conc=450); confirm=ssr.make_mix(rng,60,frac,conc=450)
            Xd,Sd,Td=feature_matrix(design,states=sor.states); Pd=design.patient_id.astype(str).to_numpy()
            B=incidence(sor.states,sor.true_graph); F=np.vstack([residual_flow(s,t,Bmat=B)[0] for s,t in zip(Sd,Td)])
            gm=fit_graphflow(Xd,Sd,Td,alpha=1.0,flow_labels=F,Bmat=B,edges=sor.true_graph,patients=Pd,patient_balanced=True)
            Xc,Sc,Tc=feature_matrix(confirm,states=sor.states); Pc=confirm.patient_id.astype(str).to_numpy()
            C=gm.predict(Xc,Sc)
            prof=tail_safety_profile(Tc,Sc,C,Pc,tail_fraction=.2,bootstrap=500,seed=SEED+rep)
            sub=confirm.reverse_subgroup.to_numpy(int)
            harm=mae(Tc,C)-mae(Tc,Sc)
            rows.append({'frac_reverse':frac,'rep':rep,'mean_harm':prof.mean_harm,'mean_nonharm_supported':prof.mean_nonharm_supported,
                         'cvar20_harm':prof.cvar_harm,'cvar20_hi':prof.cvar_ci[1],'tail_nonharm_supported':prof.tail_nonharm_supported,
                         'harmed_fraction':prof.harmed_fraction,
                         'reverse_mean_harm':float(harm[sub==1].mean()) if sub.sum() else np.nan,
                         'majority_mean_harm':float(harm[sub==0].mean())})
    d=pd.DataFrame(rows); d.to_csv(OUT/'tail_safety_runs.csv',index=False)
    s=d.groupby('frac_reverse').agg(reps=('rep','count'),mean_harm=('mean_harm','mean'),mean_nonharm_rate=('mean_nonharm_supported','mean'),
                                    mean_cvar20=('cvar20_harm','mean'),tail_nonharm_rate=('tail_nonharm_supported','mean'),
                                    mean_harmed_fraction=('harmed_fraction','mean'),reverse_mean_harm=('reverse_mean_harm','mean')).reset_index()
    s.to_csv(OUT/'tail_safety_summary.csv',index=False)
    return s


def experiment_real_core_and_assurance():
    rows=[]
    states5=['Memory','Effector','Exhausted','Regulatory','Other_T']
    df=pd.read_csv(ROOT/'data/processed_pairs/GSE123813_pairs.csv')
    r=run_generic_lopo(df,RobustBlendCandidate(states=states5),bootstrap=800,seed=SEED,patient_balanced=True,states=states5)
    rows.append({'dataset':'GSE123813','legacy_qualified':r['legacy_qualified'],'core_qualified':r['core_qualified'],
                 'core_state':r['core_evidence_state'],'PDR_lo':r['PDR_lo'],'PUC_lo':r['PUC_lo'],'NPI_lo':r['NPI_lo'],
                 'retention_stability':r['retention_stability'],'limiting_axis':r['evidence_deficit']['limiting_axis']})

    aml_states=['HSC','Progenitor','GMP','Monocytic','Other']
    aml_edges=[('Monocytic','GMP'),('GMP','Progenitor'),('Progenitor','HSC')]
    df_aml=pd.read_csv(ROOT/'data/processed_pairs/GSE235063_AML_primary_pairs.csv')
    ra=run_generic_lopo(df_aml,GraphFlowCandidate(states=aml_states,edges=aml_edges,patient_balanced=True),bootstrap=800,seed=SEED+1,patient_balanced=True,states=aml_states)
    rows.append({'dataset':'AML','legacy_qualified':ra['legacy_qualified'],'core_qualified':ra['core_qualified'],
                 'core_state':ra['core_evidence_state'],'PDR_lo':ra['PDR_lo'],'PUC_lo':ra['PUC_lo'],'NPI_lo':ra['NPI_lo'],
                 'retention_stability':ra['retention_stability'],'limiting_axis':ra['evidence_deficit']['limiting_axis']})
    pd.DataFrame(rows).to_csv(OUT/'real_core_reclassification.csv',index=False)

    x=ra['patient_influence']['NPI_contribution'].to_numpy(float)
    ns=[21,40,60,80,100,150]
    plug=empirical_t_lcb_power(x,ns,mc=10000,seed=SEED+11)
    ass=bayesian_bootstrap_assurance(x,ns,posterior_draws=250,future_draws=250,seed=SEED+12)
    plan=plug.merge(ass,on='n',suffixes=('_plugin','_assurance'))
    plan.to_csv(OUT/'aml_replication_assurance.csv',index=False)
    return pd.DataFrame(rows), plan


def experiment_real_safe_averaging():
    rows=[]
    jobs=[
        ('GSE123813',ROOT/'data/processed_pairs/GSE123813_pairs.csv',
         ['Memory','Effector','Exhausted','Regulatory','Other_T'],
         [('Memory','Effector'),('Effector','Exhausted'),('Memory','Exhausted')]),
        ('AML',ROOT/'data/processed_pairs/GSE235063_AML_primary_pairs.csv',
         ['HSC','Progenitor','GMP','Monocytic','Other'],
         [('Monocytic','GMP'),('GMP','Progenitor'),('Progenitor','HSC')]),
    ]
    for ji,(name,path,states,full) in enumerate(jobs):
        full=tuple(full)
        lib=[full]+[tuple(e for e in full if e!=drop) for drop in full]
        cand=SafeGraphAveragingCandidate(states,lib,frozen_edges=full,cv_folds=3,grid_step=.25,selector_bootstrap=120,seed=SEED+ji*100)
        df=pd.read_csv(path)
        r=run_generic_lopo(df,cand,bootstrap=400,seed=SEED+400+ji,patient_balanced=True,states=states)
        rows.append({'dataset':name,'PDR':r['PDR'],'PDR_lo':r['PDR_lo'],'PUC':r['PUC'],'PUC_lo':r['PUC_lo'],
                     'NPI':r['NPI'],'NPI_lo':r['NPI_lo'],'retention_stability':r['retention_stability'],
                     'legacy_qualified':r['legacy_qualified'],'core_qualified':r['core_qualified'],'core_state':r['core_evidence_state'],
                     'limiting_axis':r['evidence_deficit']['limiting_axis']})
    d=pd.DataFrame(rows); d.to_csv(OUT/'real_safe_graph_averaging.csv',index=False)
    return d


def main():
    import argparse
    ap=argparse.ArgumentParser(description="Validate the principled QualifyOT decision and design layer")
    ap.add_argument("--suite", choices=("quick","iut","graph-set","safe-average","tail","real-core","real-safe","all"), default="quick")
    ap.add_argument("--full", action="store_true", help="Use the larger validation replicate counts")
    args=ap.parse_args()

    if args.suite in ("quick","iut","all"):
        a=experiment_iut_calibration(reps=4000 if args.full else 1000)
        print(a.to_string(index=False),flush=True)
    if args.suite in ("quick","graph-set","all"):
        b=experiment_graph_confidence(reps=10 if args.full else 3)
        print(b.to_string(index=False),flush=True)
    if args.suite in ("quick","safe-average","all"):
        c=experiment_safe_averaging(reps=20 if args.full else 5)
        print(c.to_string(index=False),flush=True)
    if args.suite in ("quick","tail","all"):
        d=experiment_tail_safety(reps=20 if args.full else 5)
        print(d.to_string(index=False),flush=True)
    if args.suite in ("quick","real-core","all"):
        e,p=experiment_real_core_and_assurance()
        print(e.to_string(index=False),flush=True)
        print(p[["n","prob_lcb_positive","assurance_mean","assurance_lo","assurance_hi"]].to_string(index=False),flush=True)
    if args.suite in ("real-safe","all"):
        # This is intentionally separated because nested safe graph averaging is
        # substantially more expensive than the other validation suites.
        f=experiment_real_safe_averaging()
        print(f.to_string(index=False),flush=True)


if __name__=='__main__':
    main()
