from __future__ import annotations

"""Reproduce the statistical validation analyses used in the manuscript.

This script intentionally does not refit or relabel locked biological analyses.
It only re-expresses existing Honest predictions under two valid estimand scopes,
checks Movement-margin sensitivity on already stored patient contributions, and
runs a seeded bounded-score decision-target simulation comparing scalar Utility
certification with the three-axis QualifyOT decision.
"""

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def _holm(pvals: dict[str, float], alpha: float = 0.05):
    names = list(pvals)
    order = sorted(names, key=lambda n: (pvals[n], n))
    rejected = {n: False for n in names}
    thresholds, ranks = {}, {}
    active = True
    m = len(order)
    for j, name in enumerate(order, start=1):
        thr = alpha / (m - j + 1)
        thresholds[name] = thr
        ranks[name] = j
        if active and pvals[name] <= thr:
            rejected[name] = True
        else:
            active = False
    return ranks, thresholds, rejected


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[2])
    ap.add_argument('--out', type=Path, default=None)
    args = ap.parse_args()
    root = args.root.resolve()
    out = (args.out or root / 'validation' / 'statistical_inference' / 'results').resolve()
    out.mkdir(parents=True, exist_ok=True)

    core = root
    gse = root / 'validation' / 'gse174554_expression'
    sys.path.insert(0, str(core / 'src'))

    from qualifyot.inference import (
        honest_confirmation_iut_cohort_conditional,
        honest_confirmation_iut_population,
        honest_confirmation_pvalues_localized,
        honest_confirmation_pvalues_population,
        movement_margin_evidence,
    )

    def dual_scope_from_long(df: pd.DataFrame, dataset: str):
        rows, p_cc, p_pop = [], {}, {}
        for candidate, z in df.groupby('candidate', sort=False):
            gm = z['movement_contrast'].to_numpy(float)
            a = z['movement_numerator'].to_numpy(float)
            u = z['utility'].to_numpy(float)
            cu = z['utility_abs_bound'].to_numpy(float)
            n_d = z['retention'].to_numpy(float)
            cnd = z['retention_abs_bound'].to_numpy(float)
            k = len([c for c in z.columns if c.startswith('target__')])
            cc = honest_confirmation_iut_cohort_conditional(
                gm, a, u, cu, n_d, cnd, movement_margin=.01, alpha=.05
            )
            pop = honest_confirmation_iut_population(
                gm, u, n_d, n_states=k, movement_margin=.01, alpha=.05
            )
            pc = honest_confirmation_pvalues_localized(
                gm, a, u, cu, n_d, cnd, movement_margin=.01
            )
            pp = honest_confirmation_pvalues_population(
                gm, u, n_d, n_states=k, movement_margin=.01
            )
            p_cc[candidate] = pc['candidate_iut']
            p_pop[candidate] = pp['candidate_iut']
            rows.append({
                'dataset': dataset,
                'candidate': candidate,
                'n_patients': len(z),
                'n_states': k,
                'movement_estimate': cc.movement.estimate,
                'utility_estimate': cc.utility.estimate,
                'deployment_retention_estimate': cc.retention.estimate,
                'cc_movement_lcb': cc.movement.lower_bound,
                'cc_utility_lcb': cc.utility.lower_bound,
                'cc_deployment_retention_lcb': cc.retention.lower_bound,
                'cc_qualified': cc.qualified,
                'cc_candidate_iut_p': pc['candidate_iut'],
                'population_movement_lcb': pop.movement.lower_bound,
                'population_utility_lcb': pop.utility.lower_bound,
                'population_deployment_retention_lcb': pop.retention.lower_bound,
                'population_qualified': pop.qualified,
                'population_candidate_iut_p': pp['candidate_iut'],
            })
        cc_rank, cc_thr, cc_rej = _holm(p_cc)
        pp_rank, pp_thr, pp_rej = _holm(p_pop)
        for row in rows:
            c = row['candidate']
            row.update(
                cc_holm_rank=cc_rank[c],
                cc_holm_threshold=cc_thr[c],
                cc_family_qualified=bool(row['cc_qualified'] and cc_rej[c]),
                population_holm_rank=pp_rank[c],
                population_holm_threshold=pp_thr[c],
                population_family_qualified=bool(row['population_qualified'] and pp_rej[c]),
            )
        return rows

    gse_long = pd.read_csv(gse / 'results/honest_confirmation/honest_confirmation_predictions_long.csv')
    gastric_long = pd.read_csv(
        core / 'results/gse315928/complete_validation/honest_confirmation_predictions_and_effects.csv'
    )
    dual = pd.DataFrame(
        dual_scope_from_long(gse_long, 'GSE174554') +
        dual_scope_from_long(gastric_long, 'GSE315928')
    )
    dual.to_csv(out / 'dual_scope_honest_audit.csv', index=False)

    # Movement margin sensitivity: diagnostic only; primary delta remains 0.01.
    real_dir = core / 'results/inference/real_data'
    files = {
        'GSE123813': 'GSE123813_patient_influence.csv',
        'GSE272993': 'GSE272993_9state_patient_influence.csv',
        'GSE236581': 'GSE236581_patient_influence.csv',
        'GSE120575': 'GSE120575_patient_influence.csv',
        'GSE179994': 'GSE179994_patient_influence.csv',
        'GSE175522': 'GSE175522_patient_influence.csv',
        'HCC': 'HCC_patient_influence.csv',
        'AML': 'AML_patient_influence.csv',
    }
    margin_rows = []
    for dataset, fname in files.items():
        z = pd.read_csv(real_dir / fname)
        for delta in (0.005, 0.01, 0.02, 0.05):
            ev = movement_margin_evidence(
                z['movement_numerator'], z['movement_denominator'], z['patient_id'],
                movement_margin=delta, alpha=.05, method='t'
            )
            margin_rows.append({
                'dataset': dataset, 'delta_M': delta, 'n_patients': ev.n_patients,
                'movement_ratio': ev.ratio, 'movement_contrast': ev.contrast,
                'movement_lcb': ev.contrast_lcb, 'supported': ev.supported,
            })
    margin = pd.DataFrame(margin_rows)
    margin.to_csv(out / 'movement_margin_sensitivity.csv', index=False)

    # Scalar Utility-only analogue vs the conjunctive decision target.
    rng = np.random.default_rng(20260909)
    reps, n, alpha = 20_000, 80, .05
    lower, upper = -.2, .2
    radius = (upper-lower) * math.sqrt(math.log(1/alpha)/(2*n))

    def score_means(theta: float) -> np.ndarray:
        p = (theta-lower)/(upper-lower)
        return np.where(rng.random((reps, n)) < p, upper, lower).mean(axis=1)

    scenarios = {
        'all_three_supported': (.12, .12, .12),
        'movement_boundary': (0., .12, .12),
        'utility_boundary': (.12, 0., .12),
        'deployment_retention_boundary': (.12, .12, 0.),
    }
    sim_rows = []
    for name, (theta_m, theta_u, theta_n) in scenarios.items():
        gm, u, n_d = score_means(theta_m), score_means(theta_u), score_means(theta_n)
        pass_m, pass_u, pass_n = gm-radius > 0, u-radius > 0, n_d-radius > 0
        sim_rows.append({
            'scenario': name, 'n': n, 'replicates': reps,
            'movement_mean': theta_m, 'utility_mean': theta_u,
            'deployment_retention_mean': theta_n,
            'utility_only_positive': float(pass_u.mean()),
            'qualifyot_positive': float((pass_m & pass_u & pass_n).mean()),
            'hoeffding_radius': radius,
        })
    decision = pd.DataFrame(sim_rows)
    decision.to_csv(out / 'utility_only_vs_qualifyot.csv', index=False)

    summary = out / 'statistical_inference_report.md'
    expr = dual[(dual.dataset == 'GSE174554') & (dual.candidate == 'ExpressionElasticNet')].iloc[0]
    summary.write_text(
        '# Statistical inference audit\n\n'
        '## Dual-scope Honest Confirmation\n'
        f'- Analyses audited: {len(dual)} (5 GSE174554 + 4 GSE315928).\n'
        f'- Cohort-conditional Qualified: {int(dual.cc_qualified.sum())}/{len(dual)}.\n'
        f'- Population Qualified: {int(dual.population_qualified.sum())}/{len(dual)}.\n'
        f'- ExpressionElasticNet GSE174554: cc LCBs '
        f'G_M={expr.cc_movement_lcb:.6f}, U={expr.cc_utility_lcb:.6f}, N_D={expr.cc_deployment_retention_lcb:.6g}; '
        f'population LCBs G_M={expr.population_movement_lcb:.6f}, U={expr.population_utility_lcb:.6f}, '
        f'N_D={expr.population_deployment_retention_lcb:.6f}.\n\n'
        '## Movement-margin sensitivity\n'
        f'- Positive Movement LCBs across delta_M in {{0.005,0.01,0.02,0.05}}: '
        f'{int(margin.supported.sum())}/{len(margin)}.\n'
        f'- Minimum diagnostic Movement LCB: {margin.movement_lcb.min():.6f}.\n'
        '- This is a sensitivity audit and does not alter the primary pre-specified delta_M=0.01.\n\n'
        '## Scalar Utility-only analogue versus QualifyOT\n' +
        decision.to_markdown(index=False) + '\n\n'
        'This comparison is a decision-target stress test, not a reproduction or superiority claim against Learn-then-Test.\n',
        encoding='utf-8'
    )
    print(summary.read_text())


if __name__ == '__main__':
    main()
