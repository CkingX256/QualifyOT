from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

RELEASE = Path(__file__).resolve().parents[2]
ROOT = RELEASE / 'validation' / 'gse174554_expression'
sys.path.insert(0, str(ROOT / 'src'))

from qualifyot.inference import honest_confirm_predictions
from qualifyot.model import feature_matrix, fit_null, mae, predict_null
from qualifyot.weight_selection import one_se_msw, patient_curves
from qualifyot_gse174554.candidate import ExpressionElasticNetCandidate, euclidean_simplex_projection
from qualifyot_gse174554.pipeline import STATES, load_processed_inputs

OUT = RELEASE / 'validation' / 'split_and_nonlinear' / 'results'
OUT.mkdir(parents=True, exist_ok=True)
GRID = np.round(np.arange(0.0, 1.00001, 0.01), 2)
BOOTSTRAP = 1000
SPLIT_SEEDS = [93117, 104729, 130363, 155921, 181081, 205019, 229939, 254053, 279121, 304003]
MLP_SEEDS = [20260909, 20260919, 20260929, 20260939, 20260949]


def source_only(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.drop(columns=[c for c in frame.columns if c.startswith('target__')])


def development_oof(frame: pd.DataFrame, candidate):
    _, source, target = feature_matrix(frame, states=STATES)
    patients = frame.patient_id.astype(str).to_numpy()
    candidate_predictions = np.zeros_like(target)
    reference_predictions = np.zeros_like(target)
    for heldout in sorted(set(patients)):
        test_mask = patients == heldout
        train = frame.loc[~test_mask].reset_index(drop=True)
        test = frame.loc[test_mask].reset_index(drop=True)
        candidate_predictions[test_mask] = candidate.fresh().fit(train).predict(source_only(test))
        reference_model = fit_null(
            'MeanDelta', source[~test_mask], target[~test_mask],
            patients=patients[~test_mask], patient_balanced=True
        )
        reference_predictions[test_mask] = predict_null(reference_model, source[test_mask])
    return candidate_predictions, reference_predictions


def evaluate_candidate(development: pd.DataFrame, confirmation: pd.DataFrame, candidate, seed: int) -> dict:
    t0 = time.perf_counter()
    candidate_oof, reference_oof = development_oof(development, candidate)
    _, source_dev, target_dev = feature_matrix(development, states=STATES)
    dev_patients = development.patient_id.astype(str).to_numpy()
    _, harm_curves, risk_curves = patient_curves(
        target_dev, reference_oof, candidate_oof, dev_patients, GRID
    )
    selection = one_se_msw(
        harm_curves, risk_curves, GRID, alpha=0.05, delta=0.0,
        B=BOOTSTRAP, seed=seed, lam=1.0
    )
    weight = float(selection['weight'])

    full_candidate = candidate.fresh().fit(development)
    reference_model = fit_null(
        'MeanDelta', source_dev, target_dev,
        patients=dev_patients, patient_balanced=True
    )
    _, source_conf, target_conf = feature_matrix(confirmation, states=STATES)
    conf_patients = confirmation.patient_id.astype(str).to_numpy()
    candidate_conf = full_candidate.predict(source_only(confirmation))
    reference_conf = predict_null(reference_model, source_conf)
    retained_conf = (1.0 - weight) * reference_conf + weight * candidate_conf
    results, detail = honest_confirm_predictions(
        target_conf, source_conf, reference_conf, candidate_conf, retained_conf,
        conf_patients, movement_margin=0.01, alpha=0.05, return_both=True
    )
    cc = results['cohort_conditional']
    pop = results['population']
    return {
        'selected_lambda': weight,
        'reference_mae': float(mae(target_conf, reference_conf).mean()),
        'candidate_mae': float(mae(target_conf, candidate_conf).mean()),
        'retained_mae': float(mae(target_conf, retained_conf).mean()),
        'movement_estimate': float(cc.movement.estimate),
        'cc_movement_lcb': float(cc.movement.lower_bound),
        'cc_utility_estimate': float(cc.utility.estimate),
        'cc_utility_lcb': float(cc.utility.lower_bound),
        'cc_retention_estimate': float(cc.retention.estimate),
        'cc_retention_lcb': float(cc.retention.lower_bound),
        'cc_qualified': bool(cc.qualified),
        'pop_movement_lcb': float(pop.movement.lower_bound),
        'pop_utility_lcb': float(pop.utility.lower_bound),
        'pop_retention_lcb': float(pop.retention.lower_bound),
        'pop_qualified': bool(pop.qualified),
        'runtime_seconds': float(time.perf_counter() - t0),
    }


class DeepMLPProxyCandidate:
    """Post-lock nonlinear capacity proxy using only source-patient expression.

    This is deliberately not presented as Waddington-OT/scNODE/MIOFlow. Feature
    selection, scaling, PCA and MLP fitting occur inside each training fold.
    """
    name = 'DeepMLPProxy'

    def __init__(self, store, seed: int, max_genes: int = 512):
        self.store = store
        self.seed = int(seed)
        self.max_genes = int(max_genes)
        self._idx = None
        self._scaler = None
        self._pca = None
        self._model = None

    def fresh(self):
        return type(self)(self.store, self.seed, self.max_genes)

    def fit(self, pairs: pd.DataFrame):
        ids = pairs.patient_id.astype(str).tolist()
        x = self.store.rows(ids)
        source = pairs[[f'source__{s}' for s in STATES]].to_numpy(float)
        target = pairs[[f'target__{s}' for s in STATES]].to_numpy(float)
        variances = np.var(x, axis=0, ddof=0)
        # Stable descending variance order with original feature index as tie-break.
        idx = np.lexsort((np.arange(x.shape[1]), -variances))[: min(self.max_genes, x.shape[1])]
        scaler = StandardScaler().fit(x[:, idx])
        z = scaler.transform(x[:, idx])
        n_components = min(8, max(1, len(pairs) - 2), z.shape[1])
        pca = PCA(n_components=n_components, svd_solver='full').fit(z)
        q = pca.transform(z)
        model = MLPRegressor(
            hidden_layer_sizes=(16, 8), activation='tanh', solver='lbfgs',
            alpha=0.01, max_iter=2000, random_state=self.seed
        ).fit(q, target - source)
        self._idx, self._scaler, self._pca, self._model = idx, scaler, pca, model
        return self

    def predict(self, pairs: pd.DataFrame) -> np.ndarray:
        if self._model is None:
            raise RuntimeError('DeepMLPProxyCandidate is not fitted')
        x = self.store.rows(pairs.patient_id.astype(str).tolist())
        source = pairs[[f'source__{s}' for s in STATES]].to_numpy(float)
        q = self._pca.transform(self._scaler.transform(x[:, self._idx]))
        return euclidean_simplex_projection(source + self._model.predict(q))


def main() -> int:
    pairs, store = load_processed_inputs(ROOT)
    patient_ids = np.array(sorted(pairs.patient_id.astype(str).unique()))
    if len(patient_ids) != 26:
        raise RuntimeError(f'expected 26 analyzable patients, found {len(patient_ids)}')

    # A. Retrospective repeated frozen-split stress test. Seeds are fixed above.
    split_rows = []
    for split_index, seed in enumerate(SPLIT_SEEDS, start=1):
        rng = np.random.default_rng(seed)
        perm = rng.permutation(patient_ids)
        dev_ids = set(perm[:16])
        conf_ids = set(perm[16:])
        development = pairs[pairs.patient_id.astype(str).isin(dev_ids)].reset_index(drop=True)
        confirmation = pairs[pairs.patient_id.astype(str).isin(conf_ids)].reset_index(drop=True)
        result = evaluate_candidate(
            development, confirmation,
            ExpressionElasticNetCandidate(store, states=tuple(STATES)), seed
        )
        result.update({
            'split_index': split_index,
            'split_seed': seed,
            'development_patient_ids': ';'.join(sorted(dev_ids)),
            'confirmation_patient_ids': ';'.join(sorted(conf_ids)),
        })
        split_rows.append(result)
    split_df = pd.DataFrame(split_rows)
    split_df.to_csv(OUT / 'gse174554_repeated_frozen_split_audit.csv', index=False)

    # B. Nonlinear capacity proxy on the already frozen primary 16/10 split.
    development = pairs[pairs.analysis_partition == 'development'].reset_index(drop=True)
    confirmation = pairs[pairs.analysis_partition == 'confirmation'].reset_index(drop=True)
    mlp_rows = []
    for seed in MLP_SEEDS:
        result = evaluate_candidate(development, confirmation, DeepMLPProxyCandidate(store, seed), seed)
        result.update({'model': 'DeepMLPProxy', 'seed': seed, 'max_genes': 512, 'pca_components_max': 8, 'hidden_layers': '16;8'})
        mlp_rows.append(result)
    mlp_df = pd.DataFrame(mlp_rows)
    mlp_df.to_csv(OUT / 'gse174554_deep_mlp_proxy_audit.csv', index=False)

    report = {
        'scope': 'retrospective supplemental stress tests; neither analysis is an untouched prospective confirmation or a substitute for Waddington-OT/MIOFlow/scNODE',
        'repeated_split': {
            'n_splits': int(len(split_df)),
            'cohort_conditional_qualified': int(split_df.cc_qualified.sum()),
            'population_qualified': int(split_df.pop_qualified.sum()),
            'lambda_zero': int((split_df.selected_lambda == 0).sum()),
            'median_candidate_mae': float(split_df.candidate_mae.median()),
            'median_reference_mae': float(split_df.reference_mae.median()),
            'utility_lcb_min': float(split_df.cc_utility_lcb.min()),
            'utility_lcb_max': float(split_df.cc_utility_lcb.max()),
        },
        'deep_proxy': {
            'n_seeds': int(len(mlp_df)),
            'cohort_conditional_qualified': int(mlp_df.cc_qualified.sum()),
            'population_qualified': int(mlp_df.pop_qualified.sum()),
            'lambda_zero': int((mlp_df.selected_lambda == 0).sum()),
            'candidate_mae_min': float(mlp_df.candidate_mae.min()),
            'candidate_mae_max': float(mlp_df.candidate_mae.max()),
            'reference_mae': float(mlp_df.reference_mae.iloc[0]),
            'cc_utility_lcb_min': float(mlp_df.cc_utility_lcb.min()),
            'cc_utility_lcb_max': float(mlp_df.cc_utility_lcb.max()),
        },
        'fixed_split_seeds': SPLIT_SEEDS,
        'mlp_seeds': MLP_SEEDS,
        'retention_bootstrap': BOOTSTRAP,
    }
    (OUT / 'SUMMARY.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
