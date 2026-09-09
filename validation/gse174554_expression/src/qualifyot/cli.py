"""Command-line inspection and reporting utilities for QualifyOT.

The CLI validates standardized longitudinal-pair tables, applies the primary
Movement–Utility–Retention intersection-union decision (with an explicit legacy
six-gate reproduction option), emits machine-readable / Markdown reports, and
provides a rapid GraphFlow quicktest for exploratory local diagnostics.
"""
from __future__ import annotations

import argparse
import html
import json
import time
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from .candidate_api import GraphFlowCandidate
from .evidence_state import GateConfig, classify
from .decision import IUTConfig, classify_iut
from .generic_engine import run_generic_lopo


_METRIC_ALIASES = {
    "pdr": ("pdr", "PDR", "PDR_Hellinger", "PDR_point"),
    "pdr_lo": ("pdr_lo", "PDR_lo", "PDR_CI2.5", "PDR_LCB"),
    "puc_lo": ("puc_lo", "PUC_lo", "PUC_CI2.5", "PUC_LCB"),
    "puc_hi": ("puc_hi", "PUC_hi", "PUC_CI97.5", "PUC_UCB"),
    "npi_lo": ("npi_lo", "NPI_lo", "NPI_CI2.5", "NPI_LCB"),
    "positive_weight_fraction": (
        "positive_weight_fraction",
        "positive_weight_fold_fraction",
        "positive_weight_folds_fraction",
    ),
    "final_risk": ("final_risk", "qualified_risk", "retained_risk", "R_qualified"),
    "reference_risk": ("reference_risk", "R_reference", "ref_risk"),
}


def _parse_args(argv=None):
    p = argparse.ArgumentParser(prog="qualifyot")
    sub = p.add_subparsers(dest="command", required=True)

    v = sub.add_parser("validate-csv", help="Validate a standardized longitudinal-pair CSV")
    v.add_argument("csv")
    v.add_argument("--states", required=True, help="Comma-separated state names")

    e = sub.add_parser("classify", help="Classify a saved metric JSON using the core IUT or legacy rule")
    e.add_argument("json_file")
    e.add_argument("--rule", choices=("core", "legacy", "both"), default="core")

    r = sub.add_parser("report", help="Create a standardized QualifyOT JSON + Markdown report from metric JSON")
    r.add_argument("json_file")
    r.add_argument("--out-prefix", default=None, help="Output prefix (default: <json_file stem>_qualifyot_report)")

    q = sub.add_parser(
        "quicktest",
        help="Rapid GraphFlow screen from a standardized pair CSV; exploratory, not a locked final analysis",
    )
    q.add_argument("csv")
    q.add_argument("--states", required=True, help="Comma-separated state names")
    q.add_argument("--graph", required=True, help="Comma-separated directed edges, e.g. A->B,B->C")
    q.add_argument("--alpha", type=float, default=1.0, help="GraphFlow ridge penalty (default: 1)")
    q.add_argument("--bootstrap", type=int, default=300, help="Patient bootstrap resamples for the rapid screen (default: 300)")
    q.add_argument("--seed", type=int, default=20260825)
    q.add_argument("--out", default=None, help="HTML output path (default: <csv stem>_qualifyot_quicktest.html)")

    cr = sub.add_parser("candidate-run", help="Run a generic patient-level candidate. The primary neural option is one graph-constrained residual Transformer.")
    cr.add_argument("csv")
    cr.add_argument("--states", required=True, help="Comma-separated state names")
    cr.add_argument("--candidate", default="graph-transformer", metavar="METHOD",
                    help="Primary choices: graph-transformer, graphflow, robustblend. Legacy aliases remain accepted for reproducibility.")
    cr.add_argument("--graph", default=None, help="Required for graph-transformer/graphflow; A->B,B->C syntax")
    cr.add_argument("--bootstrap", type=int, default=300)
    cr.add_argument("--seed", type=int, default=20260830)
    cr.add_argument("--epochs", type=int, default=20)
    cr.add_argument("--d-model", type=int, default=8)
    cr.add_argument("--nhead", type=int, default=2)
    cr.add_argument("--layers", type=int, default=1)
    cr.add_argument("--ff", type=int, default=16)
    cr.add_argument("--residual-fraction", "--residual-scale", dest="residual_scale", type=float, default=0.20,
                    help="Bounded Transformer residual fraction of source tail-state mass (default: 0.20; --residual-scale is a legacy alias)")
    cr.add_argument("--reference-rule", choices=("nested", "Persistence", "CohortMean", "MeanDelta"), default="nested")
    cr.add_argument("--out", default=None, help="Optional JSON output path")

    bp = sub.add_parser("profile-patient", help="Orthogonal patient-level benchmark and Bayesian-bootstrap evidence profile")
    bp.add_argument("csv", help="CSV containing one row per physical patient")
    bp.add_argument("--puc-col", default="PUC_contribution")
    bp.add_argument("--npi-col", default="NPI_contribution")
    bp.add_argument("--bootstrap", type=int, default=2000)
    bp.add_argument("--draws", type=int, default=10000)
    bp.add_argument("--seed", type=int, default=20260830)
    bp.add_argument("--out", default=None, help="Optional JSON output path")

    gp = sub.add_parser("graph-profile", help="Predictive graph confidence set plus Bayesian-bootstrap graph profile")
    gp.add_argument("csv", help="CSV with patient_id and one loss column per graph")
    gp.add_argument("--models", required=True, help="Comma-separated graph loss columns")
    gp.add_argument("--bootstrap", type=int, default=2000)
    gp.add_argument("--draws", type=int, default=10000)
    gp.add_argument("--seed", type=int, default=20260830)
    gp.add_argument("--out", default=None, help="Optional JSON output path")

    ch = sub.add_parser("contract-hash", help="Print the semantic SHA256 of a frozen YAML analysis contract")
    ch.add_argument("contract")
    ch.add_argument("--write", action="store_true", help="Also write <contract>.sha256 beside the YAML file")

    fp = sub.add_parser("freeze-patients", help="Freeze an outcome-free physical-patient set and deterministic split")
    fp.add_argument("--contract", required=True, help="Frozen YAML contract containing inference.split_salt and development_fraction")
    fp.add_argument("--ids", required=True, help="Text/CSV file containing physical patient IDs")
    fp.add_argument("--id-column", default="patient_id", help="CSV patient-ID column (default: patient_id); ignored for plain text")
    fp.add_argument("--out-prefix", required=True, help="Output prefix for .patients.txt, .split.csv and .target_access_gate.json")
    fp.add_argument("--dependency", action="append", default=[], help="Additional immutable dependency to hash into the target-access gate; repeatable")

    vg = sub.add_parser("verify-target-access", help="Fail closed unless contract, patient set and split match a frozen gate")
    vg.add_argument("--contract", required=True)
    vg.add_argument("--patients", required=True)
    vg.add_argument("--split", required=True)
    vg.add_argument("--gate", required=True)
    vg.add_argument("--dependency", action="append", default=None, help="Dependency path matching the frozen gate; repeatable. If omitted, stored freeze paths are verified.")
    return p.parse_args(argv)


def _validate_csv(path: Path, states: list[str]) -> dict:
    df = pd.read_csv(path)
    if "patient_id" not in df.columns:
        raise SystemExit("Missing column: patient_id")

    def _resolve(prefix_short: str, prefix_long: str) -> list[str]:
        short = [f"{prefix_short}__{s}" for s in states]
        long = [f"{prefix_long}__{s}" for s in states]
        if all(c in df.columns for c in short):
            return short
        if all(c in df.columns for c in long):
            return long
        missing_short = [c for c in short if c not in df.columns]
        missing_long = [c for c in long if c not in df.columns]
        raise SystemExit(
            f"Missing standardized composition columns. Expected either {prefix_short}__<state> "
            f"or {prefix_long}__<state>. Missing short-form: {', '.join(missing_short)}; "
            f"missing long-form: {', '.join(missing_long)}"
        )

    src_cols = _resolve("src", "source")
    tgt_cols = _resolve("tgt", "target")
    src = df[src_cols].to_numpy(float)
    tgt = df[tgt_cols].to_numpy(float)
    tol = 1e-8
    if (src < -tol).any() or (tgt < -tol).any():
        raise SystemExit("Negative composition entries found")
    src_err = abs(src.sum(axis=1) - 1).max(initial=0.0)
    tgt_err = abs(tgt.sum(axis=1) - 1).max(initial=0.0)
    if src_err > 1e-6 or tgt_err > 1e-6:
        raise SystemExit(f"Simplex closure failed: max source error={src_err:g}, target error={tgt_err:g}")
    return {
        "rows": int(len(df)),
        "patients": int(df.patient_id.astype(str).nunique()),
        "states": states,
        "max_source_simplex_error": float(src_err),
        "max_target_simplex_error": float(tgt_err),
        "valid": True,
    }


def _first_present(d: Mapping[str, Any], aliases: tuple[str, ...], canonical: str) -> float:
    for key in aliases:
        if key in d and d[key] is not None:
            return float(d[key])
    raise SystemExit(f"Missing metric '{canonical}'. Accepted keys: {', '.join(aliases)}")


def _normalize_metrics(m: Mapping[str, Any]) -> dict[str, float]:
    return {k: _first_present(m, aliases, k) for k, aliases in _METRIC_ALIASES.items()}


def _classify_mapping(m: Mapping[str, Any]):
    q = _normalize_metrics(m)
    return classify(
        q["pdr"], q["pdr_lo"], q["puc_lo"], q["puc_hi"], q["npi_lo"],
        q["positive_weight_fraction"], q["final_risk"], q["reference_risk"], GateConfig()
    ), q


def _classify_core_mapping(m: Mapping[str, Any]):
    q = _normalize_metrics(m)
    return classify_iut(
        pdr_lcb=q["pdr_lo"], puc_lcb=q["puc_lo"], puc_ucb=q["puc_hi"],
        npi_lcb=q["npi_lo"], positive_weight_fraction=q["positive_weight_fraction"],
        cfg=IUTConfig(),
    ), q


def _make_report(raw: Mapping[str, Any]) -> dict[str, Any]:
    legacy, q = _classify_mapping(raw)
    core, _ = _classify_core_mapping(raw)
    stability = raw.get("perturbation_stability") or raw.get("stability") or {}
    point_metrics = {
        k: raw.get(k) for k in (
            "PUC", "PUC_point", "NPI", "NPI_point", "PDR", "PDR_point",
            "candidate_risk", "reference_risk", "final_risk"
        ) if k in raw
    }
    return {
        "schema": "qualifyot-report",
        "evidence_state": core.label,
        "core_evidence": core.to_dict(),
        "legacy_evidence_state": legacy.label,
        "passed_gates": legacy.passed_gates,
        "total_gates": legacy.total_gates,
        "gates": legacy.gates,
        "limiting_gates": list(legacy.limiting_gates),
        "classification_inputs": q,
        "reported_point_metrics": point_metrics,
        "perturbation_stability": stability,
        "interpretation": (
            "The primary report separates Movement, Utility and Retention as an intersection-union evidence decision. "
            "Positive-weight fold frequency is reported as retention stability rather than a qualification gate. "
            "The legacy six-gate state is retained for backward-compatible reproduction."
        ),
    }


def _report_markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# QualifyOT Report",
        "",
        f"**Core evidence state:** {report['evidence_state']}",
        f"**Legacy evidence state:** {report['legacy_evidence_state']}",
        "",
        "## Core Movement–Utility–Retention decision",
        "",
        "| Axis | Pass |",
        "|---|---:|",
    ]
    for axis, ok in report["core_evidence"]["axes"].items():
        lines.append(f"| {axis} | {'PASS' if ok else 'LIMIT'} |")
    lines += [
        "",
        f"**Retention stability:** {100*report['core_evidence']['retention_stability']:.1f}% positive-weight outer folds",
        "",
        "## Legacy six-gate audit",
        "",
        "| Gate | Pass |",
        "|---|---:|",
    ]
    for gate, ok in report["gates"].items():
        lines.append(f"| {gate} | {'PASS' if ok else 'LIMIT'} |")
    lines += ["", "## Limiting gates", ""]
    if report["limiting_gates"]:
        lines += [f"- {x}" for x in report["limiting_gates"]]
    else:
        lines.append("- None in the locked six-gate analysis")
    lines += ["", "## Perturbation stability", ""]
    stab = report.get("perturbation_stability") or {}
    if stab:
        for k, v in stab.items():
            lines.append(f"- **{k}:** {v}")
    else:
        lines.append("- Not supplied in the input metrics JSON")
    lines += ["", "## Interpretation", "", report["interpretation"], ""]
    return "\n".join(lines)


def _parse_graph(spec: str, states: list[str]) -> list[tuple[str, str]]:
    edges: list[tuple[str, str]] = []
    for raw in spec.split(","):
        raw = raw.strip()
        if not raw:
            continue
        if "->" not in raw:
            raise SystemExit(f"Invalid edge '{raw}'. Use A->B syntax.")
        a, b = [x.strip() for x in raw.split("->", 1)]
        if a not in states or b not in states:
            raise SystemExit(f"Edge '{raw}' references a state not listed in --states")
        if a == b:
            raise SystemExit(f"Self-loop '{raw}' is not supported in quicktest")
        edges.append((a, b))
    if not edges:
        raise SystemExit("--graph must contain at least one directed edge")
    return edges


def _quicktest_html(result: Mapping[str, Any], core_state, legacy_state, runtime_s: float, source_name: str, bootstrap: int) -> str:
    inf = result["patient_influence"].copy().sort_values("NPI_contribution")
    vals = inf["NPI_contribution"].astype(float).to_list()
    pids = inf["patient_id"].astype(str).to_list()
    vmax = max([abs(x) for x in vals] + [1e-12])
    W = 760
    H = max(170, 30 * len(vals) + 55)
    mid = 380
    scale = 310 / vmax
    bars = []
    for j, (pid, v) in enumerate(zip(pids, vals)):
        y = 26 + j * 28
        x2 = mid + v * scale
        x = min(mid, x2)
        w = max(1, abs(x2 - mid))
        fill = "#2f6fa3" if v >= 0 else "#b94b4b"
        bars.append(f'<text x="8" y="{y+12}" font-size="12">{html.escape(pid)}</text>')
        bars.append(f'<rect x="{x:.1f}" y="{y}" width="{w:.1f}" height="16" fill="{fill}" opacity="0.82"/>')
    svg = (
        f'<svg viewBox="0 0 {W} {H}" width="100%" role="img" aria-label="patient-level NPI contributions">'
        f'<line x1="{mid}" y1="8" x2="{mid}" y2="{H-18}" stroke="#333" stroke-width="1"/>'
        + "".join(bars) + "</svg>"
    )
    axes = "".join(
        f'<tr><td>{html.escape(k)}</td><td class="{"pass" if v else "limit"}">{"PASS" if v else "LIMIT"}</td></tr>'
        for k, v in core_state.axes.items()
    )
    lim = ", ".join(core_state.limiting_axes) if core_state.limiting_axes else "None"

    def sf(x: float) -> str:
        return f"{float(x):+.5f}"

    return f'''<!doctype html><html><head><meta charset="utf-8"><title>QualifyOT quicktest</title><style>
body{{font-family:Arial,Helvetica,sans-serif;max-width:900px;margin:28px auto;color:#222;line-height:1.42}}
h1{{font-size:24px;margin-bottom:2px}} .note{{background:#fff8db;border-left:4px solid #d6a73a;padding:10px 13px}}
.grid{{display:grid;grid-template-columns:1fr 1fr;gap:18px}} table{{border-collapse:collapse;width:100%}} td,th{{border-bottom:1px solid #ddd;padding:6px 8px;text-align:left}}
.pass{{font-weight:700;color:#176b3a}} .limit{{font-weight:700;color:#a53636}} .metric{{font-family:ui-monospace,Consolas,monospace}}
.small{{font-size:12px;color:#555}} h2{{font-size:17px;margin-top:22px}}
</style></head><body>
<h1>QualifyOT rapid diagnostic</h1><div class="small">Input: {html.escape(source_name)} | bootstrap={bootstrap} | runtime={runtime_s:.2f} s</div>
<p class="note"><b>Exploratory screen.</b> quicktest uses a reduced bootstrap count for rapid local checking. It does not replace the pre-specified locked analysis or complete-pipeline perturbation assessment.</p>
<div class="grid"><div><h2>Provisional evidence state</h2><p style="font-size:26px;font-weight:700">{html.escape(core_state.label)}</p><p><b>Limiting gates:</b> {html.escape(lim)}</p>
<table><tr><th>Metric</th><th>Estimate [95% interval]</th></tr><tr><td>PDR</td><td class="metric">{result['PDR']:.4f} [{result['PDR_lo']:.4f}, {result['PDR_hi']:.4f}]</td></tr>
<tr><td>PUC</td><td class="metric">{sf(result['PUC'])} [{sf(result['PUC_lo'])}, {sf(result['PUC_hi'])}]</td></tr><tr><td>NPI</td><td class="metric">{sf(result['NPI'])} [{sf(result['NPI_lo'])}, {sf(result['NPI_hi'])}]</td></tr>
<tr><td>Positive-weight folds</td><td class="metric">{100*result['positive_weight_fold_fraction']:.1f}%</td></tr></table></div>
<div><h2>Movement–Utility–Retention</h2><table><tr><th>Axis</th><th>Status</th></tr>{axes}</table>
<p class="small">Retention stability: {100*result['positive_weight_fold_fraction']:.1f}% positive-weight outer folds. Legacy six-gate state: {html.escape(legacy_state.label)}.</p></div></div>
<h2>Patient-level retained NPI contributions</h2>{svg}<p class="small">Positive bars indicate lower retained loss than the selected reference for that held-out patient; negative bars indicate harm relative to the reference.</p>
</body></html>'''


def _run_quicktest(args):
    states = [x.strip() for x in args.states.split(",") if x.strip()]
    _validate_csv(Path(args.csv), states)
    edges = _parse_graph(args.graph, states)
    df = pd.read_csv(args.csv)
    if "pair_id" not in df.columns:
        df = df.copy()
        df["pair_id"] = [f"pair_{i:05d}" for i in range(len(df))]
    t0 = time.perf_counter()
    cand = GraphFlowCandidate(alpha=args.alpha, patient_balanced=True, states=states, edges=edges, name="GraphFlowQuicktest")
    res = run_generic_lopo(df, cand, bootstrap=args.bootstrap, seed=args.seed, patient_balanced=True, states=states)
    legacy_state = classify(
        res["PDR"], res["PDR_lo"], res["PUC_lo"], res["PUC_hi"], res["NPI_lo"],
        res["positive_weight_fold_fraction"], res["qualified_risk"], res["reference_risk"], GateConfig()
    )
    core_state = classify_iut(
        pdr_lcb=res["PDR_lo"], puc_lcb=res["PUC_lo"], puc_ucb=res["PUC_hi"],
        npi_lcb=res["NPI_lo"], positive_weight_fraction=res["positive_weight_fold_fraction"],
    )
    runtime = time.perf_counter() - t0
    out = Path(args.out) if args.out else Path(args.csv).with_name(Path(args.csv).stem + "_qualifyot_quicktest.html")
    out.write_text(_quicktest_html(res, core_state, legacy_state, runtime, Path(args.csv).name, args.bootstrap), encoding="utf-8")
    print(json.dumps({"html": str(out), "provisional_evidence_state": core_state.label, "legacy_evidence_state": legacy_state.label, "runtime_seconds": runtime, "bootstrap": args.bootstrap}, indent=2))
    return 0



def _jsonable(x):
    import numpy as np
    import pandas as pd
    if isinstance(x, pd.DataFrame):
        return [_jsonable(r) for r in x.to_dict(orient="records")]
    if isinstance(x, pd.Series):
        return _jsonable(x.to_dict())
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.floating, np.integer, np.bool_)):
        return x.item()
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    return x


def _run_candidate(args):
    states=[x.strip() for x in args.states.split(",") if x.strip()]
    _validate_csv(Path(args.csv),states)
    df=pd.read_csv(args.csv)
    if "pair_id" not in df.columns:
        df=df.copy(); df["pair_id"]=[f"pair_{i:05d}" for i in range(len(df))]
    method=str(args.candidate).lower().strip()
    if method == "robustblend":
        from .candidates import RobustBlendCandidate
        cand=RobustBlendCandidate(states=states,name="RobustBlendCLI")
    elif method == "graphflow":
        if not args.graph:
            raise SystemExit("--graph is required for graphflow")
        edges=_parse_graph(args.graph,states)
        cand=GraphFlowCandidate(states=states,edges=edges,patient_balanced=True,name="GraphFlowCLI")
    elif method in {"graph-transformer", "transformer-graphflow"}:
        if not args.graph:
            raise SystemExit("--graph is required for graph-transformer")
        edges=_parse_graph(args.graph,states)
        try:
            from .transformer_candidate import GraphResidualTransformerCandidate
        except ImportError as e:
            raise SystemExit(str(e))
        cand=GraphResidualTransformerCandidate(
            states=states,edges=edges,d_model=args.d_model,nhead=args.nhead,
            num_layers=args.layers,dim_feedforward=args.ff,epochs=args.epochs,
            seed=args.seed,patient_balanced=True,residual_fraction=args.residual_scale,
            name=("TransformerGraphFlow" if method == "transformer-graphflow" else "GraphResidualTransformer"),
        )
    elif method in {"transformer", "legacy-residual-transformer", "graph-aware-transformer"}:
        # Exact compatibility path for the previous development release.  It is
        # intentionally not advertised as a primary methodology.
        try:
            from .legacy_transformer import ResidualStateTransformerCandidate, GraphAwareTransformerCandidate
        except ImportError as e:
            raise SystemExit(str(e))
        common=dict(states=states,d_model=args.d_model,nhead=args.nhead,num_layers=args.layers,
                    dim_feedforward=args.ff,epochs=args.epochs,residual_scale=args.residual_scale,
                    seed=args.seed,patient_balanced=True)
        if method in {"transformer", "legacy-residual-transformer"}:
            cand=ResidualStateTransformerCandidate(**common)
        else:
            if not args.graph:
                raise SystemExit("--graph is required for graph-aware-transformer")
            cand=GraphAwareTransformerCandidate(edges=_parse_graph(args.graph,states),**common)
    else:
        raise SystemExit(
            "Unknown --candidate. Use graph-transformer, graphflow or robustblend "
            "(legacy aliases: transformer, transformer-graphflow)."
        )
    t0=time.perf_counter()
    res=run_generic_lopo(df,cand,bootstrap=args.bootstrap,seed=args.seed,
                         patient_balanced=True,states=states,reference_rule=args.reference_rule,
                         orthogonal_profiles=False,return_predictions=False)
    runtime=time.perf_counter()-t0
    out={k:v for k,v in res.items() if k not in {"prediction_rows"}}
    out["runtime_seconds"]=runtime
    out["schema"]="qualifyot-candidate-run"
    text=json.dumps(_jsonable(out),indent=2)
    if args.out:
        Path(args.out).write_text(text,encoding="utf-8")
    print(text)
    return 0

def _read_patient_ids(path: Path, id_column: str = "patient_id") -> list[str]:
    """Read physical patient IDs without inspecting any target outcome column."""
    if path.suffix.lower() in {".csv", ".tsv"}:
        sep = "\t" if path.suffix.lower() == ".tsv" else ","
        df = pd.read_csv(path, sep=sep, usecols=[id_column], dtype={id_column: str})
        ids = [x.strip() for x in df[id_column].astype(str) if x.strip()]
    else:
        ids = [x.strip() for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]
    if not ids:
        raise SystemExit("No physical patient IDs found")
    return ids


def main(argv=None):
    args = _parse_args(argv)
    if args.command == "contract-hash":
        from .external_contract import contract_sha256, write_contract_digest
        digest = contract_sha256(args.contract)
        out = {"contract": str(Path(args.contract)), "semantic_sha256": digest}
        if args.write:
            out["digest_file"] = str(write_contract_digest(args.contract))
        print(json.dumps(out, indent=2))
        return 0
    if args.command == "freeze-patients":
        from .external_contract import freeze_patient_set_and_split
        ids = _read_patient_ids(Path(args.ids), args.id_column)
        prefix = Path(args.out_prefix)
        gate = freeze_patient_set_and_split(
            ids,
            contract_path=args.contract,
            patient_file=prefix.with_suffix(".patients.txt"),
            split_file=prefix.with_suffix(".split.csv"),
            gate_file=prefix.with_suffix(".target_access_gate.json"),
            dependency_paths=args.dependency,
        )
        print(json.dumps(gate, indent=2))
        return 0
    if args.command == "verify-target-access":
        from .external_contract import verify_target_access_gate
        verify_target_access_gate(
            contract_path=args.contract,
            patient_file=args.patients,
            split_file=args.split,
            gate_file=args.gate,
            dependency_paths=args.dependency,
        )
        print(json.dumps({"target_access_gate_valid": True}, indent=2))
        return 0
    if args.command == "validate-csv":
        states = [x.strip() for x in args.states.split(",") if x.strip()]
        print(json.dumps(_validate_csv(Path(args.csv), states), indent=2))
        return 0
    if args.command == "quicktest":
        return _run_quicktest(args)
    if args.command == "candidate-run":
        return _run_candidate(args)
    if args.command == "profile-patient":
        from .bayesian_evidence import bayesian_bootstrap_mean
        from .benchmarks import benchmark_patient_effects
        df=pd.read_csv(args.csv)
        missing=[c for c in (args.puc_col,args.npi_col) if c not in df.columns]
        if missing: raise SystemExit(f"Missing patient-effect columns: {missing}")
        u=df[args.puc_col].to_numpy(float); n=df[args.npi_col].to_numpy(float)
        harm=-n; k=max(1,int(__import__('math').ceil(.20*len(harm))))
        out={
            'schema':'qualifyot-patient-profile',
            'n_patients':int(len(df)),
            'utility_bayesian_bootstrap':bayesian_bootstrap_mean(u,draws=args.draws,seed=args.seed).to_dict(),
            'retention_bayesian_bootstrap':bayesian_bootstrap_mean(n,draws=args.draws,seed=args.seed+1).to_dict(),
            'utility_benchmarks':benchmark_patient_effects(u,bootstrap=args.bootstrap,bayes_draws=args.draws,seed=args.seed+2).to_dict(),
            'retention_benchmarks':benchmark_patient_effects(n,bootstrap=args.bootstrap,bayes_draws=args.draws,seed=args.seed+3).to_dict(),
            'tail_safety':{'tail_fraction':.20,'harmed_patient_fraction':float((harm>0).mean()),'upper_tail_cvar_harm':float(__import__('numpy').asarray(sorted(harm)[-k:],float).mean())},
            'note':'Orthogonal evidence profile; it does not modify the locked Movement–Utility–Retention qualification state.'
        }
        text=json.dumps(out,indent=2)
        if args.out: Path(args.out).write_text(text,encoding='utf-8')
        print(text); return 0
    if args.command == "graph-profile":
        from .partial_identification import graph_confidence_set
        from .bayesian_evidence import bayesian_bootstrap_graph_profile
        df=pd.read_csv(args.csv)
        if 'patient_id' not in df.columns: raise SystemExit('Missing column: patient_id')
        models=[x.strip() for x in args.models.split(',') if x.strip()]
        missing=[c for c in models if c not in df.columns]
        if missing: raise SystemExit(f"Missing graph-loss columns: {missing}")
        L=df[models].to_numpy(float); P=df.patient_id.astype(str).to_numpy()
        cs=graph_confidence_set(L,P,models,bootstrap=args.bootstrap,seed=args.seed)
        bb=bayesian_bootstrap_graph_profile(L,P,models,draws=args.draws,seed=args.seed+1)
        out={'schema':'qualifyot-graph-profile','confidence_set':cs.to_dict(),'bayesian_bootstrap_predictive':bb.to_dict(),
             'note':'Bayesian-bootstrap best-model frequencies are predictive library scores, not causal graph posterior probabilities.'}
        text=json.dumps(out,indent=2)
        if args.out: Path(args.out).write_text(text,encoding='utf-8')
        print(text); return 0

    raw = json.loads(Path(args.json_file).read_text(encoding="utf-8"))
    if args.command == "classify":
        legacy, _ = _classify_mapping(raw)
        core, _ = _classify_core_mapping(raw)
        if args.rule == "legacy":
            out = legacy.to_dict()
        elif args.rule == "core":
            out = core.to_dict()
            # Backward-compatible legacy audit counts remain available without
            # changing the primary three-axis decision.
            out["passed_gates"] = legacy.passed_gates
            out["total_gates"] = legacy.total_gates
            out["legacy_label"] = legacy.label
        else:
            out = {"core": core.to_dict(), "legacy": legacy.to_dict()}
        print(json.dumps(out, indent=2))
        return 0
    if args.command == "report":
        report = _make_report(raw)
        src = Path(args.json_file)
        prefix = Path(args.out_prefix) if args.out_prefix else src.with_name(src.stem + "_qualifyot_report")
        json_path = prefix.with_suffix(".json")
        md_path = prefix.with_suffix(".md")
        json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        md_path.write_text(_report_markdown(report), encoding="utf-8")
        print(json.dumps({"json": str(json_path), "markdown": str(md_path), "evidence_state": report["evidence_state"]}, indent=2))
        return 0
    raise SystemExit(2)


if __name__ == "__main__":
    raise SystemExit(main())
