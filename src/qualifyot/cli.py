"""Command-line inspection and reporting utilities for QualifyOT.

The CLI validates standardized longitudinal-pair tables, applies the frozen six-gate
classification rule to saved metrics, emits compact machine-readable / Markdown reports,
and provides a rapid GraphFlow quicktest for exploratory local diagnostics.
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
    p = argparse.ArgumentParser(prog="qualifyot-inspect")
    sub = p.add_subparsers(dest="command", required=True)

    v = sub.add_parser("validate-csv", help="Validate a standardized longitudinal-pair CSV")
    v.add_argument("csv")
    v.add_argument("--states", required=True, help="Comma-separated state names")

    e = sub.add_parser("classify", help="Classify a saved metric JSON using the frozen six-gate rule")
    e.add_argument("json_file")

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


def _make_report(raw: Mapping[str, Any]) -> dict[str, Any]:
    state, q = _classify_mapping(raw)
    stability = raw.get("perturbation_stability") or raw.get("stability") or {}
    point_metrics = {
        k: raw.get(k) for k in (
            "PUC", "PUC_point", "NPI", "NPI_point", "PDR", "PDR_point",
            "candidate_risk", "reference_risk", "final_risk"
        ) if k in raw
    }
    return {
        "schema": "qualifyot-report",
        "evidence_state": state.label,
        "passed_gates": state.passed_gates,
        "total_gates": state.total_gates,
        "gates": state.gates,
        "limiting_gates": list(state.limiting_gates),
        "classification_inputs": q,
        "reported_point_metrics": point_metrics,
        "perturbation_stability": stability,
        "interpretation": (
            "Locked six-gate state and perturbation stability are reported separately; "
            "hardening does not retroactively change the frozen fixed-analysis label."
        ),
    }


def _report_markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# QualifyOT Report",
        "",
        f"**Evidence state:** {report['evidence_state']}",
        f"**Gates passed:** {report['passed_gates']}/{report['total_gates']}",
        "",
        "## Six-gate audit",
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


def _quicktest_html(result: Mapping[str, Any], state, runtime_s: float, source_name: str, bootstrap: int) -> str:
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
    gates = "".join(
        f'<tr><td>{html.escape(k)}</td><td class="{"pass" if v else "limit"}">{"PASS" if v else "LIMIT"}</td></tr>'
        for k, v in state.gates.items()
    )
    lim = ", ".join(state.limiting_gates) if state.limiting_gates else "None"

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
<div class="grid"><div><h2>Provisional evidence state</h2><p style="font-size:26px;font-weight:700">{html.escape(state.label)}</p><p><b>Limiting gates:</b> {html.escape(lim)}</p>
<table><tr><th>Metric</th><th>Estimate [95% interval]</th></tr><tr><td>PDR</td><td class="metric">{result['PDR']:.4f} [{result['PDR_lo']:.4f}, {result['PDR_hi']:.4f}]</td></tr>
<tr><td>PUC</td><td class="metric">{sf(result['PUC'])} [{sf(result['PUC_lo'])}, {sf(result['PUC_hi'])}]</td></tr><tr><td>NPI</td><td class="metric">{sf(result['NPI'])} [{sf(result['NPI_lo'])}, {sf(result['NPI_hi'])}]</td></tr>
<tr><td>Positive-weight folds</td><td class="metric">{100*result['positive_weight_fold_fraction']:.1f}%</td></tr></table></div>
<div><h2>Six-gate screen</h2><table><tr><th>Gate</th><th>Status</th></tr>{gates}</table></div></div>
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
    state = classify(
        res["PDR"], res["PDR_lo"], res["PUC_lo"], res["PUC_hi"], res["NPI_lo"],
        res["positive_weight_fold_fraction"], res["qualified_risk"], res["reference_risk"], GateConfig()
    )
    runtime = time.perf_counter() - t0
    out = Path(args.out) if args.out else Path(args.csv).with_name(Path(args.csv).stem + "_qualifyot_quicktest.html")
    out.write_text(_quicktest_html(res, state, runtime, Path(args.csv).name, args.bootstrap), encoding="utf-8")
    print(json.dumps({"html": str(out), "provisional_evidence_state": state.label, "runtime_seconds": runtime, "bootstrap": args.bootstrap}, indent=2))
    return 0


def main(argv=None):
    args = _parse_args(argv)
    if args.command == "validate-csv":
        states = [x.strip() for x in args.states.split(",") if x.strip()]
        print(json.dumps(_validate_csv(Path(args.csv), states), indent=2))
        return 0
    if args.command == "quicktest":
        return _run_quicktest(args)

    raw = json.loads(Path(args.json_file).read_text(encoding="utf-8"))
    if args.command == "classify":
        out, _ = _classify_mapping(raw)
        print(json.dumps(out.to_dict(), indent=2))
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
