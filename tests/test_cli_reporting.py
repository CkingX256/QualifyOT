import json
from pathlib import Path

from qualifyot.cli import main


def _metrics():
    return {
        "PDR": 0.61,
        "PDR_CI2.5": 0.31,
        "PUC_CI2.5": 0.011,
        "PUC_CI97.5": 0.035,
        "NPI_CI2.5": 0.0025,
        "positive_weight_fold_fraction": 7/9,
        "final_risk": 0.08,
        "reference_risk": 0.09,
        "perturbation_stability": {"full_pipeline_qualified_fraction": 0.09},
    }


def test_cli_classify_accepts_metric_json(tmp_path, capsys):
    p = tmp_path / "metrics.json"
    p.write_text(json.dumps(_metrics()))
    assert main(["classify", str(p)]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["label"] == "Qualified"
    assert out["passed_gates"] == 6


def test_cli_report_writes_machine_and_human_readable_outputs(tmp_path, capsys):
    p = tmp_path / "metrics.json"
    p.write_text(json.dumps(_metrics()))
    prefix = tmp_path / "report"
    assert main(["report", str(p), "--out-prefix", str(prefix)]) == 0
    _ = capsys.readouterr()
    j = json.loads((tmp_path / "report.json").read_text())
    md = (tmp_path / "report.md").read_text()
    assert j["schema"] == "qualifyot-report"
    assert j["evidence_state"] == "Qualified"
    assert "Perturbation stability" in md


def test_cli_validate_accepts_archive_source_target_column_names(tmp_path, capsys):
    import pandas as pd
    p = tmp_path / "pairs.csv"
    pd.DataFrame({
        "patient_id": ["p1", "p2"],
        "source__A": [0.7, 0.4], "source__B": [0.3, 0.6],
        "target__A": [0.6, 0.5], "target__B": [0.4, 0.5],
    }).to_csv(p, index=False)
    assert main(["validate-csv", str(p), "--states", "A,B"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["valid"] is True and out["patients"] == 2
