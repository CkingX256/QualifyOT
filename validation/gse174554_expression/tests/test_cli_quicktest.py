from pathlib import Path

from qualifyot.cli import main


def test_cli_quicktest_writes_html(tmp_path, capsys):
    root = Path(__file__).resolve().parents[1]
    csv = root / "data" / "processed_pairs" / "GSE123813_pairs.csv"
    out = tmp_path / "quick.html"
    rc = main([
        "quicktest", str(csv),
        "--states", "Memory,Effector,Exhausted,Regulatory,Other_T",
        "--graph", "Memory->Effector,Effector->Exhausted,Memory->Exhausted",
        "--bootstrap", "20",
        "--out", str(out),
    ])
    assert rc == 0
    _ = capsys.readouterr()
    text = out.read_text(encoding="utf-8")
    assert "QualifyOT rapid diagnostic" in text
    assert "Provisional evidence state" in text
    assert "Patient-level retained NPI contributions" in text
