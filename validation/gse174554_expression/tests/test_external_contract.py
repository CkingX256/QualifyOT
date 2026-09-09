from pathlib import Path

import yaml

from qualifyot.external_contract import (
    contract_sha256,
    deterministic_patient_split,
    patient_set_sha256,
    verify_contract_digest,
    write_contract_digest,
)
from qualifyot.pipeline import PipelineConfig
from qualifyot.inference import FrozenAnalysisContract


def test_public_pipeline_defaults_to_inference_margin_contrast():
    assert PipelineConfig().movement_core_mode == "margin_contrast"


def test_frozen_contract_patient_identity_changes_hash():
    a = FrozenAnalysisContract(
        "candidate", "MeanDelta", ("A", "B"), "MAE", .01, (0, .5, 1), "honest",
        patient_ids=("P1", "P2"),
    )
    b = FrozenAnalysisContract(
        "candidate", "MeanDelta", ("A", "B"), "MAE", .01, (0, .5, 1), "honest",
        patient_ids=("P1", "P3"),
    )
    assert a.sha256 != b.sha256


def test_patient_set_hash_is_order_and_duplicate_invariant():
    assert patient_set_sha256(["P2", "P1", "P1"]) == patient_set_sha256(["P1", "P2"])


def test_contract_digest_semantic_hash_and_verification(tmp_path: Path):
    p = tmp_path / "contract.yaml"
    p.write_text(yaml.safe_dump({"dataset": "X", "loss": "MAE"}, sort_keys=False))
    d = write_contract_digest(p)
    assert len(contract_sha256(p)) == 64
    assert verify_contract_digest(p, d)
    p.write_text(yaml.safe_dump({"dataset": "X", "loss": "Hellinger"}, sort_keys=False))
    try:
        verify_contract_digest(p, d)
    except ValueError:
        pass
    else:
        raise AssertionError("mutated contract must fail verification")


def test_deterministic_patient_split_is_order_invariant_and_nonempty():
    ids = [f"P{i:02d}" for i in range(20)]
    a = deterministic_patient_split(ids, development_fraction=.60, salt="qualifyot-ext-v1")
    b = deterministic_patient_split(list(reversed(ids)), development_fraction=.60, salt="qualifyot-ext-v1")
    assert a == b
    assert set(a.values()) == {"development", "confirmation"}


def test_localized_honest_candidate_iut_pvalue_is_max_component():
    import numpy as np
    from qualifyot.inference import honest_confirmation_pvalues_localized
    gm = np.array([0.010, 0.012, 0.009, 0.011])
    a = np.array([0.020, 0.022, 0.019, 0.021])
    u = np.array([0.05, 0.04, 0.06, 0.05])
    cu = np.array([0.10, 0.10, 0.10, 0.10])
    r = np.array([0.03, 0.02, 0.04, 0.03])
    cr = np.array([0.08, 0.08, 0.08, 0.08])
    out = honest_confirmation_pvalues_localized(gm, a, u, cu, r, cr, movement_margin=.01)
    assert 0 <= out["candidate_iut"] <= 1
    assert out["candidate_iut"] == max(out["movement"], out["utility"], out["retention"])


def test_target_access_gate_detects_patient_or_split_mutation(tmp_path):
    import yaml
    from qualifyot.external_contract import freeze_patient_set_and_split, verify_target_access_gate
    c = tmp_path / "contract.yaml"
    c.write_text(yaml.safe_dump({"inference":{"split_salt":"frozen-salt","development_fraction":0.6}}))
    p = tmp_path / "patients.txt"; s = tmp_path / "split.csv"; g = tmp_path / "gate.json"
    gate = freeze_patient_set_and_split(
        [f"P{i:02d}" for i in range(20)], contract_path=c, patient_file=p, split_file=s, gate_file=g
    )
    assert gate["n_patients"] == 20
    assert gate["n_development"] + gate["n_confirmation"] == 20
    assert verify_target_access_gate(contract_path=c, patient_file=p, split_file=s, gate_file=g)
    original = p.read_text()
    p.write_text(original + "P99\n")
    import pytest
    with pytest.raises(ValueError, match="patient set changed"):
        verify_target_access_gate(contract_path=c, patient_file=p, split_file=s, gate_file=g)


def test_heterogeneous_hoeffding_pvalue_matches_lcb_rejection_boundary():
    import numpy as np
    from qualifyot.inference import heterogeneous_hoeffding_lcb, heterogeneous_hoeffding_pvalue
    x = np.array([0.08, 0.09, 0.07, 0.10, 0.08, 0.09])
    lo = np.full(len(x), -0.10)
    hi = np.full(len(x), 0.10)
    alpha = 0.05
    lcb = heterogeneous_hoeffding_lcb(x, lo, hi, alpha=alpha)
    p = heterogeneous_hoeffding_pvalue(x, lo, hi)
    assert (lcb > 0) == (p < alpha)


def test_hardening_helpers_are_public_exports():
    import qualifyot
    required = {
        "heterogeneous_hoeffding_pvalue",
        "honest_confirmation_pvalues_localized",
        "contract_sha256",
        "patient_set_sha256",
        "freeze_patient_set_and_split",
        "verify_target_access_gate",
    }
    assert required.issubset(set(qualifyot.__all__))
    for name in required:
        assert hasattr(qualifyot, name)


def test_frozen_contract_patient_set_is_canonical_and_self_hashing():
    from qualifyot.external_contract import patient_set_sha256
    c = FrozenAnalysisContract(
        "candidate", "MeanDelta", ("A", "B"), "MAE", .01, (0, .5, 1), "honest",
        patient_ids=(" P2 ", "P1", "P1"),
    )
    assert c.patient_ids == ("P1", "P2")
    assert c.patient_set_sha256 == patient_set_sha256(["P1", "P2"])
    d = FrozenAnalysisContract(
        "candidate", "MeanDelta", ("A", "B"), "MAE", .01, (0, .5, 1), "honest",
        patient_ids=("P2", "P1"),
    )
    assert c.sha256 == d.sha256


def test_frozen_contract_rejects_mismatched_patient_hash():
    import pytest
    with pytest.raises(ValueError, match="does not match"):
        FrozenAnalysisContract(
            "candidate", "MeanDelta", ("A", "B"), "MAE", .01, (0, .5, 1), "honest",
            patient_ids=("P1", "P2"), patient_set_sha256="0" * 64,
        )


def test_cli_external_contract_roundtrip(tmp_path, capsys):
    import json
    import yaml
    from qualifyot.cli import main
    c = tmp_path / "contract.yaml"
    c.write_text(yaml.safe_dump({"inference": {"split_salt": "locked", "development_fraction": 0.6}}))
    ids = tmp_path / "ids.txt"
    ids.write_text("P01\nP02\nP03\nP04\nP05\nP06\nP07\nP08\nP09\nP10\n")
    prefix = tmp_path / "frozen"
    assert main(["contract-hash", str(c)]) == 0
    h = json.loads(capsys.readouterr().out)
    assert len(h["semantic_sha256"]) == 64
    assert main(["freeze-patients", "--contract", str(c), "--ids", str(ids), "--out-prefix", str(prefix)]) == 0
    gate = json.loads(capsys.readouterr().out)
    assert gate["n_patients"] == 10
    assert main([
        "verify-target-access", "--contract", str(c),
        "--patients", str(prefix.with_suffix(".patients.txt")),
        "--split", str(prefix.with_suffix(".split.csv")),
        "--gate", str(prefix.with_suffix(".target_access_gate.json")),
    ]) == 0
    assert json.loads(capsys.readouterr().out)["target_access_gate_valid"] is True


def test_target_access_gate_hashes_dependency_closure(tmp_path):
    import json
    import pytest
    import yaml
    from qualifyot.external_contract import freeze_patient_set_and_split, verify_target_access_gate
    cohort = tmp_path / "cohort.yaml"
    cohort.write_text(yaml.safe_dump({"inference": {"split_salt": "s", "development_fraction": 0.6}}))
    master = tmp_path / "master.yaml"
    master.write_text(yaml.safe_dump({"candidate_family": ["A", "B"]}))
    registry = tmp_path / "methods.yaml"
    registry.write_text(yaml.safe_dump({"A": {"version": "1.0"}}))
    p = tmp_path / "patients.txt"; s = tmp_path / "split.csv"; g = tmp_path / "gate.json"
    freeze_patient_set_and_split(
        [f"P{i:02d}" for i in range(20)], contract_path=cohort,
        patient_file=p, split_file=s, gate_file=g, dependency_paths=[master, registry],
    )
    gate = json.loads(g.read_text())
    assert {d["file"] for d in gate["dependencies"]} == {"master.yaml", "methods.yaml"}
    assert verify_target_access_gate(
        contract_path=cohort, patient_file=p, split_file=s, gate_file=g,
        dependency_paths=[master, registry],
    )
    registry.write_text(yaml.safe_dump({"A": {"version": "2.0"}}))
    with pytest.raises(ValueError, match="dependency changed"):
        verify_target_access_gate(
            contract_path=cohort, patient_file=p, split_file=s, gate_file=g,
            dependency_paths=[master, registry],
        )


def test_generic_external_engine_defaults_to_movement_margin_contrast():
    import inspect
    from qualifyot.generic_engine import run_generic_lopo
    assert inspect.signature(run_generic_lopo).parameters["use_movement_contrast_for_core"].default is True
