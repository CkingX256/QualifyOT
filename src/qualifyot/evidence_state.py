from __future__ import annotations
from dataclasses import dataclass,asdict

@dataclass(frozen=True)
class GateConfig:
    pdr_min: float=.05
    pdr_lcb_min: float=.01
    puc_lcb_min: float=0.0
    npi_lcb_min: float=0.0
    positive_weight_fold_fraction_min: float=.70
    require_final_risk_below_reference: bool=True

@dataclass(frozen=True)
class EvidenceState:
    label: str
    gates: dict
    limiting_gates: tuple
    passed_gates: int
    total_gates: int=6
    def to_dict(self): return asdict(self)

def classify(pdr,pdr_lo,puc_lo,puc_hi,npi_lo,positive_weight_fraction,final_risk,reference_risk,cfg:GateConfig=GateConfig()):
    gates={
      'PDR point':pdr>cfg.pdr_min,
      'PDR lower bound':pdr_lo>cfg.pdr_lcb_min,
      'PUC lower bound':puc_lo>cfg.puc_lcb_min,
      'positive-weight folds':positive_weight_fraction>=cfg.positive_weight_fold_fraction_min,
      'NPI lower bound':npi_lo>cfg.npi_lcb_min,
      'final risk':(final_risk<reference_risk) if cfg.require_final_risk_below_reference else True,
    }
    expressive=gates['PDR point'] and gates['PDR lower bound']
    if all(gates.values()): label='Qualified'
    elif not expressive: label='Negligible'
    elif puc_hi<0: label='Adverse'
    elif puc_lo>cfg.puc_lcb_min: label='Promising'
    else: label='Equivocal'
    bad=tuple(k for k,v in gates.items() if not v)
    return EvidenceState(label,gates,bad,sum(gates.values()))
