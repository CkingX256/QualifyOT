from __future__ import annotations

"""Empirical null-calibration utilities.

Null-derived critical values are kept as *diagnostic calibration maps*.  They
must not silently replace a scientifically pre-specified minimal-relevance
margin such as delta_M, because doing so would make the meaning of Movement
sample-size and simulation-model dependent.
"""

from dataclasses import asdict, dataclass
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class EmpiricalCriticalValue:
    n: int
    statistic: str
    alpha: float
    critical_value: float
    simulations: int
    empirical_exceedance: float

    def to_dict(self):
        return asdict(self)


def empirical_upper_critical_value(
    null_values,
    *,
    n: int,
    statistic: str,
    alpha: float = 0.05,
) -> EmpiricalCriticalValue:
    x = np.asarray(null_values, float).reshape(-1)
    if len(x) < 100:
        raise ValueError("at least 100 null simulations are required")
    if not np.isfinite(x).all():
        raise ValueError("null_values must be finite")
    if not (0 < alpha < 1):
        raise ValueError("alpha must lie in (0,1)")
    # 'higher' is conservative for a finite empirical null sample.
    q = float(np.quantile(x, 1 - alpha, method="higher"))
    exc = float(np.mean(x > q))
    return EmpiricalCriticalValue(int(n), str(statistic), float(alpha), q, int(len(x)), exc)


def calibration_table(
    frame: pd.DataFrame,
    *,
    n_col: str = "n",
    statistic_col: str = "statistic",
    value_col: str = "value",
    alpha: float = 0.05,
) -> pd.DataFrame:
    needed = {n_col, statistic_col, value_col}
    if not needed.issubset(frame.columns):
        raise KeyError(f"missing columns: {sorted(needed-set(frame.columns))}")
    rows=[]
    for (n,stat),g in frame.groupby([n_col,statistic_col],sort=True):
        r=empirical_upper_critical_value(g[value_col],n=int(n),statistic=str(stat),alpha=alpha)
        rows.append(r.to_dict())
    return pd.DataFrame(rows)
