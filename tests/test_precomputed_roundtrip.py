import numpy as np
import pandas as pd
from qualifyot.candidate_api import PrecomputedPredictionCandidate

def test_precomputed_prediction_roundtrip_exact():
    states=["A","B","C"]
    df=pd.DataFrame({
        "external__A":[0.2,0.1],
        "external__B":[0.3,0.2],
        "external__C":[0.5,0.7],
    })
    c=PrecomputedPredictionCandidate(states,prefix="external",name="roundtrip").fit(df)
    got=c.predict(df)
    expected=df[["external__A","external__B","external__C"]].to_numpy(float)
    assert np.array_equal(got, expected)
    assert np.allclose(got.sum(axis=1),1.0)
