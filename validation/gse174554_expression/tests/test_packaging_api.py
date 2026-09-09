from qualifyot.candidate_api import PrecomputedPredictionCandidate


def test_public_api_exposes_precomputed_candidate():
    assert PrecomputedPredictionCandidate is not None
