"""Review scoring helpers; heuristic scores are never model probabilities."""
def review_score(model_confidence=None, similarity=None, reliable=True):
    parts=[v for v in (model_confidence,similarity) if v is not None]
    score=sum(parts)/len(parts) if parts else .35
    return max(0.,min(1.,score if reliable else score*.5))
