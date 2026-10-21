"""Evaluation metrics from explicit predictions and reference labels only."""
def binary_metrics(predicted, reference):
    if len(predicted)!=len(reference): raise ValueError("Prediction/reference lengths differ")
    tp=sum(bool(p) and bool(r) for p,r in zip(predicted,reference))
    fp=sum(bool(p) and not bool(r) for p,r in zip(predicted,reference))
    fn=sum(not bool(p) and bool(r) for p,r in zip(predicted,reference))
    precision=tp/(tp+fp) if tp+fp else 0.0; recall=tp/(tp+fn) if tp+fn else 0.0
    return {"precision":precision,"recall":recall,"f1":2*precision*recall/(precision+recall) if precision+recall else 0.0,
            "false_positive_rate":fp/(fp+sum(not bool(p) and not bool(r) for p,r in zip(predicted,reference))) if fp+sum(not bool(r) for r in reference) else 0.0,
            "counts":{"tp":tp,"fp":fp,"fn":fn}}
