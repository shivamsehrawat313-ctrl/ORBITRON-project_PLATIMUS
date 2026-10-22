"""Pixel-level evaluation from explicit predictions and reference labels only."""
def summarize_change_area(change_mask, valid_mask=None):
    """Return changed-pixel share over the valid common image region."""
    import numpy as np
    changed=np.asarray(change_mask,dtype=bool)
    valid=np.ones(changed.shape,dtype=bool) if valid_mask is None else np.asarray(valid_mask,dtype=bool)
    if changed.ndim!=2 or valid.shape!=changed.shape:
        raise ValueError("Change and valid masks must be matching 2D rasters")
    changed_pixels=int(np.count_nonzero(changed & valid))
    valid_pixels=int(np.count_nonzero(valid))
    return {"changed_pixels":changed_pixels,"valid_pixels":valid_pixels,
            "percent_of_region":100.0*changed_pixels/valid_pixels if valid_pixels else 0.0}

def binary_metrics(predicted, reference, valid_mask=None):
    import numpy as np

    pred = np.asarray(predicted, dtype=bool)
    ref = np.asarray(reference, dtype=bool)
    if pred.shape != ref.shape:
        raise ValueError("Prediction/reference shapes differ")
    valid = np.ones(pred.shape, dtype=bool) if valid_mask is None else np.asarray(valid_mask, dtype=bool)
    if valid.shape != pred.shape:
        raise ValueError("Valid mask must match prediction/reference shape")
    pred = pred[valid]
    ref = ref[valid]
    tp = int(np.count_nonzero(pred & ref))
    fp = int(np.count_nonzero(pred & ~ref))
    fn = int(np.count_nonzero(~pred & ref))
    tn = int(np.count_nonzero(~pred & ~ref))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    union = tp + fp + fn
    return {
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "iou": tp / union if union else 0.0,
        "false_positive_rate": fp / (fp + tn) if fp + tn else 0.0,
        "counts": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
        "valid_pixels": int(valid.sum()),
    }
