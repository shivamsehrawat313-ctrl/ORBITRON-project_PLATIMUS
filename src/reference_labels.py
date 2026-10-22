"""Decode optional held-out binary reference change masks."""
from io import BytesIO


def decode_change_label(raw, filename, expected_shape):
    """Return a boolean label mask; any nonzero label value means changed."""
    if not str(filename).lower().endswith(".png"):
        raise ValueError("Reference change labels must be PNG files.")
    try:
        import numpy as np
        from PIL import Image
        with Image.open(BytesIO(raw)) as image:
            mask=np.asarray(image.convert("L"))
    except Exception as exc:
        raise ValueError(f"Reference label could not be decoded: {exc}") from exc
    if mask.shape != tuple(expected_shape):
        raise ValueError(
            f"Reference label dimensions {mask.shape[1]} × {mask.shape[0]} do not match "
            f"the image dimensions {expected_shape[1]} × {expected_shape[0]}. "
            "Choose the matching label for this exact image pair."
        )
    return mask > 0
