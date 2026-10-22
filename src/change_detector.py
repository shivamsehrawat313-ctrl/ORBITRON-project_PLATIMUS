"""LEVIR-CD-tuned BIT change segmentation through its published ONNX checkpoint."""
from pathlib import Path


class ChangeModelUnavailable(RuntimeError):
    pass


class LevirChangeDetector:
    """Two-date change detector; infer overlapping native-resolution 256px crops."""

    input_size = 256
    tile_overlap = 64
    model_name = "BIT change detector fine-tuned on LEVIR-CD"

    def __init__(self, checkpoint=None, providers=None):
        self.checkpoint = Path(checkpoint) if checkpoint else None
        self.providers = providers or ["CPUExecutionProvider"]
        self.session = None

    @property
    def status(self):
        return "Model ready" if self.session is not None else "Model unavailable"

    def load(self):
        if self.checkpoint is None or not self.checkpoint.is_file():
            raise ChangeModelUnavailable(
                "LEVIR change checkpoint missing. Run scripts\\download_levir_change_model.py."
            )
        try:
            import onnxruntime as ort
            self.session = ort.InferenceSession(str(self.checkpoint), providers=self.providers)
            inputs = self.session.get_inputs()
            outputs = self.session.get_outputs()
            expected = [(1, 3, 256, 256), (1, 3, 256, 256)]
            actual = [tuple(item.shape) for item in inputs]
            if actual != expected or len(outputs) != 1 or tuple(outputs[0].shape) != (1, 2, 256, 256):
                raise ValueError(f"Unexpected ONNX contract: inputs={actual}, outputs={[x.shape for x in outputs]}")
        except Exception as exc:
            raise ChangeModelUnavailable(f"LEVIR change checkpoint could not load: {exc}") from exc
        return self

    @staticmethod
    def _preprocess(rgb):
        import numpy as np
        from PIL import Image

        image = np.asarray(rgb)
        if image.ndim != 3 or image.shape[2] < 3 or image.shape[:2] != (256, 256):
            raise ValueError("Expected an RGB 256x256 model tile; resize the scene through tiling, not downsampling")
        image = image[:, :, :3]
        if image.dtype != np.uint8:
            finite = image[np.isfinite(image)]
            if not finite.size:
                raise ValueError("Image has no finite RGB values")
            low, high = float(finite.min()), float(finite.max())
            image = np.nan_to_num(image, nan=low, posinf=high, neginf=low)
            image = np.clip((image - low) / max(high - low, 1e-6) * 255, 0, 255).astype(np.uint8)
        array = np.asarray(Image.fromarray(image, mode="RGB"), dtype=np.float32).transpose(2, 0, 1)[None]
        # Match the reference BIT/LEVIR preprocessing: ToTensor then mean/std 0.5.
        return array / 127.5 - 1.0

    def predict_pair(self, rgb_before, rgb_after, threshold=0.5):
        """Return full-size changed-class probabilities and thresholded binary mask."""
        if self.session is None:
            raise ChangeModelUnavailable("Load the LEVIR change checkpoint before inference.")
        if not 0 < threshold < 1:
            raise ValueError("Change probability threshold must be between 0 and 1")
        import numpy as np
        from PIL import Image

        a, b = np.asarray(rgb_before), np.asarray(rgb_after)
        if a.shape[:2] != b.shape[:2]:
            raise ValueError("Before and after images must have matching dimensions")
        if a.ndim != 3 or b.ndim != 3 or a.shape[2] < 3 or b.shape[2] < 3:
            raise ValueError("Before and after inputs must be RGB images")
        a, b = self._to_uint8(a[:, :, :3]), self._to_uint8(b[:, :, :3])
        h, w = a.shape[:2]
        stride = self.input_size - self.tile_overlap

        def starts(length):
            if length <= self.input_size:
                return [0]
            positions = list(range(0, length - self.input_size + 1, stride))
            if positions[-1] + self.input_size < length:
                positions.append(length - self.input_size)
            return positions

        x_positions, y_positions = starts(w), starts(h)
        probability_sum = np.zeros((h, w), dtype=np.float32)
        weight_sum = np.zeros((h, w), dtype=np.uint16)
        names = [item.name for item in self.session.get_inputs()]
        for y in y_positions:
            for x in x_positions:
                tile_a = a[y:min(y + self.input_size, h), x:min(x + self.input_size, w)]
                tile_b = b[y:min(y + self.input_size, h), x:min(x + self.input_size, w)]
                th, tw = tile_a.shape[:2]
                if th < self.input_size or tw < self.input_size:
                    pad = ((0, self.input_size - th), (0, self.input_size - tw), (0, 0))
                    mode = "reflect" if th > 1 and tw > 1 else "edge"
                    tile_a = np.pad(tile_a, pad, mode=mode)
                    tile_b = np.pad(tile_b, pad, mode=mode)
                logits = np.asarray(self.session.run(None, {
                    names[0]: self._preprocess(tile_a), names[1]: self._preprocess(tile_b)
                })[0], dtype=np.float32)
                if logits.shape != (1, 2, 256, 256):
                    raise ValueError(f"Expected 1x2x256x256 change logits, got {logits.shape}")
                logits -= logits.max(axis=1, keepdims=True)
                exp = np.exp(logits)
                tile_probability = exp[0, 1] / exp.sum(axis=1)[0]
                probability_sum[y:y + th, x:x + tw] += tile_probability[:th, :tw]
                weight_sum[y:y + th, x:x + tw] += 1
        probability = probability_sum / np.maximum(weight_sum, 1)
        return probability, probability >= threshold

    @staticmethod
    def _to_uint8(image):
        """Keep uint8 LEVIR values; stretch non-uint8 rasters once per image."""
        import numpy as np
        image = np.asarray(image)
        if image.dtype == np.uint8:
            return image
        finite = image[np.isfinite(image)]
        if not finite.size:
            raise ValueError("Image has no finite RGB values")
        low, high = float(finite.min()), float(finite.max())
        image = np.nan_to_num(image, nan=low, posinf=high, neginf=low)
        return np.clip((image - low) / max(high - low, 1e-6) * 255, 0, 255).astype(np.uint8)
