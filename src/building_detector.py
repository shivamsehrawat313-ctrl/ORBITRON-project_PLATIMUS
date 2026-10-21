"""SpaceNet-2 DeepLabV3+ footprint inference with explicit logits contract."""
from pathlib import Path

class ModelUnavailable(RuntimeError): pass
class InferenceFailed(RuntimeError): pass

class BuildingDetector:
    """DeepLabV3 ResNet-50 trained on SpaceNet-2 Paris RGB tiles.

    The upstream model emits one-channel *logits*. We apply sigmoid exactly once.
    Input preprocessing matches the upstream notebook: scene-global min/max over
    RGB bands, float32 in [0, 1]. Predictions are tiled and overlap-averaged.
    """
    def __init__(self, checkpoint=None, device=None):
        self.checkpoint=Path(checkpoint) if checkpoint else None
        self.device=device
        self.model=None
        self.torch=None
    @property
    def status(self): return "Model ready" if self.model is not None else "Model unavailable"
    def load(self):
        if self.checkpoint is None or not self.checkpoint.is_file():
            raise ModelUnavailable("DeepLab checkpoint not found. Download the documented SpaceNet model first.")
        try:
            import torch
            from torch import nn
            from torchvision.models.segmentation import deeplabv3_resnet50
            self.torch=torch
            self.device=self.device or ("cuda" if torch.cuda.is_available() else "cpu")
            # The published training code starts from the torchvision COCO/VOC
            # DeepLabV3-ResNet50 architecture, then replaces classifier[4].
            model=deeplabv3_resnet50(weights=None,weights_backbone=None,num_classes=21,aux_loss=True)
            model.classifier[4]=nn.Conv2d(256,1,kernel_size=1)
            state=torch.load(self.checkpoint,map_location="cpu",weights_only=True)
            if isinstance(state,dict) and "state_dict" in state: state=state["state_dict"]
            # The published checkpoint was saved from a wrapper whose only
            # submodule was named `model`; remove that outer namespace.
            if state and all(k.startswith("model.") for k in state):
                state={k[len("model."):]:v for k,v in state.items()}
            model.load_state_dict(state,strict=True)
            self.model=model.to(self.device).eval()
        except Exception as e:
            raise ModelUnavailable(f"DeepLab checkpoint could not be loaded: {e}") from e
        return self
    def predict(self, rgb, threshold=.5, tile_size=512, overlap=128, valid_mask=None):
        """Return (probability raster, binary mask); `rgb` is HxWx3/4 array.

        Tile predictions overlap-average continuous logits before sigmoid to
        reduce seams. RGB values are normalized scene-globally to match training.
        """
        if self.model is None: raise ModelUnavailable("Load the DeepLab checkpoint before inference.")
        if not 0 < threshold < 1: raise ValueError("threshold must be between 0 and 1")
        if tile_size < 64 or overlap < 0 or overlap >= tile_size: raise ValueError("Require tile_size >= 64 and 0 <= overlap < tile_size")
        try:
            import numpy as np
            torch=self.torch
            x=np.asarray(rgb)
            if x.ndim != 3 or x.shape[2] < 3 or not x.shape[0] or not x.shape[1]:
                raise ValueError("Expected non-empty HxWx3 RGB image")
            x=x[:,:,:3].astype(np.float32)
            valid=np.isfinite(x).all(axis=2)
            if valid_mask is not None:
                if np.asarray(valid_mask).shape!=valid.shape: raise ValueError("valid_mask must match image dimensions")
                valid &= np.asarray(valid_mask,dtype=bool)
            if not valid.any(): raise ValueError("Image contains no finite RGB pixels")
            low=float(x[valid].min()); high=float(x[valid].max())
            x=np.nan_to_num(x,nan=low,posinf=high,neginf=low)
            x=np.clip((x-low)/max(high-low,1e-6),0,1)
            h,w=x.shape[:2]; step=tile_size-overlap
            xs=list(range(0,max(w-tile_size,0)+1,step)); ys=list(range(0,max(h-tile_size,0)+1,step))
            if not xs or xs[-1]+tile_size<w: xs.append(max(0,w-tile_size))
            if not ys or ys[-1]+tile_size<h: ys.append(max(0,h-tile_size))
            logits_sum=np.zeros((h,w),dtype=np.float32); counts=np.zeros((h,w),dtype=np.uint16)
            for y in ys:
                for x0 in xs:
                    tile=x[y:min(y+tile_size,h),x0:min(x0+tile_size,w),:]
                    th,tw=tile.shape[:2]
                    if th<tile_size or tw<tile_size:
                        mode="reflect" if th>1 and tw>1 else "edge"
                        tile=np.pad(tile,((0,tile_size-th),(0,tile_size-tw),(0,0)),mode=mode)
                    tensor=torch.from_numpy(np.ascontiguousarray(tile.transpose(2,0,1))).unsqueeze(0).to(self.device)
                    with torch.inference_mode(): logits=self.model(tensor)["out"][0,0,:th,:tw].float().cpu().numpy()
                    logits_sum[y:y+th,x0:x0+tw]+=logits
                    counts[y:y+th,x0:x0+tw]+=1
            logits=logits_sum/np.maximum(counts,1)
            probs=1.0/(1.0+np.exp(-np.clip(logits,-80,80)))
            return probs.astype(np.float32),probs>=threshold
        except Exception as e:
            if isinstance(e,ValueError): raise
            raise InferenceFailed(f"DeepLab inference failed: {e}") from e
