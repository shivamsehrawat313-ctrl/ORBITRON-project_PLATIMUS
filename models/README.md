# Building segmentation checkpoint

## Selected checkpoint

- Model: DeepLabV3 with a ResNet-50 backbone, one-channel building-logit output.
- Training domain: SpaceNet 2 Paris high-resolution RGB imagery, 512 × 512 patches.
- Source code/model card: [author repository](https://github.com/AryakBhattacharya/building-footprint-extraction) and [Hugging Face model repository](https://huggingface.co/Abhatta7/building-footprint-segmentation).
- License declared by those repositories: MIT. Review repository terms and upstream imagery/data terms before redistributing derived products.
- Architecture: Torchvision `deeplabv3_resnet50`, initialized in the published notebook with default Torchvision weights; its classifier's final layer is replaced with `Conv2d(256, 1, 1)`. The trained state dict includes the `model.` wrapper prefix, which the loader strips before strict loading.
- Input: first three RGB bands, scene-global min/max scaling to float32 `[0,1]`, 512-pixel tiles with overlap.
- Output: one-channel **logits**; this application applies sigmoid once and thresholds the resulting probabilities.
- Weight: `best_deeplab.pth` (168,357,171 bytes), revision `e34a637`, SHA-256 `170c49e1f42737f5a4b3922b9fc58b25b6ba0a7dd7bf00b9186a92ccb84dc555`.

The upstream model card reports validation scores but does not provide enough detail here to reproduce or independently verify its exact split. This project does not reuse those claims as evaluation results. The model is a research prototype trained for Paris imagery; performance on other sensors, resolutions, or Indian cities is unknown.

## Download

Run once from the repository root:

```powershell
.\.venv-py314\Scripts\python.exe scripts\download_model.py
```

The script verifies size and SHA-256 before installing the file. The checkpoint is ignored by Git and must not be committed. Model loading uses `torch.load(..., weights_only=True)` and strict state-dict matching.
