# Urban Sprawl Intelligence

Streamlit prototype for comparing building footprints detected in two satellite/aerial observations. It reports image evidence for human review. A newly detected footprint is not a claim about construction date, legality, permits, or land use.

## What works now

- Upload and preview GeoTIFF, PNG, or JPEG observations with user-supplied dates and metadata checks.
- Run a real pretrained building segmenter on both inputs. It uses overlapping 512-pixel tiles and a configurable probability threshold.
- Use a separately loaded LEVIR-trained BIT model for the direct two-date change mask, overlays, change percentage, and optional label evaluation.
- Extract footprint polygons with per-polygon mean model confidence; compare footprints with overlap/distance matching; flag additions, removals, modifications, and uncertain matches.
- Block incompatible CRS, extent, pixel size, grid transform, and image dimensions. A phase-correlation diagnostic estimates image-content translation; large reliable offsets block analysis, while low-confidence registration marks results uncertain. No automatic correction is applied. Non-georeferenced inputs require explicit same-area/alignment confirmation and remain pixel-space.
- View before/after mask overlays, the change table, and an interactive Folium map for georeferenced outputs.
- Edit a separate human review field and export CSV, analysis JSON, and WGS84 GeoJSON.

Uploaded images are decoded and analyzed locally. The interactive map requests OpenStreetMap basemap tiles in the browser; it does not send uploaded imagery or inference results to that tile service.

## Model and scope

### LEVIR-specific change model

For LEVIR-CD label metrics and the changed-pixel mask, the app uses a separate BIT bi-temporal change detector trained on LEVIR-CD, from the [Hugging Face checkpoint](https://huggingface.co/LEEhyeongsuk/BIT-LEVIR-CD) based on the [official BIT implementation](https://github.com/justchenhao/BIT_CD). The checkpoint is **non-commercial use only**. Each image is split into overlapping native-resolution 256x256 crops; a 1024x1024 scene is never reduced to one 256x256 image. RGB values are normalized to [-1,1], and two-class logits (unchanged/changed) are converted to probabilities with softmax. The model file is downloaded and SHA-256 verified with `scripts/download_levir_change_model.py`.

The existing SpaceNet-2 Paris DeepLab model remains responsible for per-date building outlines and counts. It is out of domain for LEVIR, so its outlines and counts may be inaccurate on LEVIR images. BIT directly predicts temporal change; change-mask metrics are no longer computed from the XOR of those SpaceNet building masks. The reported benchmark score is not a promise of performance on your pair.

The configured checkpoint is a task-trained one-channel DeepLabV3-ResNet50 model available from the [author's source repository](https://github.com/AryakBhattacharya/building-footprint-extraction) and [Hugging Face model repository](https://huggingface.co/Abhatta7/building-footprint-segmentation). Those repositories declare MIT licensing. The training data is SpaceNet 2 Paris RGB imagery. The model card calls the architecture “DeepLabV3+,” while its published notebook constructs `torchvision.models.segmentation.deeplabv3_resnet50`; the checkpoint's state dict strictly loads into that Torchvision DeepLabV3 architecture. This implementation follows the executable source/checkpoint structure.

The public source describes 512-pixel patches and scene-wide min/max normalization. This app reads the first three raster bands as RGB, scales each full scene to float32 `[0,1]`, runs overlapping tiles, averages logits, then applies sigmoid once and thresholds probabilities. Checkpoint revision, byte size, SHA-256, output contract, and download command are in [models/README.md](models/README.md). The 168 MB weights are downloaded separately and ignored by Git.

The upstream model card reports scores, but it does not give enough detail here to reproduce its exact validation split. Those numbers are not used as this application's evaluation. The model is suited to SpaceNet-2-like high-resolution RGB imagery. Its performance on other sensors, seasons, image resolutions, and Indian cities is unvalidated. Semantic masks can merge adjacent roofs into one component.

## Environment and installation

Detected interpreter: Python 3.14.7 on Windows. The project dependencies were installed and imports verified in `.venv-py314`. The pre-existing `.venv` refers to a removed Python 3.11 installation and was access-locked by OneDrive, so it was left untouched. Python 3.14 CPU PyTorch, Rasterio, GeoPandas, Shapely, Streamlit, Pytest, and Torchvision wheels installed successfully.

PowerShell, from the repository root:

```powershell
python -m venv .venv-py314
.\.venv-py314\Scripts\python.exe -m pip install --upgrade pip
.\.venv-py314\Scripts\python.exe -m pip install -r requirements.txt
.\.venv-py314\Scripts\python.exe scripts\download_model.py
.\.venv-py314\Scripts\python.exe -m streamlit run app.py
```

Run tests with:

```powershell
.\.venv-py314\Scripts\python.exe -m pytest -q
```

Linux/macOS users can create `.venv`, install `requirements.txt`, run `python scripts/download_model.py`, then `python -m streamlit run app.py`. Choose a Torch build appropriate to the machine using the [official PyTorch installer](https://pytorch.org/get-started/locally/) if CUDA is desired. The included tested installation is CPU-only.

## Imagery requirements

Use imagery with visible RGB bands and the same ground coverage and pixel grid. GeoTIFF is recommended. At least three bands are required; for multiband imagery the application uses bands 1–3, so ensure these are RGB in that order. Inputs must be below the 180 MB app limit. Images with different dimensions, CRS, extents, resolution, or affine transforms are blocked; reproject/register them to a common grid yourself before uploading. No automatic correction is attempted, so small unreported registration errors remain a limitation. Clouds, shadows, season, illumination, and off-nadir angle can change predictions.

No sample image is bundled. The official [SpaceNet 2 dataset page](https://spacenet.ai/spacenet-buildings-dataset-v2/) provides licensed dataset details and AWS instructions; imagery from this dataset is CC BY-SA 4.0. The [SpaceNet 7 page](https://www.spacenet.ai/sn7-challenge/) describes a multi-temporal dataset, but its ~4 m imagery differs substantially from this model's high-resolution training domain. Check current terms for any dataset and imagery before redistribution. Bhuvan/NRSC products have product-specific access/use terms; do not assume open redistribution.

## Testing and evaluation

Tests cover alignment checks, CRS-aware GeoJSON reprojection, pixel-space restrictions, polygon extraction/confidence, additions/removals/modified/unchanged/uncertain matching, invalid geometry, duplicate prevention, and exports. With the ignored checkpoint present, three CPU integration smoke tests load the real checkpoint and exercise inference plus PNG/GeoTIFF pipelines. They use deterministic synthetic RGB arrays to test code paths; they are **not** satellite-image accuracy tests. Latest local run: `19 passed` (Python 3.14.7, .venv-py314).

No labeled satellite test image/reference data is currently bundled, so this repository reports no building-detection or change-detection accuracy metrics. The `src/metrics.py` helper computes binary precision, recall, F1, IoU, and false-positive rate only when supplied with aligned prediction/reference arrays. A credible local evaluation should identify the imagery, sensor, resolution, date, geographic split, and reference labels, and evaluate segmentation separately from temporal matching.

### Try a labeled LEVIR-CD pair

Download the LEVIR checkpoint as well as the footprint checkpoint before launching:

```powershell
.\.venv-py314\Scripts\python.exe scripts\download_levir_change_model.py
```

The app accepts the LEVIR-CD before image, after image, and optional binary change-label PNG. Download a split archive from the [LEVIR-CD page](https://justchenhao.github.io/LEVIR/) and extract one matching sample from each of its `A`, `B`, and `label` folders (the same filename/sample ID in all three). In the app, upload A as the earlier image, B as the later image, and the corresponding label using the separate reference-label uploader. The label is excluded from model input and is used only after inference to calculate pixel-level precision, recall, F1, and IoU. If dimensions do not match, the app blocks evaluation.

LEVIR-CD imagery is PNG and has no embedded CRS or pixel transform. To attach coordinates, also upload the LEVIR `LEVIR_CD_name_coords.json` or `.geojson` sidecar. The app requires an exact sample-filename match for both images and validates finite WGS84 bounds before enabling maps/GeoJSON. It derives a north-up affine grid from the sidecar's rectangular extent and image dimensions; this assumption is shown in the UI, so treat map placement and area estimates as approximate. Without a matched sidecar the inputs remain pixel-space. The dataset is academic-use only. Its labels identify changed pixels/buildings, not a complete building inventory for each date.

The reported changed-region share is `BIT predicted changed pixels / valid comparison pixels × 100`. It is a changed-pixel share, not building-count growth. Building counts are connected mask components from the separate SpaceNet model and may merge touching roofs or split one roof into several detections. Ground-truth precision/recall/F1/IoU score the BIT pixel change mask; they do not establish building-count accuracy. The published BIT score is not a guarantee for an individual pair; evaluate representative held-out labels for the intended imagery.

### Retrieve an image pair by location and dates (Option B)

Select **Find satellite images by location and dates**, click the map to set the area center, choose the two target dates, set a date tolerance and click **Search and retrieve image pair**. The app uses the public [Earth Search STAC catalog](https://earth-search.aws.element84.com/v1/) and reads RGB cloud-optimized GeoTIFF assets directly; no AWS access key is required. Choose Sentinel-2 for global coverage and set a cloud-cover limit. For locations in the contiguous U.S., choose NAIP for higher-resolution aerial imagery. NAIP is not satellite imagery; it is typically acquired at least once every three years. Earth Search is a best-effort public service, so catalog/network availability is not guaranteed. Sentinel data are free to users under the [Copernicus data policy](https://dataspace.copernicus.eu/data-policy).

Retrieved observations are cut to the selected square AOI and resampled onto a shared projected grid. Dates shown after retrieval are actual acquisition dates, which can differ from the requested target dates. Increase the date tolerance if the catalog returns no scene; Sentinel-2 also lets you increase the cloud-cover limit. Its cloud percentage describes the source scene, not a precise cloud mask for the selected chip. Sentinel-2 RGB is 10 m per pixel, suitable for broad land-cover changes but too coarse for many individual houses. NAIP is sub-meter aerial imagery and has better visual detail for buildings, but the existing SpaceNet/LEVIR models were not validated on NAIP either. Do not claim building-count accuracy from either source without matching labels and evaluation. The retrieved GeoTIFFs contain CRS and affine transform metadata and feed the normal pipeline.

## Troubleshooting and limits

- Model unavailable: run the weight download command and check the recorded file size/SHA-256 in `models/README.md`.
- Model load error: the checkpoint must be the documented state dict; the loader uses safe `weights_only=True` and strict key matching.
- Mismatched georeferencing or detected image shift: register/reproject both images to a common CRS, extent, resolution, and pixel grid externally, then retry.
- Pixel-only outputs cannot be exported as GeoJSON or interpreted as square metres.
- A low threshold can create false positives; a high threshold can miss buildings. Calibrate against local labeled imagery.
- Human review is separate from model category. The app does not verify permits, zoning, ownership, or legal authorization.
- Fine-tuning, registration correction, cloud masking, and independent satellite benchmark evaluation are not included.

## Current verification status

Python 3.14.7 project environment and dependencies have been installed (`pip check` is clean). The real checkpoint loads successfully on CPU. Inference was smoke-tested on deterministic synthetic RGB inputs, and change pipelines were smoke-tested on synthetic non-georeferenced PNG and georeferenced GeoTIFF inputs. Streamlit `AppTest` reported zero exceptions; the server health endpoint returned `200 ok`. Inference against an actual satellite sample was not run because no sample image was available locally. Latest pytest run: 15 passed.
#   O R B I T R O N - p r o j e c t _ P L A T I M U S  
 #   O R B I T R O N - p r o j e c t _ P L A T I M U S  
 
