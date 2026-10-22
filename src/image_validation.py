"""Input validation helpers with optional rasterio support."""
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

SUPPORTED = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}

@dataclass
class ImageInfo:
    filename: str
    width: int
    height: int
    bands: int
    dtype: str
    crs: str | None = None
    bounds: tuple | None = None
    pixel_size: tuple | None = None
    transform: tuple | None = None
    georeferenced: bool = False
    georeference_source: str | None = None
    warnings: list[str] | None = None
    def to_dict(self): return asdict(self)

def inspect_image(source: Any, filename: str | None = None) -> ImageInfo:
    """Inspect path or seekable uploaded file. Never infer location from pixels."""
    name = filename or getattr(source, "name", None) or str(source)
    suffix = Path(name).suffix.lower()
    if suffix not in SUPPORTED:
        raise ValueError(f"Unsupported image format '{suffix or '(none)'}'. Use GeoTIFF, PNG, or JPEG.")
    warnings = []
    if suffix in {".tif", ".tiff"}:
        try:
            import rasterio
        except ImportError as e:
            raise RuntimeError("Reading GeoTIFF requires rasterio. Install project dependencies.") from e
        with rasterio.open(source) as ds:
            crs = ds.crs.to_string() if ds.crs else None
            geo = bool(ds.crs and ds.transform and ds.transform != rasterio.Affine.identity())
            if not geo: warnings.append("No usable CRS/affine transform; geographic measurements unavailable.")
            return ImageInfo(Path(name).name, ds.width, ds.height, ds.count,
                ",".join(ds.dtypes), crs, tuple(ds.bounds), tuple(ds.res),
                transform=tuple(ds.transform),georeferenced=geo,
                georeference_source="embedded GeoTIFF metadata" if geo else None,warnings=warnings)
    try:
        from PIL import Image
        with Image.open(source) as im:
            im.verify()
        source.seek(0)
        with Image.open(source) as im:
            bands = len(im.getbands()); w, h = im.size; dtype = im.mode
    except ImportError as e:
        raise RuntimeError("Reading PNG/JPEG requires Pillow. Install project dependencies.") from e
    except Exception as e:
        raise ValueError(f"Image could not be decoded: {e}") from e
    warnings.append("Image has no geospatial metadata; results are pixel-space only.")
    return ImageInfo(Path(name).name, w, h, bands, dtype, warnings=warnings)

def compare_metadata(a: ImageInfo, b: ImageInfo) -> list[str]:
    out=[]
    if (a.width,a.height)!=(b.width,b.height): out.append("Image dimensions differ; align/resample explicitly before comparison.")
    if a.georeferenced != b.georeferenced: out.append("Only one image is georeferenced; geographic comparison is disabled.")
    if a.georeferenced and b.georeferenced:
        if a.crs != b.crs: out.append("CRS differs; reproject to a shared CRS before comparison.")
        if a.bounds != b.bounds: out.append("Spatial extents differ; compare only after confirming overlapping coverage.")
        if a.pixel_size != b.pixel_size: out.append("Ground sampling differs; area change may reflect resolution differences.")
        if a.transform != b.transform: out.append("Pixel grid transforms differ; images are not registered to the same pixel grid.")
    return out
