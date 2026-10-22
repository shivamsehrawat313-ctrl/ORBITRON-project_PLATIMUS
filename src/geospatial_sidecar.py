"""Attach validated LEVIR per-sample bounds to pixel-only PNG metadata."""
import json
import math
from dataclasses import replace
from pathlib import Path


def _coordinates_by_name(document):
    if isinstance(document,dict) and document.get("type")=="FeatureCollection":
        entries={}
        for feature in document.get("features",[]):
            name=feature.get("properties",{}).get("name")
            geometry=feature.get("geometry") or {}
            if not name or geometry.get("type")!="Polygon": continue
            rings=geometry.get("coordinates") or []
            if not rings or not rings[0]: continue
            points=rings[0]
            west=min(p[0] for p in points); east=max(p[0] for p in points)
            south=min(p[1] for p in points); north=max(p[1] for p in points)
            corners={(west,south),(west,north),(east,south),(east,north)}
            if len(points)!=5 or tuple(points[0])!=tuple(points[-1]) or any(tuple(p) not in corners for p in points[:-1]):
                raise ValueError(f"GeoJSON coordinates for {name} must describe a closed axis-aligned rectangular image extent.")
            entries[str(name).casefold()]=(west,north,east,south)
        return entries
    if isinstance(document,dict):
        return {str(key).casefold():value for key,value in document.items()}
    raise ValueError("Coordinate sidecar must be a LEVIR JSON mapping or GeoJSON FeatureCollection.")


def attach_levir_coordinates(info, raw, sidecar_name):
    """Return copied ImageInfo with an axis-aligned WGS84 transform from sidecar bounds.

    LEVIR sidecars map a sample filename to [west, north, east, south]. They describe
    the image footprint, not a CRS embedded in the PNG, so the app records the source.
    """
    suffix=Path(sidecar_name).suffix.lower()
    if suffix not in {".json",".geojson"}:
        raise ValueError("Upload the LEVIR coordinate sidecar as .json or .geojson.")
    try: document=json.loads(raw.decode("utf-8-sig"))
    except Exception as exc: raise ValueError(f"Coordinate sidecar could not be decoded: {exc}") from exc
    entries=_coordinates_by_name(document)
    key=Path(info.filename).name.casefold()
    if key not in entries:
        raise ValueError(f"No coordinate entry for {info.filename}. The sidecar must match the exact sample filename.")
    try:
        west,north,east,south=(float(value) for value in entries[key])
    except (TypeError,ValueError) as exc:
        raise ValueError(f"Coordinate bounds for {info.filename} must contain [west, north, east, south].") from exc
    if not all(math.isfinite(value) for value in (west,north,east,south)):
        raise ValueError("Coordinate bounds must be finite numbers.")
    if not (-180<=west<east<=180 and -90<=south<north<=90):
        raise ValueError("Invalid WGS84 bounds: expected west < east and south < north within longitude/latitude limits.")
    if info.width<=0 or info.height<=0: raise ValueError("Image dimensions must be positive to apply sample bounds.")
    from affine import Affine
    transform=Affine((east-west)/info.width,0,west,0,-(north-south)/info.height,north)
    warnings=[w for w in (info.warnings or []) if "no geospatial metadata" not in w.casefold()]
    warnings.append("WGS84 sample bounds were attached from the matching LEVIR sidecar; pixel-to-coordinate mapping assumes a north-up rectangular image grid.")
    return replace(info,crs="EPSG:4326",bounds=(west,south,east,north),
        pixel_size=(transform.a,transform.e),transform=tuple(transform),georeferenced=True,
        georeference_source=f"{Path(sidecar_name).name} entry {Path(info.filename).name}",warnings=warnings)
