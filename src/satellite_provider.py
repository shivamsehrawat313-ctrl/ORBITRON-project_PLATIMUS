"""Small Sentinel-2 L2A area/date retrieval through the public Earth Search STAC API."""
from __future__ import annotations

from datetime import date, timedelta
from io import BytesIO
import math

STAC_URL = "https://earth-search.aws.element84.com/v1/search"


def _square_aoi(longitude: float, latitude: float, radius_km: float):
    from pyproj import CRS, Transformer
    from shapely.geometry import box, mapping
    if not (-180 <= longitude <= 180 and -80 <= latitude <= 84):
        raise ValueError("Choose a location between 80°S and 84°N (the supported Sentinel-2 UTM latitude range).")
    zone = max(1, min(60, int((longitude + 180) // 6) + 1))
    epsg = (32600 if latitude >= 0 else 32700) + zone
    local = CRS.from_epsg(epsg)
    forward = Transformer.from_crs("EPSG:4326", local, always_xy=True).transform
    reverse = Transformer.from_crs(local, "EPSG:4326", always_xy=True).transform
    x, y = forward(longitude, latitude)
    radius = radius_km * 1000
    projected = box(x-radius, y-radius, x+radius, y+radius)
    from shapely.ops import transform
    geographic = transform(reverse, projected)
    return mapping(geographic), projected.bounds, local


def _search_scene(aoi, chosen_date: date, window_days: int, max_cloud: float, collection: str):
    import requests
    start = chosen_date - timedelta(days=window_days)
    end = chosen_date + timedelta(days=window_days)
    body = {
        "collections": [collection], "intersects": aoi,
        "datetime": f"{start.isoformat()}T00:00:00Z/{end.isoformat()}T23:59:59Z",
        "limit": 100,
    }
    if collection == "sentinel-2-c1-l2a":
        body["query"] = {"eo:cloud_cover": {"lte": float(max_cloud)}}
    try:
        response = requests.post(STAC_URL, json=body, timeout=(10, 60))
        response.raise_for_status()
        items = response.json().get("features", [])
    except Exception as exc:
        raise RuntimeError(f"Sentinel-2 catalog search failed: {exc}") from exc
    if not items:
        raise ValueError(f"No Sentinel-2 scene found from {start} to {end} under {max_cloud:g}% cloud cover. Widen the date window or cloud limit.")
    def rank(item):
        acquired = item.get("properties", {}).get("datetime", "")[:10]
        try: day_distance = abs((date.fromisoformat(acquired) - chosen_date).days)
        except ValueError: day_distance = 99999
        cloud = float(item.get("properties", {}).get("eo:cloud_cover") or 100)
        return day_distance, cloud
    items.sort(key=rank)
    return items[0]


def _read_rgb_scene(item, aoi, grid=None):
    import numpy as np
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.transform import from_bounds
    from rasterio.warp import transform_geom, reproject
    from shapely.geometry import shape

    assets = item.get("assets", {})
    collection = item.get("collection")
    if collection == "naip":
        if not assets.get("image", {}).get("href"):
            raise ValueError("The selected NAIP catalog item has no image asset.")
        sources = [assets["image"]] * 3
        indexes = [1, 2, 3]
    else:
        keys = ("red", "green", "blue")
        if any(key not in assets or not assets[key].get("href") for key in keys):
            raise ValueError("The selected Sentinel-2 catalog item is missing one or more RGB assets.")
        sources = [assets[key] for key in keys]
        indexes = [1, 1, 1]
    # Derive one native-resolution UTM grid from the first observation, then
    # place the later observation on that exact grid.
    first = rasterio.open(sources[0]["href"])
    if first.crs is None:
        first.close(); raise ValueError("The Sentinel-2 red-band asset has no CRS metadata.")
    if grid is None:
        target_crs = first.crs
        projected_aoi = shape(transform_geom("EPSG:4326", target_crs, aoi))
        bounds = projected_aoi.bounds
        resolution = max(abs(first.transform.a), abs(first.transform.e))
        width = max(1, math.ceil((bounds[2]-bounds[0]) / resolution))
        height = max(1, math.ceil((bounds[3]-bounds[1]) / resolution))
        transform = from_bounds(*bounds, width, height)
        grid = (target_crs, transform, width, height, resolution)
    else:
        target_crs, transform, width, height, resolution = grid
    if width*height > 4_000_000:
        first.close(); raise ValueError("Selected area is too large. Reduce the area radius to 10 km or less.")
    first.close()
    rgb = np.zeros((height, width, 3), dtype=np.uint8)
    valid = np.ones((height, width), dtype=bool)
    for channel, (asset, band_index) in enumerate(zip(sources, indexes)):
        with rasterio.open(asset["href"]) as src:
            band = np.zeros((height, width), dtype=np.float32)
            reproject(source=rasterio.band(src, band_index), destination=band,
                src_transform=src.transform, src_crs=src.crs,
                dst_transform=transform, dst_crs=target_crs,
                resampling=Resampling.bilinear, dst_nodata=0)
            # Sentinel-2 surface-reflectance values are scaled integers. A fixed
            # scale keeps the two dates comparable and avoids scene-wise stretching.
            if src.dtypes[band_index-1] == "uint8":
                rgb[:, :, channel] = np.clip(band, 0, 255).astype(np.uint8)
            else:
                rgb[:, :, channel] = np.clip(band * (255.0 / 3000.0), 0, 255).astype(np.uint8)
            source_mask = np.zeros((height, width), dtype=np.uint8)
            reproject(source=src.dataset_mask(), destination=source_mask,
                src_transform=src.transform, src_crs=src.crs,
                dst_transform=transform, dst_crs=target_crs,
                resampling=Resampling.nearest, dst_nodata=0)
            valid &= source_mask > 0
    if float(valid.mean()) < 0.95:
        raise ValueError("The selected catalog scene does not cover at least 95% of the requested AOI. Choose a smaller area or another date.")
    rgb[~valid] = 0
    profile = {"driver":"GTiff", "height":height, "width":width, "count":3,
        "dtype":"uint8", "crs":target_crs, "transform":transform,
        "compress":"deflate", "nodata":0}
    output = BytesIO()
    with rasterio.MemoryFile() as mem:
        with mem.open(**profile) as ds:
            ds.write(rgb.transpose(2, 0, 1))
            ds.write_mask(valid.astype(np.uint8)*255)
        output.write(mem.read())
    props = item.get("properties", {})
    metadata = {"item_id":item.get("id"), "acquired":props.get("datetime"),
        "cloud_cover":float(props.get("eo:cloud_cover") or 100), "collection":item.get("collection"),
        "resolution_m":resolution, "crs":target_crs.to_string(),
        "width":width, "height":height}
    return output.getvalue(), metadata, grid


def retrieve_satellite_pair(longitude: float, latitude: float, radius_km: float,
                           date_before: date, date_after: date,
                           window_days: int = 30, max_cloud: float = 20.0,
                           provider: str = "Sentinel-2"):
    """Fetch two real NAIP or Sentinel-2 RGB chips on a shared projected grid."""
    if date_before >= date_after:
        raise ValueError("Choose the earlier date before the later date.")
    if not 0.25 <= radius_km <= 10:
        raise ValueError("Area radius must be between 0.25 km and 10 km.")
    if not 0 <= window_days <= 90:
        if provider != "NAIP" or not 0 <= window_days <= 1095:
            raise ValueError("Date search window is outside the supported range.")
    collection = "naip" if provider == "NAIP" else "sentinel-2-c1-l2a"
    if provider not in {"NAIP", "Sentinel-2"}:
        raise ValueError("Choose NAIP or Sentinel-2 imagery.")
    if provider == "NAIP" and not (-125 <= longitude <= -66 and 24 <= latitude <= 50):
        raise ValueError("NAIP coverage is limited to the contiguous United States; choose Sentinel-2 for this location.")
    aoi, _, _ = _square_aoi(longitude, latitude, radius_km)
    item_a = _search_scene(aoi, date_before, window_days, max_cloud, collection)
    item_b = _search_scene(aoi, date_after, window_days, max_cloud, collection)
    acquired_a = item_a.get("properties", {}).get("datetime", "")[:10]
    acquired_b = item_b.get("properties", {}).get("datetime", "")[:10]
    if not acquired_a or not acquired_b or acquired_a >= acquired_b:
        raise ValueError("The catalog returned observations that are not chronologically ordered. Adjust the two target dates or reduce the date tolerance.")
    rgb_a, meta_a, common_grid = _read_rgb_scene(item_a, aoi)
    rgb_b, meta_b, _ = _read_rgb_scene(item_b, aoi, common_grid)
    if (meta_a["width"], meta_a["height"], meta_a["crs"]) != (meta_b["width"], meta_b["height"], meta_b["crs"]):
        raise ValueError("Selected observations could not be placed on matching grids. Try a smaller AOI or dates in the same Sentinel tile.")
    return rgb_a, rgb_b, meta_a, meta_b
