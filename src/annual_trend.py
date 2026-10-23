"""Google Open Buildings Temporal annual metrics through Earth Engine."""
from __future__ import annotations

from datetime import datetime, timezone


COLLECTION_ID = "GOOGLE/Research/open-buildings-temporal/v1"
FIRST_YEAR = 2016
LAST_YEAR = 2023


def _sum_band(ee, image, band, region, scale):
    values = image.select(band).reduceRegion(
        reducer=ee.Reducer.sum(),
        geometry=region,
        scale=scale,
        maxPixels=100_000_000,
        tileScale=4,
    ).getInfo()
    return values.get(band) if values else None


def _year_images(ee, collection, region, years):
    timestamps = (collection.filterBounds(region)
                  .aggregate_array("system:time_start").distinct().sort().getInfo())
    by_year = {}
    for millis in timestamps or []:
        year = datetime.fromtimestamp(millis / 1000, tz=timezone.utc).year
        if year in years:
            by_year[year] = collection.filter(ee.Filter.eq("system:time_start", millis)).mosaic().clip(region)
    missing = [year for year in years if year not in by_year]
    if missing:
        raise ValueError(
            "No Open Buildings Temporal tiles were found at this location for "
            + ", ".join(map(str, missing))
            + ". Try another nearby point or a shorter year range."
        )
    return by_year


def retrieve_annual_trend(project_id, latitude, longitude, radius_km, start_year, end_year, confidence_threshold):
    """Return annual AOI metrics and real Earth Engine visualization tile URLs.

    The count band is delivered at 0.5 m raster spacing. Summing the sampled
    fractional-count values at 1 m and multiplying by four accounts for the
    four 0.5 m cells per 1 m cell; outputs remain approximate model estimates.
    """
    try:
        import ee
    except ImportError as exc:
        raise RuntimeError("Earth Engine API is not installed. Install requirements.txt first.") from exc

    if not project_id.strip():
        raise ValueError("Enter a Google Cloud project ID registered for Earth Engine.")
    try:
        ee.Initialize(project=project_id.strip())
    except Exception as exc:
        raise RuntimeError(
            "Earth Engine could not initialize. In this same Windows account, run "
            "`.venv-py314\\Scripts\\earthengine.exe authenticate`, sign in, and make sure "
            "Earth Engine is registered/enabled for the Google Cloud project ID you entered. "
            f"Details: {exc}"
        ) from exc

    point = ee.Geometry.Point([float(longitude), float(latitude)])
    region = point.buffer(float(radius_km) * 1000)
    collection = ee.ImageCollection(COLLECTION_ID)
    years = list(range(int(start_year), int(end_year) + 1))
    images = _year_images(ee, collection, region, years)

    yearly = []
    presences = {}
    for year in years:
        image = images[year]
        fractional_count = _sum_band(ee, image, "building_fractional_count", region, 1)
        presence = image.select("building_presence")
        presences[year] = presence
        built_area = _sum_band(
            ee,
            presence.gte(float(confidence_threshold)).multiply(ee.Image.pixelArea()).rename("area_m2"),
            "area_m2",
            region,
            4,
        )
        if fractional_count is None or built_area is None:
            raise ValueError(f"The dataset returned no valid pixels for {year} at this location.")
        yearly.append({
            "year": year,
            "estimated_buildings": float(fractional_count) * 4.0,
            "built_area_m2": float(built_area),
        })

    before = presences[int(start_year)]
    after = presences[int(end_year)]
    for index, row in enumerate(yearly):
        previous = yearly[index - 1] if index else None
        row["building_count_change"] = row["estimated_buildings"] - previous["estimated_buildings"] if previous else None
        row["building_count_change_percent"] = (
            100 * row["building_count_change"] / previous["estimated_buildings"]
            if previous and previous["estimated_buildings"] else None
        )
        row["built_area_change_m2"] = row["built_area_m2"] - previous["built_area_m2"] if previous else None
        row["built_area_change_percent"] = (
            100 * row["built_area_change_m2"] / previous["built_area_m2"]
            if previous and previous["built_area_m2"] else None
        )
    valid = before.mask().And(after.mask())
    old_built = before.gte(float(confidence_threshold)).And(valid)
    new_built = after.gte(float(confidence_threshold)).And(valid)
    pixel_area = ee.Image.pixelArea()

    def area(mask):
        return _sum_band(ee, mask.multiply(pixel_area).rename("area_m2"), "area_m2", region, 4) or 0.0

    valid_area = area(valid)
    added_area = area(new_built.And(old_built.Not()))
    removed_area = area(old_built.And(new_built.Not()))
    changed_area = added_area + removed_area
    start_count = yearly[0]["estimated_buildings"]
    end_count = yearly[-1]["estimated_buildings"]

    # These are model-derived raster overlays, not source satellite photographs.
    old_vis = before.getMapId({"min": 0.2, "max": 1.0, "palette": ["ffffcc", "78c679", "006837"]})
    new_vis = after.getMapId({"min": 0.2, "max": 1.0, "palette": ["ffffcc", "78c679", "006837"]})
    changed_image = new_built.And(old_built.Not()).Or(old_built.And(new_built.Not())).selfMask()
    change_vis = changed_image.getMapId({"min": 0, "max": 1, "palette": ["ff3333"]})
    return {
        "yearly": yearly,
        "start_year": int(start_year),
        "end_year": int(end_year),
        "start_count": start_count,
        "end_count": end_count,
        "net_count_change": end_count - start_count,
        "added_built_area_m2": added_area,
        "removed_built_area_m2": removed_area,
        "changed_area_m2": changed_area,
        "valid_area_m2": valid_area,
        "changed_area_percent": (100 * changed_area / valid_area) if valid_area else 0.0,
        "built_area_change_percent": (100 * (yearly[-1]["built_area_m2"] - yearly[0]["built_area_m2"]) / yearly[0]["built_area_m2"])
        if yearly[0]["built_area_m2"] else None,
        "annualized_built_area_growth_percent": (
            100 * ((yearly[-1]["built_area_m2"] / yearly[0]["built_area_m2"]) ** (1 / (int(end_year) - int(start_year))) - 1)
            if yearly[0]["built_area_m2"] > 0 and int(end_year) > int(start_year) else None
        ),
        "tiles": {
            "start": old_vis["tile_fetcher"].url_format,
            "end": new_vis["tile_fetcher"].url_format,
            "changed": change_vis["tile_fetcher"].url_format,
        },
    }
