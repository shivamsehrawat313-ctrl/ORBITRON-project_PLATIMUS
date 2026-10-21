"""CRS-safe area utilities."""
def area_in_square_metres(geometry, crs):
    if crs is None: raise ValueError("Area in m² requires a known projected CRS")
    try:
        from pyproj import CRS
        parsed=CRS.from_user_input(crs)
    except ImportError as e: raise RuntimeError("Area calculation requires pyproj") from e
    if not parsed.is_projected: raise ValueError("Area in m² requires projected coordinates")
    factor=parsed.axis_info[0].unit_conversion_factor
    return geometry.area * factor * factor
