"""Image loading helpers; large GeoTIFFs can be accessed window by window."""
def open_raster(source):
    try: import rasterio
    except ImportError as e: raise RuntimeError("GeoTIFF loading requires rasterio") from e
    return rasterio.open(source)
