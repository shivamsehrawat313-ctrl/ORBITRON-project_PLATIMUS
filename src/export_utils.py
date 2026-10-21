"""Machine-readable analysis exports; GeoJSON only for genuine map coordinates."""
import csv, json
from io import StringIO

def csv_bytes(rows):
    if not rows: return b""
    f=StringIO(newline=""); writer=csv.DictWriter(f,fieldnames=list(rows[0].keys()),extrasaction="ignore"); writer.writeheader()
    for r in rows: writer.writerow({k:json.dumps(v) if isinstance(v,(list,dict)) else v for k,v in r.items()})
    return f.getvalue().encode("utf-8")

def geojson_bytes(features, crs=None, georeferenced=False):
    if not georeferenced: raise ValueError("GeoJSON export disabled: source coordinates are not georeferenced.")
    if not crs: raise ValueError("GeoJSON export requires the source CRS so coordinates can be transformed to EPSG:4326.")
    try:
        from shapely.geometry import mapping
        from shapely import transform
        import numpy as np
    except ImportError as e: raise RuntimeError("GeoJSON export requires shapely") from e
    collection={"type":"FeatureCollection","features":[]}
    try:
        from pyproj import CRS,Transformer
        source_crs=CRS.from_user_input(crs)
        projector=Transformer.from_crs(source_crs,"EPSG:4326",always_xy=True).transform
    except ImportError as e: raise RuntimeError("CRS transformation for GeoJSON requires pyproj") from e
    for props,geom in features:
        if geom is None or geom.is_empty: continue
        geom=transform(geom,lambda xy: np.column_stack(projector(xy[:,0],xy[:,1])))
        if not geom.is_valid: raise ValueError("Invalid geometry cannot be exported to GeoJSON")
        collection["features"].append({"type":"Feature","geometry":mapping(geom),"properties":props})
    return json.dumps(collection,allow_nan=False).encode("utf-8")

def summary_json_bytes(summary): return json.dumps(summary,indent=2,default=str,allow_nan=False).encode("utf-8")
