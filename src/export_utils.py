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

def mask_png_bytes(mask):
    """Encode a binary prediction as a downloadable grayscale PNG."""
    from io import BytesIO
    import numpy as np
    from PIL import Image
    raster=(np.asarray(mask,dtype=bool)*255).astype("uint8")
    output=BytesIO(); Image.fromarray(raster,mode="L").save(output,format="PNG")
    return output.getvalue()

def prediction_polygons_json_bytes(features_before,features_after,change_features):
    """Export extracted polygons in image pixel coordinates, never as fake GeoJSON."""
    from shapely.geometry import mapping
    result={"coordinate_space":"image_pixel_coordinates","crs":None,"features":[]}
    for epoch,features in (("before",features_before),("after",features_after),("changed_region",change_features)):
        for index,feature in enumerate(features,1):
            geometry=feature.get("geometry")
            if geometry is None or geometry.is_empty: continue
            result["features"].append({
                "type":"Feature","geometry":mapping(geometry),
                "properties":{"kind":epoch,"id":f"{epoch}_{index}",
                              "confidence":feature.get("confidence")},
            })
    return json.dumps(result,allow_nan=False).encode("utf-8")
