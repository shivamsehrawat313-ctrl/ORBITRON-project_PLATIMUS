"""Extract footprint polygons and mean segmentation confidence from masks."""
def mask_to_features(mask, probabilities=None, transform=None, min_pixels=3):
    try:
        import numpy as np
        from affine import Affine
        from rasterio.features import shapes, rasterize
        from shapely.geometry import shape
        from shapely.affinity import affine_transform
    except ImportError as e: raise RuntimeError("Polygon extraction requires numpy, rasterio, affine, and shapely") from e
    arr=np.asarray(mask,dtype="uint8")
    scores=np.asarray(probabilities,dtype="float32") if probabilities is not None else None
    if arr.ndim!=2 or (scores is not None and scores.shape!=arr.shape):
        raise ValueError("Mask and probability arrays must be matching 2D rasters")
    result=[]
    for geom,val in shapes(arr,mask=arr.astype(bool)):
        if not val: continue
        pixel_geom=shape(geom)
        if pixel_geom.area < min_pixels: continue
        conf=None
        if scores is not None:
            x0=max(0,int(pixel_geom.bounds[0])); y0=max(0,int(pixel_geom.bounds[1]))
            x1=min(arr.shape[1],int(pixel_geom.bounds[2]+1)); y1=min(arr.shape[0],int(pixel_geom.bounds[3]+1))
            local_geom=affine_transform(pixel_geom,[1,0,0,1,-x0,-y0])
            local=rasterize([(local_geom,1)],out_shape=(y1-y0,x1-x0),transform=Affine.identity(),fill=0,dtype="uint8")
            vals=scores[y0:y1,x0:x1][local.astype(bool)]
            if vals.size: conf=float(vals.mean())
        map_geom=pixel_geom
        if transform is not None:
            map_geom=affine_transform(pixel_geom,[transform.a,transform.b,transform.d,transform.e,transform.c,transform.f])
        if not map_geom.is_valid: map_geom=map_geom.buffer(0)
        if not map_geom.is_empty: result.append({"geometry":map_geom,"confidence":conf,"area_pixels":float(pixel_geom.area)})
    return result

def mask_to_polygons(mask, transform=None, min_pixels=3):
    return [f["geometry"] for f in mask_to_features(mask,transform=transform,min_pixels=min_pixels)]
