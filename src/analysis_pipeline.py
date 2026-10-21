"""End-to-end local image inference, footprint extraction, and temporal matching."""
from datetime import date
from io import BytesIO

def _read_rgb(upload, info):
    import numpy as np
    suffix=upload.name.lower().rsplit(".",1)[-1]
    raw=upload.getvalue()
    if suffix in {"tif","tiff"}:
        import rasterio
        from rasterio.io import MemoryFile
        with MemoryFile(raw) as mem:
            with mem.open() as ds:
                if ds.count < 3: raise ValueError(f"{info.filename}: model requires at least 3 RGB bands")
                rgb=ds.read([1,2,3]).transpose(1,2,0)
                valid=(ds.read_masks([1,2,3])>0).all(axis=0)
                transform=ds.transform; crs=ds.crs
    else:
        from PIL import Image
        rgb=np.asarray(Image.open(BytesIO(raw)).convert("RGB"))
        valid=np.ones(rgb.shape[:2],dtype=bool); transform=None; crs=None
    if rgb.shape[:2] != (info.height,info.width): raise ValueError("Decoded image dimensions differ from inspected metadata")
    return rgb,valid,transform,crs

def validate_comparison(info_a,info_b,confirm_pixel_alignment=False):
    from .image_alignment import alignment_diagnostic
    if (info_a.width,info_a.height)!=(info_b.width,info_b.height):
        raise ValueError("Analysis blocked: images have different dimensions. Resample/register both to the same grid first.")
    diag=alignment_diagnostic(info_a,info_b)
    if info_a.georeferenced or info_b.georeferenced:
        if not (info_a.georeferenced and info_b.georeferenced):
            raise ValueError("Analysis blocked: only one image has geospatial metadata.")
        if diag["warnings"]:
            raise ValueError("Analysis blocked by geospatial mismatch: " + " ".join(diag["warnings"]))
    elif not confirm_pixel_alignment:
        raise ValueError("Analysis blocked: confirm that both non-georeferenced images cover the same area and pixel grid.")
    return {"mode":"geospatial" if info_a.georeferenced else "pixel-space","warnings":diag["warnings"]}

def _metric_geometries(geometries,crs):
    if not geometries: return []
    import geopandas as gpd
    series=gpd.GeoSeries(geometries,crs=crs)
    metric_crs=series.estimate_utm_crs()
    if metric_crs is None: raise ValueError("Could not determine a local metric CRS for geographic area and distance calculations")
    return series.to_crs(metric_crs).tolist()

def run_analysis(upload_a,upload_b,info_a,info_b,detector,threshold=.5,
                 confirm_pixel_alignment=False,date_a=None,date_b=None,
                 min_footprint_pixels=3,iou_threshold=.25,max_match_distance_m=4.0,
                 area_change_threshold=.35,tile_size=512,tile_overlap=128):
    """Run real model inference on both images. No reference metrics are inferred."""
    import numpy as np
    from shapely import transform as geom_transform
    from pyproj import CRS,Transformer
    from .footprint_extraction import mask_to_features
    from .change_detection import compare_footprints
    from .confidence import review_score

    if date_a and date_b and date_a >= date_b: raise ValueError("Earlier observation date must be before later observation date")
    alignment=validate_comparison(info_a,info_b,confirm_pixel_alignment)
    rgb_a,valid_a,transform_a,crs_a=_read_rgb(upload_a,info_a)
    rgb_b,valid_b,transform_b,crs_b=_read_rgb(upload_b,info_b)
    if not detector.model: detector.load()
    prob_a,mask_a=detector.predict(rgb_a,threshold,tile_size,tile_overlap,valid_a)
    prob_b,mask_b=detector.predict(rgb_b,threshold,tile_size,tile_overlap,valid_b)
    mask_a &= valid_a; mask_b &= valid_b
    feats_a=mask_to_features(mask_a,prob_a,transform_a,min_footprint_pixels)
    feats_b=mask_to_features(mask_b,prob_b,transform_b,min_footprint_pixels)
    geoms_a=[f["geometry"] for f in feats_a]; geoms_b=[f["geometry"] for f in feats_b]
    if alignment["mode"]=="geospatial":
        source_crs=CRS.from_user_input(crs_a)
        metric_a=_metric_geometries(geoms_a,source_crs); metric_b=_metric_geometries(geoms_b,source_crs)
        area_a=[g.area for g in metric_a]; area_b=[g.area for g in metric_b]
        max_dist=max_match_distance_m
        to_wgs84=Transformer.from_crs(source_crs,"EPSG:4326",always_xy=True).transform
    else:
        metric_a=geoms_a; metric_b=geoms_b
        area_a=[None]*len(feats_a); area_b=[None]*len(feats_b)
        max_dist=3.0
        to_wgs84=None
    from .image_alignment import estimate_translation
    registration=estimate_translation(rgb_a,rgb_b)
    shift=max(abs(registration["shift_x_pixels"] or 0),abs(registration["shift_y_pixels"] or 0))
    tolerance=max(2,registration["pixel_precision"]*2)
    if registration["reliable"] and shift>tolerance:
        raise ValueError(f"Image registration check found an approximately {shift}px shift. Register both images to a common grid and retry; no automatic shift is applied.")
    alignment["registration"]=registration
    if not registration["reliable"]:
        alignment["warnings"].append("Pixel-content registration estimate has low confidence; change categories require extra human review.")
    elif shift>0:
        alignment["warnings"].append(f"Estimated residual image shift: {registration['shift_x_pixels']}px x, {registration['shift_y_pixels']}px y.")
    changes=compare_footprints(metric_a,metric_b,iou_threshold,max_dist,area_change_threshold,reliable=registration["reliable"])
    records=[]; output_features=[]
    for c in changes:
        fa=feats_a[c.before_index] if c.before_index is not None else None
        fb=feats_b[c.after_index] if c.after_index is not None else None
        geom=(fb or fa)["geometry"] if (fb or fa) else None
        conf_a=fa["confidence"] if fa else None; conf_b=fb["confidence"] if fb else None
        confs=[x for x in (conf_a,conf_b) if x is not None]
        conf=sum(confs)/len(confs) if confs else None
        score=review_score(model_confidence=conf,similarity=c.similarity,reliable=registration["reliable"])
        aa=area_a[c.before_index] if c.before_index is not None else None
        ab=area_b[c.after_index] if c.after_index is not None else None
        flags=list(c.flags or [])
        if not registration["reliable"]: flags.append("registration_uncertain")
        row={"id":c.id,"category":c.category,"date_before":str(date_a or ""),"date_after":str(date_b or ""),
             "area_before_m2":aa,"area_after_m2":ab,"area_delta_m2":(ab-aa) if aa is not None and ab is not None else None,
             "similarity":c.similarity,"model_confidence_before":conf_a,"model_confidence_after":conf_b,
             "review_score_heuristic":score,"human_review":"pending" if c.category!="uncertain" else "needs_review",
             "warning_flags":flags}
        records.append(row)
        if geom is not None:
            export_geom=geom_transform(geom,lambda xy: np.column_stack(to_wgs84(xy[:,0],xy[:,1]))) if to_wgs84 else geom
            output_features.append(({k:v for k,v in row.items() if k not in {"warning_flags"}},export_geom))
    # Vectorized visual overlays: magenta=new, orange=removed, cyan=modified, white=unchanged.
    categories={k:[] for k in ("new","removed","modified","unchanged","uncertain")}
    for c in changes:
        if c.category in categories:
            feats=feats_b if c.after_index is not None else feats_a
            idx=c.after_index if c.after_index is not None else c.before_index
            if idx is not None: categories[c.category].append(feats[idx]["geometry"])
    return {"alignment":alignment,"probabilities_before":prob_a,"probabilities_after":prob_b,
            "mask_before":mask_a,"mask_after":mask_b,"rgb_before":rgb_a,"rgb_after":rgb_b,
            "records":records,"features":output_features,"crs":str(crs_a) if crs_a else None,
            "georeferenced":alignment["mode"]=="geospatial","categories":categories}

def render_mask_overlay(rgb,mask,color=(255,40,80),alpha=.45):
    """Create a preview overlay without changing the source imagery."""
    import numpy as np
    src=np.asarray(rgb).astype(np.float32)
    finite=np.isfinite(src)
    lo=float(src[finite].min()) if finite.any() else 0.0
    hi=float(src[finite].max()) if finite.any() else 1.0
    src=np.nan_to_num(src,nan=lo,posinf=hi,neginf=lo)
    src=np.clip((src-lo)/(max(hi-lo,1e-6))*255,0,255)
    out=src.copy(); selected=np.asarray(mask,dtype=bool)
    out[selected]=out[selected]*(1-alpha)+np.asarray(color,dtype=np.float32)*alpha
    return np.clip(out,0,255).astype("uint8")
