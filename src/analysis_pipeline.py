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
        valid=np.ones(rgb.shape[:2],dtype=bool)
        if info.georeferenced and info.transform and info.crs:
            from affine import Affine
            transform=Affine(*info.transform); crs=info.crs
        else:
            transform=None; crs=None
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

def _apply_registration_gate(alignment,registration,confirm_pixel_alignment,trust_georeferenced_grid=False):
    """Block uncorrected shifts for geospatial data; allow confirmed pixel-space with caution."""
    shift=max(abs(registration["shift_x_pixels"] or 0),abs(registration["shift_y_pixels"] or 0))
    tolerance=max(2,registration["pixel_precision"]*2)
    if registration["reliable"] and shift>tolerance:
        if alignment["mode"]=="geospatial" and trust_georeferenced_grid:
            registration=dict(registration)
            alignment["warnings"].append(
                f"Pixel-content check estimated an approximately {shift}px shift, but the retrieved observations share the same validated CRS, extent, and affine grid. No automatic shift was applied; seasonal or sensor differences can still affect predictions."
            )
            alignment["registration"]=registration
            return alignment,registration
        if alignment["mode"]!="pixel-space" or not confirm_pixel_alignment:
            raise ValueError(f"Image registration check found an approximately {shift}px shift. Register both images to a common grid and retry; no automatic shift is applied.")
        # The user confirmed these pixel-space uploads are aligned. Honor that choice,
        # but lower match reliability so candidate changes are treated as uncertain.
        registration=dict(registration)
        registration["reliable"]=False
        alignment["warnings"].append(
            f"Registration check estimated an approximately {shift}px shift. Continuing because you confirmed alignment; no correction was applied, so change masks and label metrics may be affected. Building matches require review."
        )
    alignment["registration"]=registration
    if not registration["reliable"] and not any("Registration check estimated" in w for w in alignment["warnings"]):
        alignment["warnings"].append("Pixel-content registration estimate has low confidence; change categories require extra human review.")
    elif registration["reliable"] and shift>0:
        alignment["warnings"].append(f"Estimated residual image shift: {registration['shift_x_pixels']}px x, {registration['shift_y_pixels']}px y.")
    return alignment,registration

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
                 area_change_threshold=.35,tile_size=512,tile_overlap=128,
                 reference_change_mask=None,change_detector=None,change_threshold=.5,
                 trust_georeferenced_grid=False):
    """Infer both dates; an optional reference mask is evaluated only afterward."""
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
    common_valid=valid_a & valid_b
    if reference_change_mask is not None:
        reference_change_mask=np.asarray(reference_change_mask,dtype=bool)
        if reference_change_mask.shape != common_valid.shape:
            raise ValueError("Reference change-label dimensions must match both uploaded images")
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
    alignment,registration=_apply_registration_gate(alignment,registration,confirm_pixel_alignment,trust_georeferenced_grid)
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
    # Pixel-space display geometries are separate from map geometries used for measurement/export.
    display_a=mask_to_features(mask_a,prob_a,min_pixels=min_footprint_pixels)
    display_b=mask_to_features(mask_b,prob_b,min_pixels=min_footprint_pixels)
    if change_detector is not None:
        change_probability,change_mask=change_detector.predict_pair(rgb_a,rgb_b,change_threshold)
        change_mask=np.asarray(change_mask,dtype=bool) & common_valid
    else:
        # Compatibility path for callers without the LEVIR-tuned temporal model.
        change_probability=None
        change_mask=(mask_a ^ mask_b) & common_valid
    change_features=mask_to_features(change_mask,min_pixels=1)
    # The BIT model's binary change mask is directionless. Preserve the user's
    # requested heuristic: changed components are possible additions. Removals
    # are before-date footprint components with no after-date match.
    possible_removals=sum(
        c.before_index is not None and c.after_index is None for c in changes
    )
    possible_additions=len(change_features)
    uncertain_a=[]; uncertain_b=[]
    for c in changes:
        if c.category=="uncertain":
            if c.before_index is not None: uncertain_a.append(display_a[c.before_index]["geometry"])
            if c.after_index is not None: uncertain_b.append(display_b[c.after_index]["geometry"])
    from .metrics import binary_metrics,summarize_change_area
    evaluation=(binary_metrics(change_mask,reference_change_mask,common_valid)
                if reference_change_mask is not None else None)
    change_area=summarize_change_area(change_mask,common_valid)
    # Vectorized display geometry for categories remains useful to downstream callers.
    categories={k:[] for k in ("new","removed","modified","unchanged","uncertain")}
    for c in changes:
        if c.category in categories:
            feats=feats_b if c.after_index is not None else feats_a
            idx=c.after_index if c.after_index is not None else c.before_index
            if idx is not None: categories[c.category].append(feats[idx]["geometry"])
    return {"alignment":alignment,"probabilities_before":prob_a,"probabilities_after":prob_b,
            "mask_before":mask_a,"mask_after":mask_b,"rgb_before":rgb_a,"rgb_after":rgb_b,
            "change_probability":change_probability,
            "change_model":getattr(change_detector,"model_name",None) if change_detector else "XOR of building masks (fallback)",
            "building_features_before":display_a,"building_features_after":display_b,
            "change_mask":change_mask,"change_features":change_features,
            "possible_additions":possible_additions,
            "possible_removals":possible_removals,
            "net_building_count_change_percent":(
                100.0*(possible_additions-possible_removals)/len(display_a)
                if len(display_a)>0 else None
            ),
            "uncertain_before":uncertain_a,"uncertain_after":uncertain_b,
            "building_counts":{"before":len(display_a),"after":len(display_b)},
            "change_area":change_area,
            "evaluation":evaluation,"records":records,"features":output_features,
            "crs":str(crs_a) if crs_a else None,
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

def render_change_overlay(rgb,building_features,change_features,uncertain_geometries=(),
                          change_opacity=.10):
    """Overlay green building outlines, translucent red changes, and orange uncertainty."""
    import numpy as np
    from PIL import Image,ImageDraw

    source=np.asarray(rgb)
    if source.ndim!=3 or source.shape[2]<3:
        raise ValueError("Overlay requires an RGB image")
    finite=np.isfinite(source[:,:,:3])
    values=source[:,:,:3].astype(np.float32)
    lo=float(values[finite].min()) if finite.any() else 0.0
    hi=float(values[finite].max()) if finite.any() else 1.0
    values=np.nan_to_num(values,nan=lo,posinf=hi,neginf=lo)
    if hi>1.0: values=np.clip((values-lo)/max(hi-lo,1e-6)*255,0,255)
    base=Image.fromarray(values.astype("uint8"),mode="RGB").convert("RGBA")
    layer=Image.new("RGBA",base.size,(0,0,0,0)); draw=ImageDraw.Draw(layer,"RGBA")

    def polygon_parts(geom):
        if geom is None or geom.is_empty: return []
        if geom.geom_type=="Polygon": return [geom]
        if geom.geom_type=="MultiPolygon": return list(geom.geoms)
        if geom.geom_type=="GeometryCollection": return [g for g in geom.geoms if g.geom_type=="Polygon"]
        return []

    def points(ring): return [(int(round(x)),int(round(y))) for x,y in ring.coords]
    alpha=int(round(255*max(0,min(1,change_opacity))))
    for feature in change_features:
        for poly in polygon_parts(feature.get("geometry")):
            draw.polygon(points(poly.exterior),fill=(255,0,0,alpha),outline=(255,0,0,220))
            for hole in poly.interiors: draw.polygon(points(hole),fill=(0,0,0,0))
    for feature in building_features:
        for poly in polygon_parts(feature.get("geometry")):
            draw.line(points(poly.exterior),fill=(0,255,80,255),width=2,joint="curve")
            for hole in poly.interiors: draw.line(points(hole),fill=(0,255,80,255),width=2,joint="curve")
    for geom in uncertain_geometries:
        for poly in polygon_parts(geom):
            draw.line(points(poly.exterior),fill=(255,190,0,255),width=3,joint="curve")
    return np.asarray(Image.alpha_composite(base,layer).convert("RGB"))
