"""Streamlit application for local building-footprint change analysis."""
import hashlib
import json
from collections import Counter
from datetime import date,timedelta

import pandas as pd
import streamlit as st

from config.settings import (MODEL_CHECKPOINT,MAX_UPLOAD_MB,DEFAULT_CONFIDENCE,
    DEFAULT_IOU,DEFAULT_CENTROID_DISTANCE,DEFAULT_AREA_CHANGE,TILE_SIZE,TILE_OVERLAP)
from src.image_validation import inspect_image,compare_metadata
from src.building_detector import BuildingDetector,ModelUnavailable
from src.analysis_pipeline import run_analysis,validate_comparison,render_mask_overlay
from src.export_utils import csv_bytes,geojson_bytes,summary_json_bytes

st.set_page_config(page_title="Urban Sprawl Intelligence",page_icon="🛰️",layout="wide")
st.title("🛰️ Urban Sprawl Intelligence")
st.caption("AI-assisted satellite change analysis · Model predictions require human review")

@st.cache_resource(show_spinner="Loading the SpaceNet building segmentation model…")
def get_detector(path):
    detector=BuildingDetector(path)
    try: detector.load(); return detector,None
    except ModelUnavailable as exc: return detector,str(exc)

detector,model_error=get_detector(str(MODEL_CHECKPOINT))
st.sidebar.subheader("Building model")
if detector.model is not None:
    st.sidebar.success(detector.status)
    st.sidebar.caption("DeepLabV3-ResNet50 · SpaceNet 2 Paris RGB · CPU/GPU")
else:
    st.sidebar.error(detector.status)
    st.sidebar.caption(model_error or "Checkpoint unavailable")
st.sidebar.caption("Checkpoint weights are excluded from source control. See model setup for provenance and download instructions.")

def _inspect_upload(file):
    if file is None: return None
    if file.size>MAX_UPLOAD_MB*1024*1024:
        st.error(f"File exceeds the {MAX_UPLOAD_MB} MB upload limit."); return None
    try: return inspect_image(file,file.name)
    except Exception as exc: st.error(f"{file.name}: {exc}"); return None

today=date.today()
try: earlier_default=today.replace(year=today.year-1)
except ValueError: earlier_default=today-timedelta(days=365)
columns=st.columns(2)
files=[]; infos=[]; dates=[]
for idx,(col,label,default) in enumerate(zip(columns,["Earlier observation (A)","Later observation (B)"],[earlier_default,today])):
    with col:
        st.subheader(label)
        file=st.file_uploader("Upload GeoTIFF, PNG, or JPEG",type=["tif","tiff","png","jpg","jpeg"],key=f"image{idx}")
        acquisition_date=st.date_input("Acquisition date",value=default,key=f"date{idx}")
        files.append(file); dates.append(acquisition_date)
        info=_inspect_upload(file); infos.append(info)
        if file and info:
            st.image(file.getvalue(),caption=f"{info.filename} · {info.width} × {info.height} · {info.bands} bands",use_container_width=True)
            with st.expander("Image metadata"):
                st.json(info.to_dict())
            for warning in info.warnings or []: st.warning(warning)

if all(files) and all(infos):
    st.subheader("Comparison readiness")
    warnings=compare_metadata(infos[0],infos[1])
    for warning in warnings: st.warning(warning)
    is_pixel=not(infos[0].georeferenced or infos[1].georeferenced)
    confirmed=True
    if is_pixel:
        st.info("No valid CRS/transform: any results remain in pixel coordinates, with no geographic area claims or GeoJSON export.")
        confirmed=st.checkbox("I confirm the images show the same area, have the same coverage, and are already aligned",key="confirm_alignment")
    alignment_error=None
    try: validate_comparison(infos[0],infos[1],confirmed)
    except ValueError as exc: alignment_error=str(exc)
    if alignment_error: st.error(alignment_error)
    if dates[0]>=dates[1]: st.error("Choose an earlier acquisition date for image A than for image B.")
    threshold=st.slider("Building probability threshold",0.05,0.95,DEFAULT_CONFIDENCE,0.05,
        help="Higher thresholds generally reduce false positives but can miss low-contrast or small roofs. Validate locally.")
    can_run=detector.model is not None and alignment_error is None and dates[0]<dates[1]
    run=st.button("Run building-change analysis",type="primary",disabled=not can_run)
    hasher=hashlib.sha256()
    hasher.update(files[0].getvalue()); hasher.update(files[1].getvalue())
    hasher.update(str(dates).encode()); hasher.update(str(threshold).encode()); hasher.update(str(confirmed).encode())
    digest=hasher.hexdigest()
    if st.session_state.get("analysis_digest")!=digest:
        st.session_state.pop("analysis_result",None); st.session_state["analysis_digest"]=digest
    if run:
        try:
            with st.status("Analyzing both observations…",expanded=True) as status:
                st.write("Running tiled DeepLab inference for image A")
                result=run_analysis(files[0],files[1],infos[0],infos[1],detector,threshold,
                    confirmed,dates[0],dates[1],iou_threshold=DEFAULT_IOU,
                    max_match_distance_m=DEFAULT_CENTROID_DISTANCE,
                    area_change_threshold=DEFAULT_AREA_CHANGE,tile_size=TILE_SIZE,tile_overlap=TILE_OVERLAP)
                st.session_state["analysis_result"]=result
                status.update(label="Change analysis complete",state="complete",expanded=False)
        except Exception as exc:
            st.error(f"Analysis failed: {exc}")

    result=st.session_state.get("analysis_result")
    if result:
        rows=result["records"]
        st.divider(); st.subheader("Observed building-footprint differences")
        st.warning("These are model predictions from imagery, not construction or legal findings. Confirm each row against source imagery and authoritative records where needed.")
        for warning in result["alignment"]["warnings"]: st.warning(warning)
        counts=Counter(row["category"] for row in rows)
        metric_cols=st.columns(5)
        for col,key,label in zip(metric_cols,["new","removed","modified","unchanged","uncertain"],["Potential additions","Potential removals","Potentially modified","Matched","Needs review"]):
            col.metric(label,counts.get(key,0))
        preview_cols=st.columns(2)
        with preview_cols[0]:
            st.image(render_mask_overlay(result["rgb_before"],result["mask_before"]),caption=f"{files[0].name} · detected footprints",use_container_width=True)
            st.image(result["mask_before"].astype("uint8")*255,caption="Image A building mask",use_container_width=True)
        with preview_cols[1]:
            st.image(render_mask_overlay(result["rgb_after"],result["mask_after"]),caption=f"{files[1].name} · detected footprints",use_container_width=True)
            st.image(result["mask_after"].astype("uint8")*255,caption="Image B building mask",use_container_width=True)

        st.markdown("**Review results:** `human_review` starts as pending. Edit it to record a human disposition; model category and model confidence remain separate fields.")
        frame=pd.DataFrame(rows)
        if len(frame):
            frame=st.data_editor(frame,key="review_table",use_container_width=True,hide_index=True,
                disabled=[c for c in frame.columns if c!="human_review"],
                column_config={"human_review":st.column_config.SelectboxColumn("Human review",options=["pending","confirmed","rejected","needs_review"]),
                               "review_score_heuristic":st.column_config.NumberColumn("Review score (heuristic)",format="%.2f")})
            edited=frame.to_dict("records")
            st.download_button("Download results CSV",csv_bytes(edited),"urban_change_results.csv","text/csv")
            if result["georeferenced"]:
                features=[]
                for (old_props,geom),row in zip(result["features"],edited):
                    props=dict(old_props); props["human_review"]=row["human_review"]
                    features.append((props,geom))
                geo=geojson_bytes(features,crs="EPSG:4326",georeferenced=True)
                st.download_button("Download change GeoJSON (WGS84)",geo,"urban_change_results.geojson","application/geo+json")
                import folium
                from streamlit_folium import st_folium
                center=[0,0]
                if features:
                    c=features[0][1].centroid; center=[c.y,c.x]
                map_obj=folium.Map(location=center,zoom_start=14,control_scale=True)
                colors={"new":"#d81b60","removed":"#f57c00","modified":"#00acc1","unchanged":"#43a047","uncertain":"#6d4c41"}
                for props,geom in features:
                    folium.GeoJson({"type":"Feature","geometry":__import__("shapely.geometry",fromlist=["mapping"]).mapping(geom),"properties":props},
                        name=props.get("category","change"),style_function=lambda _,p=props:{"color":colors.get(p.get("category"),"#777"),"weight":2,"fillOpacity":.25},
                        tooltip=f"{props.get('id')} · {props.get('category')} · human review: {props.get('human_review')}").add_to(map_obj)
                folium.LayerControl().add_to(map_obj)
                st.subheader("Interactive change map")
                st.caption("Map tiles are retrieved from OpenStreetMap by the browser; uploaded imagery and model inference stay local.")
                st_folium(map_obj,use_container_width=True,height=550,returned_objects=[])
            else:
                st.info("Pixel-space analysis: geographic map and GeoJSON export are disabled.")
            summary={"project":"Urban Sprawl Intelligence","dates":[str(x) for x in dates],
                "inputs":[i.to_dict() for i in infos],"model":"DeepLabV3-ResNet50 SpaceNet-2 Paris RGB checkpoint",
                "threshold":threshold,"alignment":result["alignment"],"counts":dict(counts),
                "limitations":["Observed image changes are not legal findings","Model domain is SpaceNet-2 Paris high-resolution RGB","Review all predictions"]}
            st.download_button("Download analysis summary JSON",summary_json_bytes(summary),"analysis_summary.json","application/json")
        else:
            st.success("No building footprint components passed the configured minimum-area threshold.")
else:
    st.info("Upload two valid images to continue.")
    if model_error: st.error(model_error)

st.divider()
st.caption("Cloud, shadows, seasonal vegetation, acquisition angle, resolution, and registration differences can affect detections. New footprints do not prove when a structure was built or whether it is authorized.")
