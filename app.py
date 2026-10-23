"""Streamlit application for local building-footprint change analysis."""
import hashlib
import json
from collections import Counter
from datetime import date,timedelta

import pandas as pd
import streamlit as st

from config.settings import (MODEL_CHECKPOINT,MAX_UPLOAD_MB,DEFAULT_CONFIDENCE,
    DEFAULT_IOU,DEFAULT_CENTROID_DISTANCE,DEFAULT_AREA_CHANGE,TILE_SIZE,TILE_OVERLAP,
    LEVIR_CHANGE_CHECKPOINT)
from src.image_validation import inspect_image,compare_metadata
from src.building_detector import BuildingDetector,ModelUnavailable
from src.analysis_pipeline import run_analysis,validate_comparison,render_change_overlay
from src.export_utils import (csv_bytes,geojson_bytes,summary_json_bytes,
    mask_png_bytes,prediction_polygons_json_bytes)
from src.reference_labels import decode_change_label
from src.geospatial_sidecar import attach_levir_coordinates
from src.change_detector import LevirChangeDetector,ChangeModelUnavailable

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

@st.cache_resource(show_spinner="Loading the LEVIR change-detection model…")
def get_change_detector(path):
    model=LevirChangeDetector(path)
    try: model.load(); return model,None
    except ChangeModelUnavailable as exc: return model,str(exc)

change_detector,change_model_error=get_change_detector(str(LEVIR_CHANGE_CHECKPOINT))
st.sidebar.subheader("Change model")
if change_detector.session is not None:
    st.sidebar.success(change_detector.status)
    st.sidebar.caption("BIT · trained on LEVIR-CD · ONNX CPU inference")
    st.sidebar.caption("Checkpoint terms: non-commercial use only")
else:
    st.sidebar.error(change_detector.status)
    st.sidebar.caption(change_model_error or "Checkpoint unavailable")

def _inspect_upload(file):
    if file is None: return None
    if file.size>MAX_UPLOAD_MB*1024*1024:
        st.error(f"File exceeds the {MAX_UPLOAD_MB} MB upload limit."); return None
    try: return inspect_image(file,file.name)
    except Exception as exc: st.error(f"{file.name}: {exc}"); return None

today=date.today()
try: earlier_default=today.replace(year=today.year-1)
except ValueError: earlier_default=today-timedelta(days=365)
input_mode=st.radio("Image source",["Manual image upload","Annual urbanization trend (Sentinel-2-derived)"],horizontal=True,key="image_source_mode")
files=[]; infos=[]; dates=[]
if input_mode=="Annual urbanization trend (Sentinel-2-derived)":
    st.subheader("Annual urbanization trend Â· map-selected area")
    st.caption("Click the map to choose an area. This reads Google's precomputed Sentinel-2-derived building estimates; it does not fetch original Sentinel-2 photographs.")
    import folium
    import os
    from streamlit_folium import st_folium
    selected_location=st.session_state.get("annual_trend_location")
    map_obj=folium.Map(location=selected_location or [20.5937,78.9629],zoom_start=12 if selected_location else 5,control_scale=True)
    if selected_location:
        folium.Marker(selected_location,tooltip="Selected trend center").add_to(map_obj)
        map_radius_km=float(st.session_state.get("annual_radius_km",1.0))
        folium.Circle(location=selected_location,radius=map_radius_km*1000,color="#ff4b4b",weight=2,fill=True,fill_color="#ff4b4b",fill_opacity=0.06,tooltip=f"Analysis boundary · {map_radius_km:g} km radius").add_to(map_obj)
    result=st.session_state.get("annual_trend_result")
    result_config=st.session_state.get("annual_trend_config")
    map_config=(tuple(selected_location) if selected_location else (),
        float(st.session_state.get("annual_radius_km",1.0)),
        int(st.session_state.get("annual_start_year",2016)),
        int(st.session_state.get("annual_end_year",2023)),
        float(st.session_state.get("annual_presence_threshold",0.50)),
        str(st.session_state.get("annual_ee_project",os.environ.get("EARTHENGINE_PROJECT",""))).strip())
    if result and result_config and map_config==result_config:
        for key,label in (("start","Start year building presence"),("end","End year building presence"),("changed","Changed built-up pixels")):
            folium.TileLayer(tiles=result["tiles"][key],attr="Google Earth Engine Â· Google Research",name=label,overlay=True,control=True,show=(key=="changed")).add_to(map_obj)
        folium.LayerControl(collapsed=False).add_to(map_obj)
    map_state=st_folium(map_obj,key="annual_trend_map",height=400,use_container_width=True)
    clicked=(map_state or {}).get("last_clicked")
    if clicked:
        selected_location=[float(clicked["lat"]),float(clicked["lng"])]
        st.session_state["annual_trend_location"]=selected_location
    if selected_location: st.caption(f"Selected point: {selected_location[0]:.5f}Â° latitude, {selected_location[1]:.5f}Â° longitude")
    else: st.info("Click the map to choose a point. The map opens centered on India.")
    ee_project=st.text_input("Google Cloud project ID registered for Earth Engine",value=os.environ.get("EARTHENGINE_PROJECT",""),key="annual_ee_project",help="Earth Engine API access and local authentication are required. Setup steps are in README.md.")
    controls=st.columns(4)
    with controls[0]: radius_km=st.number_input("Area radius (km)",min_value=0.25,max_value=3.0,value=1.0,step=0.25,key="annual_radius_km")
    with controls[1]: start_year=st.selectbox("Start year",range(2016,2024),index=0,key="annual_start_year")
    with controls[2]: end_year=st.selectbox("End year",range(2016,2024),index=7,key="annual_end_year")
    with controls[3]: presence_threshold=st.slider("Building presence threshold",0.10,0.90,0.50,0.05,key="annual_presence_threshold",help="Model confidence is uncalibrated; this threshold is not a literal probability.")
    st.caption("Annual dataset coverage: 2016–2023. A year is a modeled annual state, not an exact satellite acquisition date.")
    config=(tuple(selected_location) if selected_location else (),float(radius_km),int(start_year),int(end_year),float(presence_threshold),ee_project.strip())
    if st.button("Calculate annual trend",type="primary",disabled=not selected_location or start_year>=end_year):
        try:
            from src.annual_trend import retrieve_annual_trend
            with st.spinner("Querying Open Buildings annual data and calculating the trend…"):
                trend=retrieve_annual_trend(ee_project,selected_location[0],selected_location[1],radius_km,start_year,end_year,presence_threshold)
            st.session_state["annual_trend_result"]=trend
            st.session_state["annual_trend_config"]=config
            st.rerun()
        except Exception as exc: st.error(f"Annual trend could not be calculated: {exc}")
    trend=st.session_state.get("annual_trend_result")
    if trend and st.session_state.get("annual_trend_config")==config:
        st.subheader(f"Estimated built-up trend Â· {start_year}–{end_year}")
        st.warning("These are model-derived estimates, not verified building counts. Open Buildings confidence is uncalibrated and may vary across years; use local reference data before making strong claims.")
        metrics=st.columns(5)
        metrics[0].metric(f"Estimated building count Â· {start_year}",f"{trend['start_count']:,.1f}")
        metrics[1].metric(f"Estimated building count Â· {end_year}",f"{trend['end_count']:,.1f}")
        metrics[2].metric("Estimated net count change",f"{trend['net_count_change']:+,.1f}")
        metrics[3].metric("Built-up area change",f"{trend['built_area_change_percent']:+.2f}%" if trend["built_area_change_percent"] is not None else "N/A")
        metrics[4].metric("Annualized built-up area growth",f"{trend['annualized_built_area_growth_percent']:+.2f}% / yr" if trend["annualized_built_area_growth_percent"] is not None else "N/A")
        changes=st.columns(3)
        changes[0].metric("New built-up footprint area",f"{trend['added_built_area_m2']:,.0f} mÂ²")
        changes[1].metric("No-longer-detected area",f"{trend['removed_built_area_m2']:,.0f} mÂ²")
        changes[2].metric("Changed area share",f"{trend['changed_area_percent']:.2f}%")
        st.caption(f"Changed area share = (new + no-longer-detected built-up pixels) Ã· valid compared area Ã— 100. Valid area: {trend['valid_area_m2']:,.0f} mÂ². Areas are raster pixels, not distinct building counts.")
        table=pd.DataFrame(trend["yearly"]).rename(columns={"year":"Year","estimated_buildings":"Estimated building count","building_count_change":"Building count change vs prior year","building_count_change_percent":"Building count change vs prior year (%)","built_area_m2":"Estimated built-up area (mÂ²)","built_area_change_m2":"Built-up area change vs prior year (mÂ²)","built_area_change_percent":"Built-up area change vs prior year (%)"})
        st.line_chart(table.set_index("Year")[["Estimated building count"]])
        st.line_chart(table.set_index("Year")[["Estimated built-up area (mÂ²)"]])
        st.dataframe(table,use_container_width=True,hide_index=True)
        st.download_button("Download annual trend CSV",table.to_csv(index=False).encode("utf-8"),file_name="urban_sprawl_annual_trend.csv",mime="text/csv")
        st.caption("Map layers show Google Open Buildings Temporal model rasters, not the original Sentinel-2 scenes. See README.md for source and limitations.")
    elif trend: st.info("The location or analysis settings changed. Recalculate to update this result.")
    st.stop()
else:
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

st.subheader("Optional reference label for evaluation")
label_file=st.file_uploader("Upload the matching LEVIR change-label PNG (held out from inference)",
    type=["png"],key="reference_change_label")
reference_mask=None; label_error=None
if label_file and all(infos):
    try:
        expected=(infos[0].height,infos[0].width)
        reference_mask=decode_change_label(label_file.getvalue(),label_file.name,expected)
        st.caption(f"Reference label loaded: {int(reference_mask.sum()):,} changed pixels. It is used only for evaluation after inference.")
    except ValueError as exc:
        label_error=str(exc); st.error(label_error)
elif not label_file:
    st.info("No reference label uploaded: change-mask evaluation metrics will be unavailable.")

st.subheader("Optional LEVIR geographic coordinates")
coordinate_file=st.file_uploader("Upload LEVIR_CD_name_coords.json or .geojson to enable a map for a matching sample",
    type=["json","geojson"],key="coordinate_sidecar")
coordinate_error=None
if coordinate_file and all(infos):
    try:
        infos=[attach_levir_coordinates(info,coordinate_file.getvalue(),coordinate_file.name) for info in infos]
        st.warning("Coordinates were attached by exact sample filename from the sidecar. The image grid is assumed north-up and rectangular; inspect the bounds before relying on map placement or area estimates.")
    except ValueError as exc:
        coordinate_error=str(exc); st.error(coordinate_error)

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
    building_threshold=st.slider("Building-footprint probability threshold",0.05,0.95,DEFAULT_CONFIDENCE,0.05,
        help="Controls the separate SpaceNet-trained building outlines. It does not control the LEVIR change mask.")
    change_threshold=st.slider("Change probability threshold",0.05,0.95,0.50,0.05,
        help="Applied to the LEVIR-trained temporal model output. Select thresholds using validation data, not a final test set.")
    can_run=(detector.model is not None and change_detector.session is not None and alignment_error is None and dates[0]<dates[1]
             and label_error is None and coordinate_error is None)
    run=st.button("Run building-change analysis",type="primary",disabled=not can_run)
    hasher=hashlib.sha256()
    hasher.update(files[0].getvalue()); hasher.update(files[1].getvalue())
    if label_file: hasher.update(label_file.getvalue())
    if coordinate_file: hasher.update(coordinate_file.getvalue())
    hasher.update(str(dates).encode()); hasher.update(str(building_threshold).encode())
    hasher.update(str(change_threshold).encode()); hasher.update(str(confirmed).encode())
    digest=hasher.hexdigest()
    if st.session_state.get("analysis_digest")!=digest:
        st.session_state.pop("analysis_result",None); st.session_state["analysis_digest"]=digest
    if run:
        try:
            with st.status("Analyzing both observations…",expanded=True) as status:
                st.write("Running tiled DeepLab inference for image A")
                st.write("Running LEVIR-trained BIT change inference on the image pair")
                result=run_analysis(files[0],files[1],infos[0],infos[1],detector,building_threshold,
                    confirmed,dates[0],dates[1],iou_threshold=DEFAULT_IOU,
                    max_match_distance_m=DEFAULT_CENTROID_DISTANCE,
                    area_change_threshold=DEFAULT_AREA_CHANGE,tile_size=TILE_SIZE,tile_overlap=TILE_OVERLAP,
                    reference_change_mask=reference_mask,change_detector=change_detector,
                    change_threshold=change_threshold,
                    trust_georeferenced_grid=input_mode=="Find satellite images by location and dates")
                st.session_state["analysis_result"]=result
                status.update(label="Change analysis complete",state="complete",expanded=False)
        except Exception as exc:
            st.error(f"Analysis failed: {exc}")

    result=st.session_state.get("analysis_result")
    if result:
        rows=result["records"]
        st.divider(); st.subheader("Observed building-footprint differences")
        st.warning("These are model predictions from imagery, not construction or legal findings. Confirm each row against source imagery and authoritative records where needed.")
        if result["change_probability"] is not None:
            st.info("Change-mask metrics below evaluate the LEVIR-trained BIT model. Green building outlines and per-date building counts still come from the separate SpaceNet Paris model and are not validated on LEVIR; the change-mask F1 score is not building-count accuracy.")
        for warning in result["alignment"]["warnings"]: st.warning(warning)
        counts=Counter(row["category"] for row in rows)
        if result["change_probability"] is not None:
            st.markdown("**Possible building changes**")
            change_cols=st.columns(3)
            change_cols[0].metric("Possible additions",result["possible_additions"])
            change_cols[1].metric("Possible removals",result["possible_removals"])
            net_percent=result["net_building_count_change_percent"]
            change_cols[2].metric("Estimated net count change",f"{net_percent:.2f}%" if net_percent is not None else "N/A")
            st.caption("Requested estimate: possible additions = connected red BIT change regions; possible removals = before-date footprint detections with no after-date match. Net count change = (additions − removals) ÷ before-date predicted footprints × 100.")
            st.caption("This is a provisional heuristic: BIT detects change without saying whether it is new or removed, and connected red regions may include non-building changes or combine nearby roofs. Review the overlays before treating these as building counts.")
            area_cols=st.columns(2)
            area_cols[0].metric("Changed image area",f"{result['change_area']['percent_of_region']:.3f}%")
            area_cols[1].metric("Changed pixels",f"{result['change_area']['changed_pixels']:,}")
        else:
            st.markdown("**Footprint match statistics**")
        registration=result["alignment"].get("registration",{})
        registration_reliable=registration.get("reliable",True)
        if not registration_reliable:
            st.warning(f"Footprint matching assigned {counts.get('uncertain',0)} candidates to review because image alignment is uncertain. Possible additions use the red change-region heuristic; possible removals use before-only footprint detections. Modified and matched counts are unavailable (—).")
        metric_cols=st.columns(5)
        for col,key,label in zip(metric_cols,["new","removed","modified","unchanged","uncertain"],["Potential additions","Potential removals","Potentially modified","Matched","Needs review"]):
            if result["change_probability"] is not None and key=="new":
                value=result["possible_additions"]
            elif result["change_probability"] is not None and key=="removed":
                value=result["possible_removals"]
            else:
                value=counts.get(key,0) if registration_reliable or key=="uncertain" else "—"
            col.metric(label,value)
        summary_cols=st.columns(2)
        summary_cols[0].metric("Building footprints · before (SpaceNet)",result["building_counts"]["before"])
        summary_cols[1].metric("Building footprints · after (SpaceNet)",result["building_counts"]["after"])
        if result["change_probability"] is None:
            st.metric("Predicted region changed",f"{result['change_area']['percent_of_region']:.3f}%")
            st.caption(f"Changed image area = changed pixels ÷ valid comparison pixels × 100 = {result['change_area']['changed_pixels']:,} ÷ {result['change_area']['valid_pixels']:,} × 100. This area percentage is separate from the estimated net count change above.")
        if result["evaluation"] is None:
            st.info("Reference change-mask metrics unavailable: upload the matching LEVIR label PNG to evaluate this pair.")
        else:
            st.subheader("Reference change-mask evaluation")
            st.caption(f"The supplied label was kept out of inference. Metrics compare the {result['change_model']} predicted change mask with that label; they are pixel-level change-mask scores, not building-count accuracy.")
            eval_cols=st.columns(4)
            for col,key,label in zip(eval_cols,["precision","recall","f1","iou"],["Precision","Recall","F1 score","IoU"]):
                col.metric(label,f"{result['evaluation'][key]*100:.2f}%")
            st.caption(f"Evaluated {result['evaluation']['valid_pixels']:,} valid pixels · TP {result['evaluation']['counts']['tp']:,} · FP {result['evaluation']['counts']['fp']:,} · FN {result['evaluation']['counts']['fn']:,}.")
        preview_cols=st.columns(2)
        st.markdown("**Overlay legend:** green outline = predicted building footprint · red fill = predicted changed pixels (10% opacity / 90% transparent) · orange outline = uncertain temporal match.")
        with preview_cols[0]:
            before_overlay=render_change_overlay(result["rgb_before"],result["building_features_before"],
                result["change_features"],result["uncertain_before"],change_opacity=.10)
            st.image(before_overlay,caption=f"{files[0].name} · building and change polygons",use_container_width=True)
        with preview_cols[1]:
            after_overlay=render_change_overlay(result["rgb_after"],result["building_features_after"],
                result["change_features"],result["uncertain_after"],change_opacity=.10)
            st.image(after_overlay,caption=f"{files[1].name} · building and change polygons",use_container_width=True)
        with st.expander("Inspect binary masks"):
            mask_cols=st.columns(3)
            mask_cols[0].image(result["mask_before"].astype("uint8")*255,caption="Before building mask",use_container_width=True)
            mask_cols[1].image(result["mask_after"].astype("uint8")*255,caption="After building mask",use_container_width=True)
            mask_cols[2].image(result["change_mask"].astype("uint8")*255,caption="XOR change mask",use_container_width=True)
        download_cols=st.columns(4)
        download_cols[0].download_button("Download before mask",mask_png_bytes(result["mask_before"]),"before_building_mask.png","image/png")
        download_cols[1].download_button("Download after mask",mask_png_bytes(result["mask_after"]),"after_building_mask.png","image/png")
        download_cols[2].download_button("Download change mask",mask_png_bytes(result["change_mask"]),"predicted_change_mask.png","image/png")
        download_cols[3].download_button("Download prediction polygons JSON",prediction_polygons_json_bytes(
            result["building_features_before"],result["building_features_after"],result["change_features"]),
            "prediction_polygons_pixel_coordinates.json","application/json")

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
                "inputs":[i.to_dict() for i in infos],"building_model":"DeepLabV3-ResNet50 SpaceNet-2 Paris RGB checkpoint",
                "change_model":result["change_model"],
                "building_threshold":building_threshold,"change_threshold":change_threshold,
                "alignment":result["alignment"],"counts":dict(counts),
                "building_counts":result["building_counts"],"change_area":result["change_area"],
                "possible_additions":result["possible_additions"],
                "possible_removals":result["possible_removals"],
                "net_building_count_change_percent":result["net_building_count_change_percent"],
                "change_mask_evaluation":result["evaluation"],
                "limitations":["Observed image changes are not legal findings","Model domain is SpaceNet-2 Paris high-resolution RGB","Review all predictions"]}
            st.download_button("Download analysis summary JSON",summary_json_bytes(summary),"analysis_summary.json","application/json")
        else:
            st.success("No building footprint components passed the configured minimum-area threshold.")
else:
    st.info("Upload two valid images to continue.")
    if model_error: st.error(model_error)

st.divider()
st.caption("Cloud, shadows, seasonal vegetation, acquisition angle, resolution, and registration differences can affect detections. New footprints do not prove when a structure was built or whether it is authorized.")
