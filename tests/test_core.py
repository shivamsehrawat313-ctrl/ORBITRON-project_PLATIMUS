import io
import json
import unittest
from datetime import date
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from affine import Affine
from shapely.geometry import box,Polygon

from src.preprocessing import tile_windows
from src.image_validation import ImageInfo,compare_metadata
from src.image_alignment import alignment_diagnostic
from src.image_alignment import estimate_translation
from src.analysis_pipeline import validate_comparison,run_analysis,render_change_overlay,_apply_registration_gate
from src.building_detector import BuildingDetector,ModelUnavailable
from src.change_detection import compare_footprints
from src.export_utils import (csv_bytes,geojson_bytes,mask_png_bytes,
    prediction_polygons_json_bytes)
from src.footprint_extraction import mask_to_features
from src.metrics import binary_metrics,summarize_change_area
from src.reference_labels import decode_change_label
from src.geospatial_sidecar import attach_levir_coordinates
from src.change_detector import LevirChangeDetector

def geoinfo(filename,*,crs="EPSG:32643",bounds=(0,0,64,64),pixel=(1,-1),transform=None,width=64,height=64,geo=True):
    return ImageInfo(filename,width,height,3,"uint8",crs if geo else None,bounds if geo else None,
        pixel if geo else None,transform=transform or ((0,1,0,64,0,-1,0,0,1) if geo else None),georeferenced=geo,
        warnings=[] if geo else ["No geospatial metadata"])

class CoreTests(unittest.TestCase):
    def test_tiles_cover_edges_and_overlap(self):
        windows=list(tile_windows(700,600,512,64))
        self.assertTrue(any(x+w==700 for x,y,w,h in windows))
        self.assertTrue(any(y+h==600 for x,y,w,h in windows))
        self.assertLess(windows[1][0],windows[0][0]+windows[0][2])

    def test_missing_metadata_and_crs_mismatch(self):
        a=geoinfo("a.tif",crs="EPSG:4326"); b=geoinfo("b.tif",crs="EPSG:3857")
        self.assertTrue(any("CRS differs" in x for x in compare_metadata(a,b)))
        pixel=geoinfo("a.png",geo=False)
        self.assertFalse(pixel.georeferenced); self.assertTrue(pixel.warnings)

    def test_alignment_rejects_crs_transform_and_dimension_mismatch(self):
        a=geoinfo("a.tif")
        with self.assertRaisesRegex(ValueError,"geospatial mismatch"):
            validate_comparison(a,geoinfo("b.tif",crs="EPSG:4326"))
        with self.assertRaisesRegex(ValueError,"dimensions"):
            validate_comparison(a,geoinfo("b.tif",width=32))
        with self.assertRaisesRegex(ValueError,"registered to the same pixel grid"):
            validate_comparison(a,geoinfo("b.tif",transform=(0,2,0,64,0,-2,0,0,1),pixel=(2,-2),bounds=(0,0,64,64)))

    def test_pixel_space_requires_explicit_confirmation(self):
        a=geoinfo("a.png",geo=False); b=geoinfo("b.png",geo=False)
        with self.assertRaisesRegex(ValueError,"confirm"):
            validate_comparison(a,b)
        self.assertEqual(validate_comparison(a,b,True)["mode"],"pixel-space")

    def test_phase_alignment_diagnostic_detects_known_shift(self):
        img=np.random.default_rng(22).integers(0,255,(128,128,3),dtype=np.uint8)
        shifted=np.roll(img,5,axis=1)
        result=estimate_translation(img,shifted)
        self.assertTrue(result["reliable"])
        self.assertEqual(abs(result["shift_x_pixels"]),5)
        self.assertEqual(result["shift_y_pixels"],0)

    def test_confirmed_pixel_space_can_continue_with_shift_warning(self):
        alignment={"mode":"pixel-space","warnings":[]}
        registration={"shift_x_pixels":3,"shift_y_pixels":0,"pixel_precision":1,"reliable":True}
        result,checked=_apply_registration_gate(alignment,registration,True)
        self.assertFalse(checked["reliable"])
        self.assertTrue(any("no correction was applied" in w for w in result["warnings"]))

    def test_unconfirmed_pixel_shift_and_geospatial_shift_still_block(self):
        registration={"shift_x_pixels":3,"shift_y_pixels":0,"pixel_precision":1,"reliable":True}
        with self.assertRaisesRegex(ValueError,"3px shift"):
            _apply_registration_gate({"mode":"pixel-space","warnings":[]},registration,False)
        with self.assertRaisesRegex(ValueError,"3px shift"):
            _apply_registration_gate({"mode":"geospatial","warnings":[]},registration,True)

    def test_extraction_keeps_minimum_area_in_pixels_and_confidence(self):
        mask=np.zeros((12,12),dtype=np.uint8); mask[2:7,3:8]=1
        prob=np.where(mask,.8,.1).astype("float32")
        feats=mask_to_features(mask,prob,Affine(0.5,0,500000,0,-0.5,1000000),min_pixels=3)
        self.assertEqual(len(feats),1); self.assertAlmostEqual(feats[0]["confidence"],.8,places=5)
        self.assertAlmostEqual(feats[0]["geometry"].area,6.25)

    def test_add_remove_unchanged_modified_and_uncertain_matching(self):
        old=[box(0,0,2,2),box(10,10,12,12),box(40,40,42,42)]
        new=[box(0,0,2,2),box(10,10,14,12),box(30,30,32,32)]
        results=compare_footprints(old,new,iou_threshold=.25,max_centroid_distance=3,area_change_threshold=.35)
        categories={r.category for r in results}
        self.assertTrue({"new","removed","unchanged","modified"}.issubset(categories))
        split_old=box(50,50,54,52)
        split_new=[box(50,50,52,52),box(52,50,54,52)]
        split=compare_footprints([split_old],split_new,iou_threshold=.2,max_centroid_distance=3)
        self.assertIn("uncertain",{r.category for r in split})

    def test_invalid_geometry_becomes_uncertain(self):
        invalid=Polygon([(0,0),(2,2),(0,2),(2,0),(0,0)])
        result=compare_footprints([invalid],[])
        self.assertEqual(result[0].category,"uncertain")
        self.assertIn("invalid_geometry",result[0].flags)

    def test_duplicate_prevention_and_empty_cases(self):
        results=compare_footprints([box(0,0,1,1),box(10,10,11,11)],[box(0,0,1,1)])
        self.assertEqual(sum(r.after_index is not None for r in results),1)
        self.assertEqual(compare_footprints([],[]),[])
        self.assertEqual(compare_footprints([box(0,0,1,1)],[],reliable=False)[0].category,"uncertain")

    def test_geojson_reprojects_projected_coordinates_to_wgs84(self):
        raw=geojson_bytes([({"id":"x"},box(500000,4649000,500010,4649010))],crs="EPSG:32643",georeferenced=True)
        doc=json.loads(raw); coords=doc["features"][0]["geometry"]["coordinates"][0]
        lon,lat=coords[0]
        self.assertTrue(-180<=lon<=180 and -90<=lat<=90)
        with self.assertRaisesRegex(ValueError,"requires the source CRS"):
            geojson_bytes([({},box(0,0,1,1))],georeferenced=True)
        with self.assertRaises(ValueError): geojson_bytes([],georeferenced=False)

    def test_csv_and_metrics(self):
        self.assertEqual(csv_bytes([]),b"")
        self.assertIn(b"category",csv_bytes([{"category":"new"}]))
        m=binary_metrics([1,1,0,0],[1,0,0,0])
        self.assertAlmostEqual(m["false_positive_rate"],1/3)
        self.assertAlmostEqual(m["precision"],.5)
        self.assertAlmostEqual(m["recall"],1.0)
        self.assertAlmostEqual(m["f1"],2/3)
        self.assertAlmostEqual(m["iou"],.5)
        self.assertEqual(m["counts"],{"tp":1,"fp":1,"fn":0,"tn":2})
        with self.assertRaises(ValueError): binary_metrics([1],[1,0])

    def test_region_change_percentage_uses_valid_common_pixels(self):
        changed=np.array([[1,1,0],[0,1,1]],dtype=bool)
        valid=np.array([[1,1,1],[1,1,0]],dtype=bool)
        result=summarize_change_area(changed,valid)
        self.assertEqual(result["changed_pixels"],3)
        self.assertEqual(result["valid_pixels"],5)
        self.assertAlmostEqual(result["percent_of_region"],60.0)
        self.assertEqual(summarize_change_area(changed,np.zeros_like(valid))["percent_of_region"],0.0)

    def test_change_label_decoding_and_dimensions(self):
        from PIL import Image
        data=io.BytesIO(); Image.fromarray(np.array([[0,255],[0,1]],dtype=np.uint8)).save(data,format="PNG")
        decoded=decode_change_label(data.getvalue(),"val_1.png",(2,2))
        np.testing.assert_array_equal(decoded,[[False,True],[False,True]])
        with self.assertRaisesRegex(ValueError,"matching label"):
            decode_change_label(data.getvalue(),"wrong.png",(3,2))
        with self.assertRaisesRegex(ValueError,"PNG"):
            decode_change_label(data.getvalue(),"label.jpg",(2,2))

    def test_levir_coordinate_sidecar_requires_exact_name_and_valid_bounds(self):
        import json
        info=geoinfo("val_1.png",geo=False,width=100,height=50)
        sidecar=json.dumps({"val_1.png":[-98.0,30.0,-97.0,29.0]}).encode()
        located=attach_levir_coordinates(info,sidecar,"LEVIR_CD_name_coords.json")
        self.assertTrue(located.georeferenced)
        self.assertEqual(located.crs,"EPSG:4326")
        self.assertIn("val_1.png",located.georeference_source)
        self.assertAlmostEqual(located.pixel_size[0],.01)
        self.assertAlmostEqual(located.pixel_size[1],-.02)
        with self.assertRaisesRegex(ValueError,"exact sample filename"):
            attach_levir_coordinates(info,json.dumps({"train_1.png":[-98,30,-97,29]}).encode(),"coords.json")
        with self.assertRaisesRegex(ValueError,"Invalid WGS84 bounds"):
            attach_levir_coordinates(info,json.dumps({"val_1.png":[-200,30,-97,29]}).encode(),"coords.json")

    def test_prediction_overlay_and_pixel_polygon_export(self):
        from shapely.geometry import box
        rgb=np.full((12,12,3),100,dtype=np.uint8)
        building=[{"geometry":box(2,2,10,10),"confidence":.9}]
        change=[{"geometry":box(4,4,8,8),"confidence":None}]
        overlay=render_change_overlay(rgb,building,change,[box(2,2,10,10)],change_opacity=.10)
        self.assertEqual(overlay.shape,rgb.shape)
        # The interior change fill is subtle (10% red), not an opaque red mask.
        self.assertGreater(int(overlay[5,5,0]),int(overlay[5,5,1]))
        self.assertLess(int(overlay[5,5,0])-100,40)
        payload=json.loads(prediction_polygons_json_bytes(building,building,change))
        self.assertEqual(payload["coordinate_space"],"image_pixel_coordinates")
        self.assertEqual([f["properties"]["kind"] for f in payload["features"]],
                         ["before","after","changed_region"])
        from PIL import Image
        encoded=mask_png_bytes(np.array([[0,1]],dtype=bool))
        with Image.open(io.BytesIO(encoded)) as im:
            np.testing.assert_array_equal(np.asarray(im.convert("L")),[[0,255]])

    def test_missing_model_is_explicit(self):
        with self.assertRaises(ModelUnavailable): BuildingDetector("missing-file.pth").load()

    def test_levir_detector_preprocessing_contract(self):
        rgb=np.zeros((256,256,3),dtype=np.uint8)
        tensor=LevirChangeDetector._preprocess(rgb)
        self.assertEqual(tensor.shape,(1,3,256,256))
        self.assertEqual(float(tensor.min()),-1.0)
        self.assertEqual(float(tensor.max()),-1.0)

    @unittest.skipUnless(Path("models/BIT_LEVIR.onnx").is_file(),"download LEVIR change weights for inference smoke test")
    def test_real_levir_change_checkpoint_inference_smoke(self):
        model=LevirChangeDetector("models/BIT_LEVIR.onnx").load()
        image=np.random.default_rng(17).integers(0,255,(300,400,3),dtype=np.uint8)
        probability,mask=model.predict_pair(image,image.copy())
        self.assertEqual(probability.shape,(300,400)); self.assertEqual(mask.shape,(300,400))
        self.assertTrue(np.isfinite(probability).all())
        self.assertTrue(((probability>=0)&(probability<=1)).all())

    @unittest.skipUnless(Path("models/best_deeplab.pth").is_file(),"download real checkpoint for integration smoke test")
    def test_real_checkpoint_inference_smoke(self):
        detector=BuildingDetector("models/best_deeplab.pth").load()
        rgb=np.random.default_rng(3).integers(0,255,(64,80,3),dtype=np.uint8)
        probs,mask=detector.predict(rgb,tile_size=64,overlap=16)
        self.assertEqual(probs.shape,(64,80)); self.assertEqual(mask.shape,(64,80))
        self.assertTrue(np.isfinite(probs).all()); self.assertTrue(((probs>=0)&(probs<=1)).all())

    @unittest.skipUnless(Path("models/best_deeplab.pth").is_file(),"download real checkpoint for pipeline smoke test")
    def test_full_pipeline_on_non_georeferenced_png_inputs(self):
        from PIL import Image
        import tempfile
        rng=np.random.default_rng(4)
        a=rng.integers(0,255,(64,64,3),dtype=np.uint8); b=a.copy()
        buf_a=io.BytesIO(); Image.fromarray(a).save(buf_a,format="PNG")
        buf_b=io.BytesIO(); Image.fromarray(b).save(buf_b,format="PNG")
        up_a=SimpleNamespace(name="a.png",getvalue=lambda:buf_a.getvalue())
        up_b=SimpleNamespace(name="b.png",getvalue=lambda:buf_b.getvalue())
        info_a=geoinfo("a.png",geo=False); info_b=geoinfo("b.png",geo=False)
        detector=BuildingDetector("models/best_deeplab.pth").load()
        change_detector=(LevirChangeDetector("models/BIT_LEVIR.onnx").load()
                         if Path("models/BIT_LEVIR.onnx").is_file() else None)
        reference=np.zeros((64,64),dtype=bool); reference[8:12,9:14]=True
        result=run_analysis(up_a,up_b,info_a,info_b,detector,.5,True,date(2020,1,1),date(2021,1,1),
            tile_size=64,tile_overlap=16,reference_change_mask=reference,
            change_detector=change_detector)
        self.assertEqual(result["alignment"]["mode"],"pixel-space")
        self.assertEqual(result["mask_before"].shape,(64,64))
        self.assertFalse(result["georeferenced"])
        self.assertEqual(len(result["records"]),len(result["features"]))
        self.assertEqual(result["building_counts"]["before"],len(result["building_features_before"]))
        self.assertEqual(result["building_counts"]["after"],len(result["building_features_after"]))
        if change_detector:
            self.assertEqual(result["change_model"],change_detector.model_name)
            self.assertIsNotNone(result["change_probability"])
        else:
            np.testing.assert_array_equal(result["change_mask"],result["mask_before"] ^ result["mask_after"])
        self.assertEqual(result["evaluation"]["valid_pixels"],64*64)
        self.assertEqual(result["change_area"]["valid_pixels"],64*64)
        self.assertAlmostEqual(result["change_area"]["percent_of_region"],
            100*result["change_area"]["changed_pixels"]/(64*64))

    @unittest.skipUnless(Path("models/best_deeplab.pth").is_file(),"download real checkpoint for GeoTIFF integration smoke test")
    def test_geotiff_pipeline_and_wgs84_export(self):
        from rasterio.io import MemoryFile
        from rasterio.transform import from_origin
        from src.image_validation import inspect_image
        from src.export_utils import summary_json_bytes
        rgb=np.random.default_rng(5).integers(1,255,(3,64,64),dtype=np.uint8)
        uploads=[]; infos=[]
        for name in ("a.tif","b.tif"):
            with MemoryFile() as mem:
                with mem.open(driver="GTiff",width=64,height=64,count=3,dtype="uint8",crs="EPSG:32643",
                              transform=from_origin(500000,3200000,0.5,0.5)) as ds: ds.write(rgb)
                raw=mem.read()
            uploaded=SimpleNamespace(name=name,getvalue=lambda d=raw:d)
            uploads.append(uploaded); infos.append(inspect_image(io.BytesIO(raw),name))
        detector=BuildingDetector("models/best_deeplab.pth").load()
        result=run_analysis(uploads[0],uploads[1],infos[0],infos[1],detector,.5,False,
            date(2020,1,1),date(2021,1,1),tile_size=64,tile_overlap=16)
        self.assertTrue(result["georeferenced"])
        self.assertEqual(result["alignment"]["mode"],"geospatial")
        encoded=geojson_bytes(result["features"],crs="EPSG:4326",georeferenced=True)
        self.assertEqual(json.loads(encoded)["type"],"FeatureCollection")
        self.assertTrue(summary_json_bytes({"crs":result["crs"]}))

if __name__=="__main__": unittest.main()
