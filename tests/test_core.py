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
from src.analysis_pipeline import validate_comparison,run_analysis
from src.building_detector import BuildingDetector,ModelUnavailable
from src.change_detection import compare_footprints
from src.export_utils import csv_bytes,geojson_bytes
from src.footprint_extraction import mask_to_features
from src.metrics import binary_metrics

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
        with self.assertRaises(ValueError): binary_metrics([1],[1,0])

    def test_missing_model_is_explicit(self):
        with self.assertRaises(ModelUnavailable): BuildingDetector("missing-file.pth").load()

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
        result=run_analysis(up_a,up_b,info_a,info_b,detector,.5,True,date(2020,1,1),date(2021,1,1),tile_size=64,tile_overlap=16)
        self.assertEqual(result["alignment"]["mode"],"pixel-space")
        self.assertEqual(result["mask_before"].shape,(64,64))
        self.assertFalse(result["georeferenced"])
        self.assertEqual(len(result["records"]),len(result["features"]))

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
