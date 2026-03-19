import multiprocessing as mp
import sys
if __name__ == "__main__":
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

import os
import shutil
from flask import Flask, request, jsonify, send_from_directory, send_file
from flask_cors import CORS
import time
import random
from PIL import Image, ImageDraw, ImageOps
import io
import math
from threading import Thread
import requests
import numpy as np
import json
import re
import sys
import logging
import multiprocessing
from multiprocessing import Queue
import importlib.util
import matplotlib.pyplot as plt
import cv2
import osgeo.gdal as gdal
# --- Disable Flask's default request logger ---
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

# --- Add path for custom planner ---
try:
    planners_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../modules/planners'))
    if os.path.isdir(planners_path):
        sys.path.append(planners_path)
        import astar_bind
        HAS_ASTAR = True
        print("SUCCESS: A* planner library found and imported.")
    else:
        HAS_ASTAR = False
        print(f"WARNING: A* planner directory not found at {planners_path}. Planning will be disabled.")
except ImportError as e:
    HAS_ASTAR = False
    print(f"WARNING: Failed to import astar_bind. Planning will be disabled. Error: {e}")


# --- Check for Optional GIS Libraries ---
try:
    import rasterio
    from rasterio.transform import from_bounds
    from rasterio.merge import merge
    HAS_RASTERIO = True
    print("SUCCESS: rasterio library found. Real rasterization is enabled.")
except ImportError:
    HAS_RASTERIO = False
    print("WARNING: rasterio library not found. Falling back to dummy rasterization.")

try:
    import pynvml
    HAS_PYNVML = True
except ImportError:
    HAS_PYNVML = False

from overseec.overseec_config import AllConfig
from overseec.modules.llm.vllm_client import overseec_query_llm
from overseec.OVerSeeC import OVerSeeC

# --- Live Console Log Capture ---
class ConsoleLog:
    def __init__(self):
        self.logs = []
        self.original_stdout = sys.stdout
        self.original_stderr = sys.stderr
        sys.stdout = self
        sys.stderr = self

    def write(self, text):
        self.original_stdout.write(text)
        self.logs.append(text)
        if len(self.logs) > 200:
            self.logs = self.logs[-200:]

    def flush(self):
        self.original_stdout.flush()
        self.original_stderr.flush()

    def get_logs(self):
        return "".join(self.logs)

console_log = ConsoleLog()

# --- Multiprocessing and Task Management ---
pipeline_processes = {}
# Use spawn context for CUDA compatibility
mp_ctx = multiprocessing.get_context('spawn')
log_queue = mp_ctx.Queue()

def log_listener(queue, logger):
    while True:
        try:
            message = queue.get()
            if message is None: break
            logger.write(message)
        except (IOError, EOFError):
            break

listener_thread = Thread(target=log_listener, args=(log_queue, console_log))
listener_thread.daemon = True
listener_thread.start()

def _get_tiff_folder_name(tiff_filename):
    """Get the results folder name for a given tiff filename.
    Only strips .tif/.tiff extensions, preserves dots in names like 'pickle-north-0.2'."""
    import re
    name = os.path.basename(tiff_filename)
    # Only strip .tif or .tiff extension (case-insensitive)
    return re.sub(r'\.tiff?$', '', name, flags=re.IGNORECASE)

def _resolve_tiff_path(tiff_identifier):
    """Resolve a tiff identifier to the actual file path.
    Checks results/<name>/original.tif first, then uploads/tiffs/<name>."""
    # Check new location first
    new_path = os.path.join(RESULTS_FOLDER, tiff_identifier, "original.tif")
    if os.path.exists(new_path):
        return new_path
    # Fallback to old location (with original extension)
    for ext in ['.tif', '.tiff']:
        old_path = os.path.join(TIFF_FOLDER, tiff_identifier + ext)
        if os.path.exists(old_path):
            return old_path
    # Try exact filename in old location
    old_path = os.path.join(TIFF_FOLDER, tiff_identifier)
    if os.path.exists(old_path):
        return old_path
    return None

def _ensure_tiff_in_results(tiff_filename):
    """Ensure the TIFF is copied to results/<name>/original.tif.
    Returns the tiff folder name."""
    folder_name = _get_tiff_folder_name(tiff_filename)
    tiff_results_dir = os.path.join(RESULTS_FOLDER, folder_name)
    os.makedirs(tiff_results_dir, exist_ok=True)
    # Standardized subfolders
    os.makedirs(os.path.join(tiff_results_dir, "mask"), exist_ok=True)
    os.makedirs(os.path.join(tiff_results_dir, "costmap"), exist_ok=True)
    dest_path = os.path.join(tiff_results_dir, "original.tif")
    if not os.path.exists(dest_path):
        # Try to find the source TIFF
        src_path = os.path.join(TIFF_FOLDER, tiff_filename)
        if os.path.exists(src_path):
            shutil.copy2(src_path, dest_path)
            print(f"Copied TIFF to {dest_path}")
        else:
            print(f"WARNING: Source TIFF not found at {src_path}")
    return folder_name

def pipeline_worker(log_q, task_id, tiff_filename, classes_data, params):
    class QueueLogger:
        def __init__(self, queue): self.queue = queue
        def write(self, text): self.queue.put(text)
        def flush(self): pass

    sys.stdout = QueueLogger(log_q)
    sys.stderr = QueueLogger(log_q)

    try:
        # Resolve the TIFF path
        sat_img_path = _resolve_tiff_path(tiff_filename)
        if sat_img_path is None:
            sat_img_path = os.path.join(TIFF_FOLDER, tiff_filename)
        
        # Always ensure the TIFF is stored in results/<name>/original.tif
        tiff_folder_name = _ensure_tiff_in_results(tiff_filename)

        clipseg_classes, clipseg_classes_semseg_knobs = {}, {}
        for class_obj in classes_data:
            color_str = class_obj.get("color", "rgb(0,0,0)")
            rgb_values = tuple(map(int, re.findall(r'\d+', color_str)))
            clipseg_classes[class_obj["name"]] = rgb_values
            clipseg_classes_semseg_knobs[class_obj["name"]] = float(class_obj["threshold"])

        final_config = AllConfig(
            model_ckpt="xyz", model_name=params.get("model_name"),
            mask_refiner_name=params.get("mask_refiner_name"), classes=clipseg_classes,
            classes_semseg_knobs=clipseg_classes_semseg_knobs,
            semseg_tile_size=(int(params.get("semseg_tile_size")), int(params.get("semseg_tile_size"))),
            semseg_stride=int(params.get("semseg_stride")), semseg_tile_combine_method=params.get("semseg_combine_method"),
            mask_refiner_tile_size=(int(params.get("refiner_tile_size")), int(params.get("refiner_tile_size"))),
            mask_refiner_stride=int(params.get("refiner_stride")), mask_refiner_tile_combine_method=params.get("refiner_combine_method"),
            use_negative_points=True, sam_model=params.get("sam_model"), sam_device=params.get("sam_device"),
            cmap_device=params.get("cmap_device"), semseg_device=params.get("semseg_device"),
        )
        final_config.semseg_config.num_workers = 0
        final_config.semseg_config.pin_memory = False
        final_config.mask_refiner_config.num_workers = 0
        final_config.mask_refiner_config.pin_memory = False
        final_config.reset()
        print("\n--- CONFIGURATION SET FOR PIPELINE RUN ---"); print(final_config); print("------------------------------------------\n")
        
        print(f"Running OVerSeeC model on: {sat_img_path}")
        overseec_sat_2_mask = OVerSeeC(config=final_config)
        (clipseg_sigmoid_mask_refiner_logits, _, _, clipseg_sigmoid_semseg_logits, _) = overseec_sat_2_mask(sat_img_path, None)
        
        # Save to mask/temp_latest inside the tiff's results folder
        temp_dir = os.path.join(RESULTS_FOLDER, tiff_folder_name, "mask", "temp_latest")
        # Clean up old temp_latest if it exists
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        semantic_mask_dir = os.path.join(temp_dir, "semantic")
        refined_mask_dir = os.path.join(temp_dir, "refined")
        os.makedirs(semantic_mask_dir, exist_ok=True)
        os.makedirs(refined_mask_dir, exist_ok=True)

        # Build URL path relative to results/
        url_prefix = f"{tiff_folder_name}/mask/temp_latest"

        semantic_masks_urls, refined_masks_urls = {}, {}
        class_names = list(clipseg_classes.keys())
        
        for i, class_name in enumerate(class_names):
            class_name_safe = class_name.replace(" ", "_")
            
            # --- Save PNG and NPY for Semantic Mask ---
            sem_path = os.path.join(semantic_mask_dir, f"{class_name_safe}.png")
            sem_npy_path = os.path.join(semantic_mask_dir, f"{class_name_safe}.npy")
            sem_tensor_slice = clipseg_sigmoid_semseg_logits[i]
            save_tensor_as_image(sem_tensor_slice, sem_path)
            sem_numpy_array = sem_tensor_slice.cpu().numpy() if hasattr(sem_tensor_slice, 'cpu') else np.array(sem_tensor_slice)
            np.save(sem_npy_path, sem_numpy_array)
            semantic_masks_urls[class_name] = f"/results/{url_prefix}/semantic/{class_name_safe}.png"

            # --- Save PNG and NPY for Refined Mask ---
            ref_path = os.path.join(refined_mask_dir, f"{class_name_safe}.png")
            ref_npy_path = os.path.join(refined_mask_dir, f"{class_name_safe}.npy")
            ref_tensor_slice = clipseg_sigmoid_mask_refiner_logits[i]
            save_tensor_as_image(ref_tensor_slice, ref_path)
            ref_numpy_array = ref_tensor_slice.cpu().numpy() if hasattr(ref_tensor_slice, 'cpu') else np.array(ref_tensor_slice)
            np.save(ref_npy_path, ref_numpy_array)
            refined_masks_urls[class_name] = f"/results/{url_prefix}/refined/{class_name_safe}.png"

        print("--- Model processing complete, masks saved to mask/temp_latest. ---")
        results_data = {
            "message": "Pipeline completed successfully!",
            "semantic_masks": semantic_masks_urls,
            "refined_masks": refined_masks_urls,
            "tiff_folder": tiff_folder_name
        }
        with open(os.path.join(RESULTS_FOLDER, f"{task_id}.json"), 'w') as f:
            json.dump(results_data, f)
    except Exception as e:
        print(f"!!! PIPELINE ERROR for task {task_id}: {e}")
        import traceback
        traceback.print_exc()
        with open(os.path.join(RESULTS_FOLDER, f"{task_id}.error"), 'w') as f:
            f.write(str(e))
    finally:
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            # Delete model to free memory
            if 'overseec_sat_2_mask' in locals():
                del overseec_sat_2_mask
        except:
            pass

# --- Get the absolute path of the script's directory ---
_basedir = os.path.abspath(os.path.dirname(__file__))

# --- Configuration ---
UPLOAD_FOLDER = os.path.join(_basedir, "uploads")
TIFF_FOLDER = os.path.join(UPLOAD_FOLDER, "tiffs")
PREVIEWS_FOLDER = os.path.join(_basedir, "previews")
TEMP_FOLDER = os.path.join(_basedir, "temp_tiles")
RESULTS_FOLDER = os.path.join(_basedir, "results")
COSTMAP_FUNCTIONS_FOLDER = os.path.join(_basedir, "costmap_functions")
TEMP_PARAMS_FILE = os.path.join(_basedir, "temp_parameters.json")

# Default parameters
DEFAULT_PARAMS = {
    "areal_threshold": "0.8",
    "linear_threshold": "0.4",
    "model_name": "clipseg",
    "mask_refiner_name": "samrefiner",
    "sam_model": "vit_h",
    "cmap_device": "cuda:0",
    "sam_device": "cuda:0",
    "semseg_device": "cuda:0",
    "semseg_tile_size": "512",
    "semseg_stride": "256",
    "refiner_tile_size": "512",
    "refiner_stride": "256",
    "semseg_combine_method": "max",
    "refiner_combine_method": "max"
}

# --- Flask App Initialization ---
app = Flask(__name__)
CORS(app)

# --- Ensure Directories Exist ---
os.makedirs(TIFF_FOLDER, exist_ok=True)
os.makedirs(PREVIEWS_FOLDER, exist_ok=True)
os.makedirs(TEMP_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)
os.makedirs(COSTMAP_FUNCTIONS_FOLDER, exist_ok=True)

if not os.path.exists(os.path.join(TIFF_FOLDER, "dummy_image.tiff")):
    with open(os.path.join(TIFF_FOLDER, "dummy_image.tiff"), "w") as f: f.write("This is a dummy TIFF file.")

default_costmap_path = os.path.join(COSTMAP_FUNCTIONS_FOLDER, "default_costmap.py")
if not os.path.exists(default_costmap_path):
    with open(default_costmap_path, "w") as f:
        f.write("# Default costmap generation logic\n\n")
        f.write("import numpy as np\n\n")
        f.write("def generate_costmap(mask_dict):\n")
        f.write("    # Example: create a simple costmap\n")
        f.write("    if not mask_dict:\n")
        f.write("        return np.zeros((512, 512)) # Return a default shape if no masks\n")
        f.write("    first_mask = next(iter(mask_dict.values()))\n")
        f.write("    costmap = np.ones_like(first_mask, dtype=np.float32)\n")
        f.write("    if 'road' in mask_dict:\n")
        f.write("        costmap[mask_dict['road'] == 1] = 0.1\n")
        f.write("    if 'water' in mask_dict:\n")
        f.write("        costmap[mask_dict['water'] == 1] = 0.9\n")
        f.write("    return costmap\n")

rasterization_tasks = {}

# --- Helper Functions ---
def save_tensor_as_image(tensor_slice, path):
    try:
        if hasattr(tensor_slice, 'cpu'): tensor_slice = tensor_slice.cpu()
        if hasattr(tensor_slice, 'numpy'): image_array = tensor_slice.numpy()
        else: image_array = np.array(tensor_slice)
        image_array = (image_array * 255).astype(np.uint8)
        Image.fromarray(image_array).save(path)
    except Exception as e:
        print(f"Error saving tensor as image: {e}")
        create_colorful_placeholder(path)

def create_colorful_placeholder(path, size=(512, 512)):
    img = Image.new('RGB', size, color='black')
    draw = ImageDraw.Draw(img)
    for i in range(0, size[0], 20):
        draw.line((i, 0, i, size[1]), fill=(random.randint(50, 255), random.randint(50, 255), random.randint(50, 255)), width=2)
        draw.line((0, i, size[0], i), fill=(random.randint(50, 255), random.randint(50, 255), random.randint(50, 255)), width=2)
    img.save(path, 'PNG')

# --- Real Rasterization Logic ---
def deg2num(lat_deg, lon_deg, zoom):
    lat_rad = math.radians(lat_deg)
    n = 2.0 ** zoom
    xtile = int((lon_deg + 180.0) / 360.0 * n)
    ytile = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
    return (xtile, ytile)

def num2deg(xtile, ytile, zoom):
    n = 2.0 ** zoom
    lon_deg = xtile / n * 360.0 - 180.0
    lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * ytile / n)))
    lat_deg = math.degrees(lat_rad)
    return (lat_deg, lon_deg)
  
def calculate_zoom_from_resolution(resolution_mpp, latitude):
    C = 40075017
    zoom = math.log2( (C * math.cos(math.radians(latitude))) / (resolution_mpp * 256) )
    return max(1, min(19, int(round(zoom))))

def perform_rasterization(task_id, bounds_str, filename, resolution, tile_size):
    try:
        if not HAS_RASTERIO: raise ImportError("rasterio is not installed.")
        if not filename.lower().endswith(('.tiff', '.tif')): filename += ".tiff"
        rasterization_tasks[task_id] = {"progress": 5, "status": "Parsing coordinates..."}
        b = [float(c) for c in bounds_str.split(',')]
        west, south, east, north = b[0], b[1], b[2], b[3]
        center_lat = (north + south) / 2
        zoom = calculate_zoom_from_resolution(resolution, center_lat)
        print(f"Calculated Zoom Level: {zoom} for resolution {resolution} m/px")
        nw_xtile, nw_ytile = deg2num(north, west, zoom)
        se_xtile, se_ytile = deg2num(south, east, zoom)
        tile_count = (se_xtile - nw_xtile + 1) * (se_ytile - nw_ytile + 1)
        if tile_count > 500: raise ValueError(f"Area too large ({tile_count} tiles). Select a smaller area or increase Map Units per Pixel.")
        rasterization_tasks[task_id] = {"progress": 10, "status": f"Downloading {tile_count} tiles at zoom {zoom}..."}
        temp_geotiffs, task_temp_folder, downloaded_count = [], os.path.join(TEMP_FOLDER, task_id), 0
        os.makedirs(task_temp_folder, exist_ok=True)
        for x in range(nw_xtile, se_xtile + 1):
            for y in range(nw_ytile, se_ytile + 1):
                res = requests.get(f"https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={zoom}", stream=True, headers={'User-Agent': 'Mozilla/5.0'})
                if res.status_code == 200:
                    tile_image = Image.open(io.BytesIO(res.content)).convert("RGB")
                    tile_width, tile_height = tile_image.size
                    top_left_lat, top_left_lon = num2deg(x, y, zoom)
                    bottom_right_lat, bottom_right_lon = num2deg(x + 1, y + 1, zoom)
                    transform = from_bounds(west=top_left_lon, south=bottom_right_lat, east=bottom_right_lon, north=top_left_lat, width=tile_width, height=tile_height)
                    temp_geotiff_path = os.path.join(task_temp_folder, f"{x}_{y}.tif")
                    profile = {'driver': 'GTiff', 'height': tile_height, 'width': tile_width, 'count': 3, 'dtype': 'uint8', 'crs': 'EPSG:4326', 'transform': transform}
                    with rasterio.open(temp_geotiff_path, 'w', **profile) as dst:
                        r, g, b = tile_image.split(); dst.write(np.array(r), 1); dst.write(np.array(g), 2); dst.write(np.array(b), 3)
                    temp_geotiffs.append(temp_geotiff_path)
                downloaded_count += 1
                rasterization_tasks[task_id] = {"progress": 10 + int(60 * (downloaded_count / tile_count)), "status": f"Processing tile {downloaded_count}/{tile_count}"}
        rasterization_tasks[task_id] = {"progress": 75, "status": "Stitching GeoTIFFs..."}
        sources_to_merge = [rasterio.open(path) for path in temp_geotiffs]
        mosaic, out_trans = merge(sources_to_merge)
        for src in sources_to_merge: src.close()
        out_meta = sources_to_merge[0].meta.copy()
        out_meta.update({"driver": "GTiff", "height": mosaic.shape[1], "width": mosaic.shape[2], "transform": out_trans, "crs": "EPSG:4326"})
        rasterization_tasks[task_id] = {"progress": 90, "status": "Saving final GeoTIFF..."}
        # Save to uploads/tiffs as before (for backward compat)
        with rasterio.open(os.path.join(TIFF_FOLDER, filename), "w", **out_meta) as dest: dest.write(mosaic)
        # Also save to organized results structure
        _ensure_tiff_in_results(filename)
        shutil.rmtree(task_temp_folder)
        rasterization_tasks[task_id] = {"progress": 100, "status": "Completed", "filename": filename}
    except Exception as e:
        print(f"Rasterization error: {e}")
        rasterization_tasks[task_id] = {"progress": 100, "status": f"Error: {e}", "error": True}

def simulate_rasterization(task_id, filename, **kwargs):
    if not filename.lower().endswith(('.tiff', '.tif')): filename += ".tiff"
    time.sleep(1); rasterization_tasks[task_id] = {"progress": 25, "status": "Simulating: Fetching tiles..."}
    time.sleep(1); rasterization_tasks[task_id] = {"progress": 60, "status": "Simulating: Stitching..."}
    time.sleep(1); rasterization_tasks[task_id] = {"progress": 90, "status": "Simulating: Saving..."}
    create_colorful_placeholder(os.path.join(TIFF_FOLDER, filename))
    time.sleep(1); rasterization_tasks[task_id] = {"progress": 100, "status": "Completed (Simulated)", "filename": filename}

# --- API Endpoints ---
@app.route("/")
def index(): return send_from_directory(_basedir, "index.html")
@app.route('/script.js')
def script(): return send_from_directory(_basedir, 'script.js')
@app.route('/previews/<path:filename>')
def serve_preview(filename): return send_from_directory(PREVIEWS_FOLDER, filename)
@app.route('/results/<path:path>')
def serve_results(path): return send_from_directory(RESULTS_FOLDER, path)
    
@app.route("/api/get-console-output", methods=["GET"])
def get_console_output():
    return jsonify({"logs": console_log.get_logs()})

@app.route("/api/get-default-config", methods=["GET"])
def get_default_config():
    clipseg_classes = {
        "road": [128, 64, 128], "trail or footway": [244, 35, 232], "tree": [0, 128, 0],
        "grass": [0, 128, 128], "building": [0, 0, 128], "water": [0, 0, 255],
    }
    clipseg_classes_semseg_knobs = {
        "road": 0.4, "trail or footway": 0.4, "tree": 0.8, "grass": 0.8, "building": 0.8, "water": 0.95,
    }
    default_config = []
    for name, color in clipseg_classes.items():
        threshold = clipseg_classes_semseg_knobs.get(name, 0.5)
        class_type = "Linear" if threshold <= 0.4 else "Areal"
        default_config.append({
            "name": name, "type": class_type, "threshold": threshold,
            "color": f"rgb({color[0]}, {color[1]}, {color[2]})"
        })
    return jsonify({"classes": default_config})

@app.route("/api/get-gpu-status", methods=["GET"])
def get_gpu_status():
    if not HAS_PYNVML: return jsonify({"gpus": [{"id": i, "usage": 0, "memory_used": 0, "memory_total": 0} for i in range(8)], "source": "mock"})
    try:
        pynvml.nvmlInit()
        gpus = []
        for i in range(pynvml.nvmlDeviceGetCount()):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            util, mem = pynvml.nvmlDeviceGetUtilizationRates(handle), pynvml.nvmlDeviceGetMemoryInfo(handle)
            gpus.append({ "id": i, "usage": util.gpu, "memory_used": mem.used // (1024**2), "memory_total": mem.total // (1024**2) })
        pynvml.nvmlShutdown()
        return jsonify({"gpus": gpus, "source": "real"})
    except pynvml.NVMLError as e: 
        print(f"pynvml error: {e}")
        return jsonify({"gpus": [{"id": i, "usage": 0, "memory_used": 0, "memory_total": 0} for i in range(8)], "source": "mock-error"})

@app.route("/api/rasterize-area", methods=["POST"])
def rasterize_area_start():
    data, task_id = request.json, str(random.randint(1000, 9999))
    args = {
        "task_id": task_id, "filename": data.get("filename"), "bounds_str": data.get("bounds"),
        "resolution": float(data.get("resolution", 1.0)), "tile_size": int(data.get("tileSize", 512))
    }
    target = perform_rasterization if HAS_RASTERIO else simulate_rasterization
    kwargs = args if HAS_RASTERIO else {"task_id": task_id, "filename": args["filename"]}
    thread = Thread(target=target, kwargs=kwargs)
    thread.start()
    return jsonify({"task_id": task_id})

@app.route("/api/rasterize-status/<task_id>", methods=["GET"])
def rasterize_status(task_id): return jsonify(rasterization_tasks.get(task_id, {}))

@app.route("/api/generate-preview", methods=["POST"])
def generate_preview():
    data = request.json
    filename = data.get("filename")
    if not filename: return jsonify({"error": "No filename provided."}), 400
    # Try to resolve the tiff path from the new structure first
    tiff_path = _resolve_tiff_path(filename)
    if tiff_path is None:
        tiff_path = os.path.join(TIFF_FOLDER, filename)
    if not os.path.exists(tiff_path): return jsonify({"error": "File not found."}), 404
    try:
        # Ensure results/<tiff>/ exists so we can persist a stable preview.png there too
        tiff_folder_name = _ensure_tiff_in_results(filename)
        preview_filename = f"{_get_tiff_folder_name(filename)}_{int(time.time())}.png"
        preview_path = os.path.join(PREVIEWS_FOLDER, preview_filename)
        with Image.open(tiff_path) as img:
            img.thumbnail((800, 800)); img.convert("RGB").save(preview_path, "PNG")
            # Also save a stable preview for this TIFF folder (used for overlays)
            results_preview_path = os.path.join(RESULTS_FOLDER, tiff_folder_name, "preview.png")
            img.convert("RGB").save(results_preview_path, "PNG")
        return jsonify({"preview_url": f"/previews/{preview_filename}"})
    except Exception as e: return jsonify({"error": f"Failed to generate preview: {e}"}), 500

@app.route("/api/get-tiff-files", methods=["GET"])
def get_tiff_files():
    try:
        files = set()
        # Scan results/ for folders containing original.tif (new structure)
        if os.path.exists(RESULTS_FOLDER):
            for name in os.listdir(RESULTS_FOLDER):
                result_dir = os.path.join(RESULTS_FOLDER, name)
                if os.path.isdir(result_dir) and os.path.exists(os.path.join(result_dir, "original.tif")):
                    files.add(name)
        # Also scan uploads/tiffs for backward compatibility
        if os.path.exists(TIFF_FOLDER):
            for f in os.listdir(TIFF_FOLDER):
                if f.lower().endswith(('.tiff', '.tif')):
                    files.add(f)
        return jsonify({"files": sorted(list(files))})
    except Exception as e: return jsonify({"error": str(e)}), 500

@app.route("/api/process-prompt", methods=["POST"])
def process_prompt():
    data = request.json
    prompt = data.get("prompt", "")
    if not prompt: return jsonify({"error": "Prompt cannot be empty."}), 400
    
    generated_path = os.path.join(COSTMAP_FUNCTIONS_FOLDER, "generated_costmap.py")
    try:
        classes_from_llm = overseec_query_llm(prompt, generated_path)
        classes_list = []
        for name, details in classes_from_llm.items():
            threshold, color_rgb = details
            class_type = "Linear" if threshold <= 0.4 else "Areal"
            classes_list.append({
                "name": name, "type": class_type, "threshold": threshold,
                "color": f"rgb({color_rgb[0]}, {color_rgb[1]}, {color_rgb[2]})"
            })
        return jsonify({"classes": classes_list})
    except Exception as e:
        print(f"An error occurred in overseec_query_llm: {e}")
        return jsonify({"error": "Failed to get classes from LLM."}), 500

# --- Pipeline Endpoints ---
@app.route("/api/run-pipeline", methods=["POST"])
def run_pipeline():
    data = request.json
    if not data.get("tiff_file") or not data.get("classes"):
        return jsonify({"error": "Missing TIFF file or classes."}), 400
    
    task_id = f"pipeline_{int(time.time())}_{random.randint(1000, 9999)}"
    
    worker_args = (
        log_queue,
        task_id,
        data.get("tiff_file"),
        data.get("classes", []),
        data.get("params", {})
    )
    
    process = mp_ctx.Process(target=pipeline_worker, args=worker_args)
    process.start()
    pipeline_processes[task_id] = process
    
    return jsonify({"message": "Pipeline started.", "task_id": task_id})

@app.route("/api/pipeline-status/<task_id>", methods=["GET"])
def pipeline_status(task_id):
    result_file = os.path.join(RESULTS_FOLDER, f"{task_id}.json")
    error_file = os.path.join(RESULTS_FOLDER, f"{task_id}.error")

    if os.path.exists(result_file):
        with open(result_file, 'r') as f:
            results = json.load(f)
        if task_id in pipeline_processes:
            del pipeline_processes[task_id]
        try:
            os.remove(result_file)
        except FileNotFoundError:
            pass
        return jsonify({"status": "completed", "results": results})
    
    if os.path.exists(error_file):
        with open(error_file, 'r') as f:
            error_message = f.read()
        if task_id in pipeline_processes:
            del pipeline_processes[task_id]
        try:
            os.remove(error_file)
        except FileNotFoundError:
            pass
        return jsonify({"status": "error", "message": error_message})

    process = pipeline_processes.get(task_id)
    if process and process.is_alive():
        return jsonify({"status": "running"})

    if task_id in pipeline_processes:
        del pipeline_processes[task_id]
    return jsonify({"status": "canceled"})

@app.route("/api/cancel-pipeline/<task_id>", methods=["POST"])
def cancel_pipeline(task_id):
    process = pipeline_processes.get(task_id)
    if process and process.is_alive():
        print(f"--- Terminating process for task {task_id} ---")
        process.terminate()
        process.join()
        if task_id in pipeline_processes:
            del pipeline_processes[task_id]
        return jsonify({"message": f"Pipeline task {task_id} canceled."})
    return jsonify({"error": "Task not found or already finished."}), 404

# --- Saved Goals Endpoints ---
@app.route("/api/goals", methods=["GET"])
def get_goals():
    tiff_name = request.args.get("tiff_name")
    if not tiff_name:
         return jsonify({"error": "Missing tiff_name parameter."}), 400
    
    goals_file = os.path.join(RESULTS_FOLDER, tiff_name, "goals.json")
    if not os.path.exists(goals_file):
        return jsonify({"goals": []})
        
    try:
        with open(goals_file, 'r') as f:
            goals = json.load(f)
        return jsonify({"goals": goals})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/goals", methods=["POST"])
def save_goal():
    data = request.json
    tiff_name = data.get("tiff_name")
    start = data.get("start")
    end = data.get("end")
    
    if not tiff_name or not start or not end:
        return jsonify({"error": "Missing parameters."}), 400
        
    goals_dir = os.path.join(RESULTS_FOLDER, tiff_name)
    if not os.path.exists(goals_dir):
        return jsonify({"error": "TIFF results directory not found."}), 404
        
    goals_file = os.path.join(goals_dir, "goals.json")
    goals = []
    if os.path.exists(goals_file):
        try:
            with open(goals_file, 'r') as f:
                goals = json.load(f)
        except:
            pass
            
    new_id = 1
    if goals:
        new_id = max(g.get("id", 0) for g in goals) + 1
        
    new_goal = {
        "id": new_id,
        "name": f"Goal {new_id}",
        "start": start,
        "end": end,
        "timestamp": time.time()
    }
    
    goals.append(new_goal)
    
    with open(goals_file, 'w') as f:
        json.dump(goals, f, indent=2)
        
    return jsonify({"message": "Goal saved.", "goal": new_goal})

@app.route("/api/goals/<int:goal_id>", methods=["PUT"])
def rename_goal(goal_id):
    data = request.json
    tiff_name = data.get("tiff_name")
    new_name = data.get("name")
    
    if not tiff_name or not new_name:
        return jsonify({"error": "Missing parameters."}), 400

    goals_file = os.path.join(RESULTS_FOLDER, tiff_name, "goals.json")
    if not os.path.exists(goals_file):
        return jsonify({"error": "Goals file not found."}), 404

    try:
        with open(goals_file, 'r') as f:
            goals = json.load(f)
        
        goal_found = False
        target_goal = None
        for g in goals:
            if g.get("id") == goal_id:
                g["name"] = new_name
                target_goal = g
                goal_found = True
                break
        
        if not goal_found:
             return jsonify({"error": "Goal not found."}), 404
             
        with open(goals_file, 'w') as f:
            json.dump(goals, f, indent=2)
            
        return jsonify({"message": "Goal renamed.", "goal": target_goal})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/goals/<int:goal_id>", methods=["DELETE"])
def delete_goal(goal_id):
    tiff_name = request.args.get("tiff_name")
    if not tiff_name:
        return jsonify({"error": "Missing tiff_name parameter."}), 400

    goals_file = os.path.join(RESULTS_FOLDER, tiff_name, "goals.json")
    if not os.path.exists(goals_file):
        return jsonify({"error": "Goals file not found."}), 404

    try:
        with open(goals_file, 'r') as f:
            goals = json.load(f)
        
        goals = [g for g in goals if g.get("id") != goal_id]
        
        with open(goals_file, 'w') as f:
            json.dump(goals, f, indent=2)
            
        return jsonify({"message": "Goal deleted.", "goals": goals})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# --- Costmap Function Endpoints ---
@app.route("/api/get-costmap-functions", methods=["GET"])
def get_costmap_functions():
    default_path = os.path.join(COSTMAP_FUNCTIONS_FOLDER, "default_costmap.py")
    generated_path = os.path.join(COSTMAP_FUNCTIONS_FOLDER, "generated_costmap.py")
    
    try:
        with open(default_path, 'r') as f:
            default_code = f.read()
    except FileNotFoundError:
        default_code = "# Default costmap function not found."

    try:
        with open(generated_path, 'r') as f:
            generated_code = f.read()
    except FileNotFoundError:
        generated_code = "# No generated costmap function yet.\n# Run the pipeline with a prompt to generate one."

    return jsonify({"default": default_code, "generated": generated_code})

@app.route("/api/save-costmap-function", methods=["POST"])
def save_costmap_function():
    data = request.json
    code = data.get("code")
    if code is None:
        return jsonify({"error": "No code provided."}), 400
    
    try:
        path = os.path.join(COSTMAP_FUNCTIONS_FOLDER, "generated_costmap.py")
        with open(path, 'w') as f:
            f.write(code)
        return jsonify({"message": "Generated costmap function saved successfully."})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/restore-costmap-function", methods=["POST"])
def restore_costmap_function():
    """Copies the default costmap file to the generated costmap file."""
    try:
        default_path = os.path.join(COSTMAP_FUNCTIONS_FOLDER, "default_costmap.py")
        generated_path = os.path.join(COSTMAP_FUNCTIONS_FOLDER, "generated_costmap.py")
        
        if not os.path.exists(default_path):
             return jsonify({"error": "Default costmap file not found on server."}), 404

        shutil.copyfile(default_path, generated_path)
        
        return jsonify({"message": "Costmap function restored to default successfully."})
    except Exception as e:
        print(f"!!! COSTMAP RESTORE ERROR: {e}")
        return jsonify({"error": str(e)}), 500
    
@app.route("/api/generate-costmap", methods=["POST"])
def generate_final_costmap():
    data = request.json
    mask_urls = data.get('refined_masks')
    if not mask_urls:
        return jsonify({"error": "No mask URLs provided."}), 400

    try:
        mask_dict = {}

        for class_name, url in mask_urls.items():
            mask_path = os.path.join(_basedir, url.lstrip('/'))
            
            if os.path.exists(mask_path):
                with Image.open(mask_path).convert('L') as img:
                    mask_array = np.array(img)
                    # binary_mask = (mask_array > binary_threshold).astype(np.uint8)
                    # mask_dict[class_name] = binary_mask
                    
                    mask_dict[class_name] = mask_array / 255.0  # Normalize to [0, 1] range
            else:
                print(f"Warning: Mask file not found at {mask_path}")

        if not mask_dict:
             return jsonify({"error": "Could not load any masks to generate costmap."}), 500

        costmap_func_path = os.path.join(COSTMAP_FUNCTIONS_FOLDER, "generated_costmap.py")
        if not os.path.exists(costmap_func_path):
            costmap_func_path = os.path.join(COSTMAP_FUNCTIONS_FOLDER, "default_costmap.py")

        spec = importlib.util.spec_from_file_location("costmap_module", costmap_func_path)
        if not spec or not spec.loader:
            return jsonify({"error": f"Could not load costmap function from {costmap_func_path}"}), 500
        
        costmap_module = importlib.util.module_from_spec(spec)
        sys.modules['costmap_module'] = costmap_module
        spec.loader.exec_module(costmap_module)
        
        tiff_folder_name = data.get('tiff_folder', '')
        device = data.get('device', 'cpu')
        
        costmap_raw = costmap_module.generate_costmap(
            mask_dict, 
            t_dict=data.get('t_dict', {"t_l": 0.4, "t_a": 0.6}), 
            device=device
        )

        min_val, max_val = costmap_raw.min(), costmap_raw.max()
        if max_val == min_val:
            normalized_costmap = np.zeros_like(costmap_raw, dtype=np.float32)
        else:
            normalized_costmap = (costmap_raw - min_val) / (max_val - min_val)
        
        cmap = plt.get_cmap('hot')
        heatmap_rgb = (cmap(normalized_costmap)[:, :, :3] * 255).astype(np.uint8)
        
        # Save a pure heatmap image (no RGB blended in).
        # Client-side UI can overlay it over the RGB preview with an opacity slider.
        costmap_img = Image.fromarray(heatmap_rgb)
        run_id = f"costmap_{int(time.time())}"
        costmap_filename = f"{run_id}.png"
        costmap_path = os.path.join(RESULTS_FOLDER, "final_costmaps")
        os.makedirs(costmap_path, exist_ok=True)
        final_path = os.path.join(costmap_path, costmap_filename)
        
        costmap_img.save(final_path)

        costmap_filename = f"{run_id}_bw.png"
        final_path = os.path.join(costmap_path, costmap_filename)
        cv2.imwrite(final_path, (normalized_costmap * 255).astype(np.uint8))

        # Also save inside the TIFF folder if we know which TIFF
        if tiff_folder_name:
            tiff_costmap_dir = os.path.join(RESULTS_FOLDER, tiff_folder_name, "costmap", "temp_latest")
            if os.path.exists(tiff_costmap_dir):
                shutil.rmtree(tiff_costmap_dir)
            os.makedirs(tiff_costmap_dir, exist_ok=True)
            costmap_img.save(os.path.join(tiff_costmap_dir, "costmap.png"))
            cv2.imwrite(os.path.join(tiff_costmap_dir, "costmap_bw.png"), (normalized_costmap * 255).astype(np.uint8))
            print(f"Costmaps also saved to {tiff_costmap_dir}")

        return jsonify({
            "costmap_url": f"/results/final_costmaps/{run_id}_bw.png",
            "colored_url": f"/results/final_costmaps/{run_id}.png"
        })

    except Exception as e:
        print(f"!!! COSTMAP GENERATION ERROR: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"An error occurred during costmap generation: {e}"}), 500

# [MODIFIED] Planning Endpoint
@app.route("/api/plan-path", methods=["POST"])
def plan_path():
    if not HAS_ASTAR:
        return jsonify({"error": "A* planner is not available on the server."}), 503

    data = request.json
    costmap_url = data.get('costmap_url')
    tiff_filename = data.get('tiff_filename')
    start_point = data.get('start_point')
    goal_point = data.get('goal_point')
    display_dims = data.get('display_dimensions')

    if not all([costmap_url, tiff_filename, start_point, goal_point, display_dims]):
        return jsonify({"error": "Missing data for planning."}), 400

    try:
        costmap_path = os.path.join(_basedir, costmap_url.lstrip('/'))
        with Image.open(costmap_path) as img:
            costmap_for_planner = np.array(ImageOps.grayscale(img))

        tiff_path = _resolve_tiff_path(tiff_filename)
        if not tiff_path:
            return jsonify({"error": f"TIFF '{tiff_filename}' not found."}), 404
        with Image.open(tiff_path) as img:
            original_dims = img.size
        
        costmap_pil = Image.fromarray(costmap_for_planner)
        resized_costmap = costmap_pil.resize(original_dims, Image.NEAREST)
        costmap_array = np.array(resized_costmap)

        scale_x = original_dims[0] / display_dims['width']
        scale_y = original_dims[1] / display_dims['height']

        start_scaled = (int(start_point['x'] * scale_x), int(start_point['y'] * scale_y))
        goal_scaled = (int(goal_point['x'] * scale_x), int(goal_point['y'] * scale_y))

        print(f"Planning from {start_scaled} to {goal_scaled} on a {costmap_array.shape} costmap.")
        path = astar_bind.astar(costmap_array.astype(np.float32), start_scaled[0], start_scaled[1], goal_scaled[0], goal_scaled[1])

        if not path:
            return jsonify({"error": "No path found."}), 404
            
        return jsonify({
            "path": path, 
            "original_dimensions": {"width": original_dims[0], "height": original_dims[1]}
        })

    except Exception as e:
        print(f"!!! PLANNING ERROR: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"An error occurred during planning: {e}"}), 500

# --- Parameter Persistence ---
@app.route("/api/params", methods=["GET"])
def get_params():
    """Load saved parameters from temp_parameters.json, or return defaults."""
    try:
        if os.path.exists(TEMP_PARAMS_FILE):
            with open(TEMP_PARAMS_FILE, 'r') as f:
                params = json.load(f)
            # Merge with defaults to handle any new params added after save
            merged = dict(DEFAULT_PARAMS)
            merged.update(params)
            return jsonify({"params": merged, "source": "saved"})
    except (json.JSONDecodeError, Exception) as e:
        print(f"Corrupt temp_parameters.json, reverting to defaults: {e}")
        try:
            os.remove(TEMP_PARAMS_FILE)
        except FileNotFoundError:
            pass
    return jsonify({"params": DEFAULT_PARAMS, "source": "default"})

@app.route("/api/params", methods=["POST"])
def save_params():
    """Save parameters to temp_parameters.json."""
    data = request.json or {}
    params = data.get("params", {})
    try:
        with open(TEMP_PARAMS_FILE, 'w') as f:
            json.dump(params, f, indent=2)
        return jsonify({"message": "Parameters saved."})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/params", methods=["DELETE"])
def reset_params():
    """Reset parameters to default by deleting temp_parameters.json."""
    try:
        if os.path.exists(TEMP_PARAMS_FILE):
            os.remove(TEMP_PARAMS_FILE)
        return jsonify({"params": DEFAULT_PARAMS, "message": "Parameters reset to default."})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/get-result-folders", methods=["GET"])
def get_result_folders():
    try:
        folders = [d for d in os.listdir(RESULTS_FOLDER) if os.path.isdir(os.path.join(RESULTS_FOLDER, d)) and d != "final_costmaps"]
        folders.sort(reverse=True)
        return jsonify({"folders": folders})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/browse-results", methods=["POST"])
def browse_results():
    """Browse the results folder structure. Returns list of items at the given subpath.
    mode: 'tiff' = only show tiff files (hide mask dirs), 'masks' = only show dirs (hide files), 'all' = show everything
    """
    data = request.json or {}
    subpath = data.get("path", "")
    mode = data.get("mode", "all")  # 'tiff', 'masks', 'all'
    
    # Sanitize: prevent escaping results folder
    subpath = subpath.strip("/")
    # Reject paths with .. to prevent directory traversal
    if ".." in subpath:
        return jsonify({"error": "Invalid path."}), 400
    
    target_dir = os.path.join(RESULTS_FOLDER, subpath) if subpath else RESULTS_FOLDER
    
    if not os.path.exists(target_dir) or not os.path.isdir(target_dir):
        return jsonify({"error": "Path not found."}), 404
    
    # Ensure we're still within RESULTS_FOLDER
    real_target = os.path.realpath(target_dir)
    real_results = os.path.realpath(RESULTS_FOLDER)
    if not real_target.startswith(real_results):
        return jsonify({"error": "Invalid path."}), 400
    
    try:
        items = []
        for name in sorted(os.listdir(target_dir)):
            full_path = os.path.join(target_dir, name)
            # Skip hidden files, special folders, and json/error task files
            if name.startswith(".") or name == "final_costmaps":
                continue
            if name.endswith(".json") or name.endswith(".error"):
                continue
            if mode == "masks" and name == "costmap":
                continue
                
            if os.path.isdir(full_path):
                # Check if this folder contains mask PNGs (is a "run" folder)
                has_refined = os.path.isdir(os.path.join(full_path, "refined"))
                
                if mode == "masks" and not subpath:
                    # In masks mode at root, skip temp_latest dirs
                    if name == "temp_latest":
                        continue
                
                items.append({"name": name, "type": "dir", "has_masks": has_refined})
            else:
                if mode == "masks":
                    # In masks mode, skip all files (we only want dirs)
                    continue
                    
                items.append({"name": name, "type": "file"})
        
        return jsonify({"items": items, "path": subpath})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/search-masks", methods=["POST"])
def search_masks():
    """Search for mask folders (folders containing 'refined/' subfolder) matching a query."""
    data = request.json or {}
    query = data.get("query", "").strip().lower()
    
    if not query:
        return jsonify({"results": []})
    
    results = []
    
    for root, dirs, files in os.walk(RESULTS_FOLDER):
        # Get relative path from RESULTS_FOLDER
        rel_path = os.path.relpath(root, RESULTS_FOLDER)
        if rel_path == ".":
            rel_path = ""
        
        # Skip hidden dirs and special dirs
        dirs[:] = [d for d in dirs if not d.startswith(".") and d != "final_costmaps"]
        
        folder_name = os.path.basename(root)
        
        # Check if this directory has a 'refined' subfolder (is a mask run folder)
        if os.path.isdir(os.path.join(root, "refined")):
            # Check if any part of the path matches the query
            if query in rel_path.lower() or query in folder_name.lower():
                results.append({
                    "name": folder_name,
                    "path": rel_path,
                    "has_masks": True
                })
        
        # Limit results
        if len(results) >= 20:
            break
    
    return jsonify({"results": results})

@app.route("/api/rename-tiff-folder", methods=["POST"])
def rename_tiff_folder():
    """Rename a TIFF folder in results/."""
    data = request.json or {}
    old_name = data.get("old_name", "").strip()
    new_name = data.get("new_name", "").strip()
    
    if not old_name or not new_name:
        return jsonify({"error": "Both old_name and new_name are required."}), 400
    
    # Sanitize
    if ".." in old_name or "/" in old_name or ".." in new_name or "/" in new_name:
        return jsonify({"error": "Invalid folder name."}), 400
    
    old_path = os.path.join(RESULTS_FOLDER, old_name)
    new_path = os.path.join(RESULTS_FOLDER, new_name)
    
    if not os.path.exists(old_path):
        return jsonify({"error": f"Folder '{old_name}' not found."}), 404
    
    if os.path.exists(new_path):
        return jsonify({"error": f"Folder '{new_name}' already exists."}), 409
    
    try:
        os.rename(old_path, new_path)
        
        # Also rename in uploads/tiffs for backward compat
        for ext in ['.tif', '.tiff']:
            old_tiff = os.path.join(TIFF_FOLDER, old_name + ext)
            new_tiff = os.path.join(TIFF_FOLDER, new_name + ext)
            if os.path.exists(old_tiff):
                os.rename(old_tiff, new_tiff)
        
        print(f"Renamed: {old_name} -> {new_name}")
        return jsonify({"message": f"Renamed '{old_name}' to '{new_name}'", "new_name": new_name})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/download-result-file", methods=["POST"])
def download_result_file():
    """Download a file from within results/."""
    data = request.json or {}
    subpath = data.get("path", "").strip("/")
    
    if ".." in subpath:
        return jsonify({"error": "Invalid path."}), 400
    
    full_path = os.path.join(RESULTS_FOLDER, subpath)
    
    # Ensure within RESULTS_FOLDER
    real_path = os.path.realpath(full_path)
    real_results = os.path.realpath(RESULTS_FOLDER)
    if not real_path.startswith(real_results):
        return jsonify({"error": "Invalid path."}), 400
    
    if not os.path.exists(full_path) or not os.path.isfile(full_path):
        return jsonify({"error": "File not found."}), 404
    
    return send_file(full_path, as_attachment=True, download_name=os.path.basename(full_path))

@app.route("/api/save-masks", methods=["POST"])
def save_masks():
    """Save masks from temp_latest to a permanent date/time folder."""
    data = request.json
    tiff_folder = data.get("tiff_folder")
    suffix = data.get("suffix", "").strip()
    
    if not tiff_folder:
        return jsonify({"error": "Missing tiff_folder."}), 400
    
    # Sanitize folder name
    if ".." in tiff_folder or "/" in tiff_folder:
        return jsonify({"error": "Invalid tiff_folder."}), 400
    
    temp_dir = os.path.join(RESULTS_FOLDER, tiff_folder, "mask", "temp_latest")
    if not os.path.exists(temp_dir):
        return jsonify({"error": "No temporary results found. Run the pipeline first."}), 404
    
    try:
        from datetime import datetime
        now = datetime.now()
        date_str = now.strftime("%Y-%m-%d")
        time_str = now.strftime("%H-%M-%S")
        
        folder_name = f"{time_str}_{suffix}" if suffix else time_str
        
        dest_dir = os.path.join(RESULTS_FOLDER, tiff_folder, "mask", date_str, folder_name)
        os.makedirs(os.path.dirname(dest_dir), exist_ok=True)
        
        # Copy temp_latest to the new permanent location
        shutil.copytree(temp_dir, dest_dir)
        
        print(f"Masks saved to: {dest_dir}")
        return jsonify({
            "message": f"Masks saved successfully to {tiff_folder}/mask/{date_str}/{folder_name}",
            "saved_path": f"{tiff_folder}/mask/{date_str}/{folder_name}"
        })
    except Exception as e:
        print(f"!!! SAVE MASKS ERROR: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/api/save-costmap", methods=["POST"])
def save_costmap():
    """Save costmaps to a structured date/time/suffix folder."""
    data = request.json
    tiff_folder = data.get("tiff_folder")
    suffix = data.get("suffix", "").strip()
    costmap_url = data.get("costmap_url")

    if not tiff_folder or not costmap_url:
        return jsonify({"error": "Missing tiff_folder or costmap_url."}), 400

    # Sanitize folder name
    if ".." in tiff_folder or "/" in tiff_folder:
        return jsonify({"error": "Invalid tiff_folder."}), 400

    # Determine source paths
    # costmap_url is relative like /results/final_costmaps/run_id.png
    # But usually we want both run_id.png (overlay) and run_id_bw.png (heatmap).
    # Extract filename from URL
    try:
        filename = os.path.basename(costmap_url) # e.g. costmap_123.png
        run_name = os.path.splitext(filename)[0] # costmap_123
        
        # Check if _bw suffix is present, handle accordingly.
        # Usually overlay is costmap_123.png, bw is costmap_123_bw.png
        # If passed URL is bw, strip it.
        if run_name.endswith("_bw"):
            run_name = run_name[:-3]
            
        src_overlay = os.path.join(RESULTS_FOLDER, "final_costmaps", f"{run_name}.png")
        src_bw = os.path.join(RESULTS_FOLDER, "final_costmaps", f"{run_name}_bw.png")
        
        print(f"Saving Costmap: Run Name={run_name}, Src Overlay={src_overlay}, Src BW={src_bw}")

        if not os.path.exists(src_overlay):
             print(f"Error: Source overlay not found: {src_overlay}")
             return jsonify({"error": f"Source costmap not found: {src_overlay}"}), 404

        from datetime import datetime
        now = datetime.now()
        date_str = now.strftime("%Y-%m-%d")
        time_str = now.strftime("%H-%M-%S")
        
        folder_name = f"{time_str}_{suffix}" if suffix else time_str
        
        # Structure: results/<tiff>/costmap/<date>/<folder_name>/
        dest_dir = os.path.join(RESULTS_FOLDER, tiff_folder, "costmap", date_str, folder_name)
        os.makedirs(dest_dir, exist_ok=True)
        
        shutil.copy(src_overlay, os.path.join(dest_dir, "costmap.png"))
        if os.path.exists(src_bw):
            shutil.copy(src_bw, os.path.join(dest_dir, "costmap_bw.png"))
        else:
            print(f"Warning: Source BW costmap not found: {src_bw}")
            
        return jsonify({
            "message": f"Costmap saved to {tiff_folder}/costmap/{date_str}/{folder_name}/"
        })
        
    except Exception as e:
        print(f"SAVE COSTMAP ERROR: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/api/load-masks-from-path", methods=["POST"])
def load_masks_from_path():
    """Load masks from a specific path within results/."""
    data = request.json or {}
    subpath = data.get("path", "")
    
    # Sanitize
    subpath = subpath.strip("/")
    if ".." in subpath:
        return jsonify({"error": "Invalid path."}), 400
    
    refined_mask_dir = os.path.join(RESULTS_FOLDER, subpath, "refined")
    if not os.path.exists(refined_mask_dir):
        return jsonify({"error": f"Refined masks folder not found at: {subpath}"}), 404
    
    try:
        mask_urls = {}
        first_shape = None
        for filename in os.listdir(refined_mask_dir):
            if filename.endswith(".png"):
                with Image.open(os.path.join(refined_mask_dir, filename)) as img:
                    if first_shape is None:
                        first_shape = img.size
                    elif img.size != first_shape:
                        return jsonify({"error": "Masks in this folder have inconsistent shapes."}), 400
                
                class_name = os.path.splitext(filename)[0].replace("_", " ")
                mask_urls[class_name] = f"/results/{subpath}/refined/{filename}"

        semantic_masks = {}
        semantic_mask_dir = os.path.join(RESULTS_FOLDER, subpath, "semantic")
        if os.path.exists(semantic_mask_dir):
            for filename in os.listdir(semantic_mask_dir):
                 if filename.endswith(".png"):
                    class_name = os.path.splitext(filename)[0].replace("_", " ")
                    semantic_masks[class_name] = f"/results/{subpath}/semantic/{filename}"

        return jsonify({"refined_masks": mask_urls, "semantic_masks": semantic_masks})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Keep old endpoint for backward compatibility
@app.route("/api/load-masks-from-folder/<run_id>", methods=["GET"])
def load_masks_from_folder(run_id):
    refined_mask_dir = os.path.join(RESULTS_FOLDER, run_id, "refined")
    if not os.path.exists(refined_mask_dir):
        return jsonify({"error": f"Refined masks folder not found for run ID: {run_id}"}), 404
    
    try:
        mask_urls = {}
        first_shape = None
        for filename in os.listdir(refined_mask_dir):
            if filename.endswith(".png"):
                with Image.open(os.path.join(refined_mask_dir, filename)) as img:
                    if first_shape is None:
                        first_shape = img.size
                    elif img.size != first_shape:
                        return jsonify({"error": "Masks in this folder have inconsistent shapes."}), 400
                
                class_name = os.path.splitext(filename)[0].replace("_", " ")
                mask_urls[class_name] = f"/results/{run_id}/refined/{filename}"

        semantic_masks = {}
        semantic_mask_dir = os.path.join(RESULTS_FOLDER, run_id, "semantic")
        if os.path.exists(semantic_mask_dir):
            for filename in os.listdir(semantic_mask_dir):
                 if filename.endswith(".png"):
                    class_name = os.path.splitext(filename)[0].replace("_", " ")
                    semantic_masks[class_name] = f"/results/{run_id}/semantic/{filename}"

        return jsonify({"refined_masks": mask_urls, "semantic_masks": semantic_masks})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/download-costmap", methods=["POST"])
def download_costmap():
    """
    Create a downloadable costmap in PNG or GeoTIFF format.
    """
    data = request.json
    format_choice = data.get("format", "png")  # "png" or "tiff"
    costmap_url = data.get("costmap_url")
    tiff_filename = data.get("tiff_filename")

    if not costmap_url:
        return jsonify({"error": "No costmap file provided."}), 400

    costmap_path = os.path.join(_basedir, costmap_url.lstrip('/'))
    if not os.path.exists(costmap_path):
        return jsonify({"error": "Costmap file not found."}), 404

    if format_choice == "png":
        # Simply return PNG directly
        return send_file(costmap_path, as_attachment=True)

    elif format_choice == "tiff":
        if not tiff_filename:
            return jsonify({"error": "Original GeoTIFF filename required for TIFF download."}), 400

        input_tiff = _resolve_tiff_path(tiff_filename)
        if not input_tiff:
            return jsonify({"error": f"Original GeoTIFF '{tiff_filename}' not found."}), 404

        # Prepare output path
        output_dir = os.path.join(RESULTS_FOLDER, "final_costmaps")
        os.makedirs(output_dir, exist_ok=True)
        output_tiff = os.path.join(output_dir, f"{_get_tiff_folder_name(tiff_filename)}_costmap.tif")

        try:
            # --- Replace image inside the original GeoTIFF ---
            dataset = gdal.Open(input_tiff, gdal.GA_ReadOnly)
            geotransform = dataset.GetGeoTransform()
            projection = dataset.GetProjection()

            new_image = cv2.imread(costmap_path, cv2.IMREAD_UNCHANGED)
            if new_image is None:
                raise ValueError("Failed to read the generated costmap.")

            height, width = new_image.shape[:2]
            bands = new_image.shape[2] if len(new_image.shape) == 3 else 1

            driver = gdal.GetDriverByName("GTiff")
            new_dataset = driver.Create(output_tiff, width, height, bands, gdal.GDT_Byte)

            if geotransform:
                new_dataset.SetGeoTransform(geotransform)
            if projection:
                new_dataset.SetProjection(projection)

            if bands > 1:
                for i in range(bands):
                    new_dataset.GetRasterBand(i + 1).WriteArray(new_image[:, :, i])
            else:
                new_dataset.GetRasterBand(1).WriteArray(new_image)

            new_dataset.FlushCache()
            new_dataset = None
            dataset = None

            return send_file(output_tiff, as_attachment=True)

        except Exception as e:
            import traceback; traceback.print_exc()
            return jsonify({"error": f"Failed to create GeoTIFF: {e}"}), 500

    else:
        return jsonify({"error": f"Unsupported format: {format_choice}"}), 400

@app.route("/api/upload-tiff", methods=["POST"])
def upload_tiff():
    """
    Upload a new TIFF file and save it to both TIFF_FOLDER and results/<name>/original.tif.
    """
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if not file.filename.lower().endswith((".tif", ".tiff")):
        return jsonify({"error": "Only .tif or .tiff files are allowed"}), 400

    # Save to uploads/tiffs (backward compat)
    save_path = os.path.join(TIFF_FOLDER, file.filename)
    file.save(save_path)
    
    # Also save into organized results structure
    folder_name = _ensure_tiff_in_results(file.filename)

    return jsonify({"message": "TIFF uploaded successfully", "filename": file.filename, "folder_name": folder_name})



@app.route("/api/download-plan", methods=["POST"])
def download_plan():
    """
    Downloads a ZIP file containing selectable items:
    1. RGB Plan (Full Res)
    2. White Plan (Full Res)
    3. Metadata.txt
    4. Costmap Folder (PNGs)
    5. Costmap TIFF (GeoTIFF)
    6. Original TIFF
    7. Masks Folder
    """
    data = request.json or {}
    tiff_folder = data.get("tiff_folder")
    costmap_url = data.get("costmap_url") # Relative URL
    path_points = data.get("path") # List of {x, y}
    start = data.get("start")
    end = data.get("end")
    options = data.get("options", {}) # { "include_rgb": true, ... }
    zip_name = (data.get("zip_name") or "").strip()
    output = data.get("output") or {}
    want_zip = output.get("zip", True)
    want_individual = output.get("individual", False)
    
    # Default options (if empty) - Enable ALL by default per user request
    if not options:
        options = {
            "rgb_plan": True,
            "white_plan": True,
            "metadata": True,
            "costmap_files": True,
            "costmap_tiff": True,
            "original_tiff": True,
            "masks": True
        }

    if not path_points or not tiff_folder:
         return jsonify({"error": "Missing path or tiff info."}), 400
	     
    try:
        def _normalize_xy(p):
            if p is None:
                return None
            if isinstance(p, dict):
                if "x" in p and "y" in p:
                    return (int(p["x"]), int(p["y"]))
                raise ValueError(f"Invalid point dict: {p}")
            if isinstance(p, (list, tuple)) and len(p) >= 2:
                return (int(p[0]), int(p[1]))
            raise ValueError(f"Invalid point: {p}")

        # Normalize planner payloads: frontend planner uses [x, y] path points,
        # while older callers may send [{"x":..,"y":..}, ...].
        path_points_xy = [_normalize_xy(p) for p in path_points]
        start_xy = _normalize_xy(start) if start else None
        end_xy = _normalize_xy(end) if end else None

        # Resolve paths
        if not costmap_url:
            return jsonify({"error": "Missing costmap_url."}), 400

        if costmap_url.startswith("/results/"):
            rel_path = costmap_url.replace("/results/", "", 1)
            costmap_path = os.path.join(RESULTS_FOLDER, rel_path)
        else:
            costmap_path = os.path.join(RESULTS_FOLDER, costmap_url)
            
        if not os.path.exists(costmap_path):
             return jsonify({"error": f"Costmap file not found at {costmap_path}"}), 404
             
        # Resolve original TIFF path
        original_tiff_path = os.path.join(RESULTS_FOLDER, tiff_folder, "original.tif")
        if not os.path.exists(original_tiff_path):
             # Fallback to upload folder?
             original_tiff_path = os.path.join(TIFF_FOLDER, tiff_folder) # Assuming tiff_folder is filename
             if not os.path.exists(original_tiff_path):
                  # Try searching uploads
                  pass 

        def _safe_zip_basename(name: str) -> str:
            # Keep only safe characters; collapse whitespace; prevent path traversal.
            allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_.() ")
            cleaned = "".join(ch for ch in name if ch in allowed).strip()
            cleaned = re.sub(r"\s+", " ", cleaned)
            cleaned = cleaned.replace("..", ".")
            if not cleaned:
                return ""
            if not cleaned.lower().endswith(".zip"):
                cleaned += ".zip"
            return cleaned

        # Create temp dir for outputs (allow overwriting by name)
        run_timestamp = int(time.time())
        default_name = f"plan_{tiff_folder}_{run_timestamp}.zip"
        zip_basename = _safe_zip_basename(zip_name) or _safe_zip_basename(default_name)
        base_name = os.path.splitext(zip_basename)[0]
        downloads_dir = os.path.join(RESULTS_FOLDER, "temp_downloads")
        os.makedirs(downloads_dir, exist_ok=True)
        zip_path = os.path.join(downloads_dir, zip_basename)
        temp_dir = os.path.join(downloads_dir, os.path.splitext(zip_basename)[0])

        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        if os.path.exists(zip_path):
            os.remove(zip_path)
        masks_zip_path = os.path.join(downloads_dir, f"{base_name}_masks.zip")
        if os.path.exists(masks_zip_path):
            os.remove(masks_zip_path)
        os.makedirs(temp_dir, exist_ok=True)

        individual_files = []

        def _add_individual(url_path: str, suggested_filename: str):
            individual_files.append({"url": url_path, "filename": suggested_filename})
        
        # --- 1. RGB Plan (Full Res) ---
        if options.get("rgb_plan") and os.path.exists(original_tiff_path):
            try:
                # Load Original TIFF (Max resolution)
                # Note: PIL.Image.MAX_IMAGE_PIXELS might need bumping for huge images
                Image.MAX_IMAGE_PIXELS = None 
                with Image.open(original_tiff_path) as img:
                    img = img.convert("RGB")
                    draw = ImageDraw.Draw(img)
                    
                    # Draw Path (Red, thicker line for high res)
                    points = path_points_xy
                    if len(points) > 1:
                        draw.line(points, fill="red", width=5)
                        
                    img.save(os.path.join(temp_dir, "plan_on_rgb.png")) # Save as PNG? Or TIFF? PNG for viewing.
            except Exception as e:
                print(f"Error generating RGB plan: {e}")
                
        # --- 2. White Plan (Full Res) ---
        if options.get("white_plan") and os.path.exists(original_tiff_path):
            try:
                Image.MAX_IMAGE_PIXELS = None
                with Image.open(original_tiff_path) as ref_img:
                    size = ref_img.size
                
                white_img = Image.new("RGB", size, "white")
                draw = ImageDraw.Draw(white_img)
                
                points = path_points_xy
                if len(points) > 1:
                    draw.line(points, fill="red", width=5)
                    
                white_img.save(os.path.join(temp_dir, "plan_on_white.png"))
            except Exception as e:
                print(f"Error generating White plan: {e}")

        # --- 3. Metadata ---
        if options.get("metadata"):
            with open(os.path.join(temp_dir, "metadata.txt"), "w") as f:
                f.write(f"Start (raw): {start}\n")
                f.write(f"End (raw): {end}\n")
                f.write(f"TIFF: {tiff_folder}\n")
                f.write(f"Timestamp: {run_timestamp}\n")
                f.write(f"Points Count: {len(path_points)}\n")
        
        # --- 4. Costmap Folder ---
        if options.get("costmap_files"):
            costmap_dest = os.path.join(temp_dir, "costmap")
            os.makedirs(costmap_dest, exist_ok=True)

            base, ext = os.path.splitext(costmap_path)
            if base.endswith("_bw"):
                bw_path = costmap_path
                overlay_path = base[:-3] + ext
            else:
                overlay_path = costmap_path
                bw_path = base + "_bw" + ext

            if os.path.exists(overlay_path):
                shutil.copy(overlay_path, os.path.join(costmap_dest, "costmap.png"))
            if os.path.exists(bw_path):
                shutil.copy(bw_path, os.path.join(costmap_dest, "costmap_bw.png"))
        
        # --- 5. Costmap TIFF ---
        if options.get("costmap_tiff") and os.path.exists(original_tiff_path):
             # Generate GeoTIFF from costmap (using logic from download_geotiff)
             # Needs gdal
             try:
                 # Load BW costmap for values
                 base, ext = os.path.splitext(costmap_path)
                 bw_path = base + "_bw" + ext if not base.endswith("_bw") else costmap_path
                 if os.path.exists(bw_path):
                     # Call helper or inline logic? Inline for now to avoid complexity
                     ds = gdal.Open(original_tiff_path, gdal.GA_ReadOnly)
                     if ds:
                         gt = ds.GetGeoTransform()
                         proj = ds.GetProjection()
                         
                         cost_img = cv2.imread(bw_path, cv2.IMREAD_UNCHANGED)
                         h, w = cost_img.shape[:2]
                         
                         out_tiff = os.path.join(temp_dir, "costmap.tif")
                         driver = gdal.GetDriverByName("GTiff")
                         out_ds = driver.Create(out_tiff, w, h, 1, gdal.GDT_Byte) # Byte sufficient? Normalized [0,255]
                         if gt: out_ds.SetGeoTransform(gt)
                         if proj: out_ds.SetProjection(proj)
                         out_ds.GetRasterBand(1).WriteArray(cost_img)
                         out_ds = None
                         ds = None
             except Exception as e:
                 print(f"Error generating Costmap TIFF: {e}")

        # --- 6. Original TIFF ---
        if options.get("original_tiff") and os.path.exists(original_tiff_path):
            shutil.copy(original_tiff_path, os.path.join(temp_dir, "original.tif"))

        # --- 7. Masks Folder ---
        if options.get("masks"):
            # Where are masks? Assuming temp_latest if just generated, or ...
            # We don't track specific mask folder used for costmap generation easily unless passed.
            # But usually it's `temp_latest` if fresh.
            # Or if generating from saved masks?
            # User wants "masks folder with the masks in it".
            # I'll try `results/<tiff>/mask/temp_latest/refined`.
            masks_src = os.path.join(RESULTS_FOLDER, tiff_folder, "mask", "temp_latest", "refined")
            if os.path.exists(masks_src):
                shutil.copytree(masks_src, os.path.join(temp_dir, "masks"))

        # Build individual downloads manifest (and any per-item zips)
        if want_individual:
            # These files live under /results/temp_downloads/<base_name>/...
            base_url_prefix = f"/results/temp_downloads/{base_name}"

            def _maybe_add(rel_name: str, suffix_name: str):
                full_path = os.path.join(temp_dir, rel_name)
                if os.path.exists(full_path):
                    _add_individual(f"{base_url_prefix}/{rel_name}", f"{base_name}_{suffix_name}")

            if options.get("rgb_plan"):
                _maybe_add("plan_on_rgb.png", "plan_on_rgb.png")
            if options.get("white_plan"):
                _maybe_add("plan_on_white.png", "plan_on_white.png")
            if options.get("metadata"):
                _maybe_add("metadata.txt", "metadata.txt")
            if options.get("costmap_tiff"):
                _maybe_add("costmap.tif", "costmap.tif")
            if options.get("original_tiff"):
                _maybe_add("original.tif", "original.tif")

            if options.get("costmap_files"):
                _maybe_add("costmap/costmap.png", "costmap.png")
                _maybe_add("costmap/costmap_bw.png", "costmap_bw.png")

            if options.get("masks"):
                masks_dir = os.path.join(temp_dir, "masks")
                if os.path.exists(masks_dir):
                    # Zip masks folder as a single downloadable item
                    shutil.make_archive(masks_zip_path.replace(".zip", ""), "zip", temp_dir, "masks")
                    _add_individual(
                        f"/results/temp_downloads/{os.path.basename(masks_zip_path)}",
                        os.path.basename(masks_zip_path),
                    )

        # Zip (optional)
        zip_url = None
        if want_zip:
            shutil.make_archive(zip_path.replace('.zip', ''), 'zip', temp_dir)
            zip_url = f"/results/temp_downloads/{zip_basename}"
        
        # Cleanup
        # shutil.rmtree(temp_dir) 

        return jsonify({
            "download_url": zip_url,
            "zip_filename": zip_basename if want_zip else None,
            "base_name": base_name,
            "individual_files": individual_files if want_individual else [],
        })
        
    except Exception as e:
        print(f"DOWNLOAD PLAN ERROR: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/api/clear-temp-downloads", methods=["POST"])
def clear_temp_downloads():
    """Delete files/folders under results/temp_downloads/ (keeps the folder)."""
    try:
        downloads_dir = os.path.join(RESULTS_FOLDER, "temp_downloads")
        if not os.path.exists(downloads_dir):
            return jsonify({"deleted": 0})

        deleted = 0
        for name in os.listdir(downloads_dir):
            path = os.path.join(downloads_dir, name)
            try:
                if os.path.isdir(path):
                    shutil.rmtree(path)
                    deleted += 1
                elif os.path.isfile(path):
                    os.remove(path)
                    deleted += 1
            except Exception as e:
                print(f"Failed to delete {path}: {e}")
        return jsonify({"deleted": deleted})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, port=5002, host='0.0.0.0', threaded=True, processes=1)
