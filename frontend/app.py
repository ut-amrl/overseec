import os
import shutil
from flask import Flask, request, jsonify, send_from_directory
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
from multiprocessing import Process, Queue
import importlib.util
import matplotlib.pyplot as plt
import cv2
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
log_queue = Queue()

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

def pipeline_worker(log_q, task_id, tiff_filename, classes_data, params):
    class QueueLogger:
        def __init__(self, queue): self.queue = queue
        def write(self, text): self.queue.put(text)
        def flush(self): pass

    sys.stdout = QueueLogger(log_q)
    sys.stderr = QueueLogger(log_q)

    try:
        sat_img_path = os.path.join(TIFF_FOLDER, tiff_filename)
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
        final_config.reset()
        print("\n--- CONFIGURATION SET FOR PIPELINE RUN ---"); print(final_config); print("------------------------------------------\n")
        
        print(f"Running OVerSeeC model on: {sat_img_path}")
        overseec_sat_2_mask = OVerSeeC(config=final_config)
        (clipseg_sigmoid_mask_refiner_logits, _, _, clipseg_sigmoid_semseg_logits, _) = overseec_sat_2_mask(sat_img_path, None)
        
        run_id = f"{os.path.splitext(tiff_filename)[0]}_{int(time.time())}"
        semantic_mask_dir = os.path.join(RESULTS_FOLDER, run_id, "semantic")
        refined_mask_dir = os.path.join(RESULTS_FOLDER, run_id, "refined")
        os.makedirs(semantic_mask_dir, exist_ok=True)
        os.makedirs(refined_mask_dir, exist_ok=True)

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
            semantic_masks_urls[class_name] = f"/results/{run_id}/semantic/{class_name_safe}.png"

            # --- Save PNG and NPY for Refined Mask ---
            ref_path = os.path.join(refined_mask_dir, f"{class_name_safe}.png")
            ref_npy_path = os.path.join(refined_mask_dir, f"{class_name_safe}.npy")
            ref_tensor_slice = clipseg_sigmoid_mask_refiner_logits[i]
            save_tensor_as_image(ref_tensor_slice, ref_path)
            ref_numpy_array = ref_tensor_slice.cpu().numpy() if hasattr(ref_tensor_slice, 'cpu') else np.array(ref_tensor_slice)
            np.save(ref_npy_path, ref_numpy_array)
            refined_masks_urls[class_name] = f"/results/{run_id}/refined/{class_name_safe}.png"

        print("--- Model processing complete, masks saved. ---")
        results_data = {
            "message": "Pipeline completed successfully!",
            "semantic_masks": semantic_masks_urls,
            "refined_masks": refined_masks_urls
        }
        with open(os.path.join(RESULTS_FOLDER, f"{task_id}.json"), 'w') as f:
            json.dump(results_data, f)
    except Exception as e:
        print(f"!!! PIPELINE ERROR for task {task_id}: {e}")
        with open(os.path.join(RESULTS_FOLDER, f"{task_id}.error"), 'w') as f:
            f.write(str(e))

# --- Get the absolute path of the script's directory ---
_basedir = os.path.abspath(os.path.dirname(__file__))

# --- Configuration ---
UPLOAD_FOLDER = os.path.join(_basedir, "uploads")
TIFF_FOLDER = os.path.join(UPLOAD_FOLDER, "tiffs")
PREVIEWS_FOLDER = os.path.join(_basedir, "previews")
TEMP_FOLDER = os.path.join(_basedir, "temp_tiles")
RESULTS_FOLDER = os.path.join(_basedir, "results")
COSTMAP_FUNCTIONS_FOLDER = os.path.join(_basedir, "costmap_functions")

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
        with rasterio.open(os.path.join(TIFF_FOLDER, filename), "w", **out_meta) as dest: dest.write(mosaic)
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
    tiff_path = os.path.join(TIFF_FOLDER, filename)
    if not os.path.exists(tiff_path): return jsonify({"error": "File not found."}), 404
    try:
        preview_filename = f"{os.path.splitext(filename)[0]}_{int(time.time())}.png"
        preview_path = os.path.join(PREVIEWS_FOLDER, preview_filename)
        with Image.open(tiff_path) as img:
            img.thumbnail((800, 800)); img.convert("RGB").save(preview_path, "PNG")
        return jsonify({"preview_url": f"/previews/{preview_filename}"})
    except Exception as e: return jsonify({"error": f"Failed to generate preview: {e}"}), 500

@app.route("/api/get-tiff-files", methods=["GET"])
def get_tiff_files():
    try: return jsonify({"files": [f for f in os.listdir(TIFF_FOLDER) if f.lower().endswith(('.tiff', '.tif'))]})
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
    
    process = Process(target=pipeline_worker, args=worker_args)
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
        os.remove(result_file)
        return jsonify({"status": "completed", "results": results})
    
    if os.path.exists(error_file):
        with open(error_file, 'r') as f:
            error_message = f.read()
        if task_id in pipeline_processes:
            del pipeline_processes[task_id]
        os.remove(error_file)
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

@app.route("/api/generate-costmap", methods=["POST"])
def generate_final_costmap():
    data = request.json
    mask_urls = data.get('refined_masks')
    if not mask_urls:
        return jsonify({"error": "No mask URLs provided."}), 400

    try:
        mask_dict = {}
        binary_threshold = 0.5 * 255

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
        
        costmap_raw = costmap_module.generate_costmap(mask_dict)

        min_val, max_val = costmap_raw.min(), costmap_raw.max()
        if max_val == min_val:
            normalized_costmap = np.zeros_like(costmap_raw, dtype=np.float32)
        else:
            normalized_costmap = (costmap_raw - min_val) / (max_val - min_val)
        
        cmap = plt.get_cmap('hot')
        colored_costmap = (cmap(normalized_costmap)[:, :, :3] * 255).astype(np.uint8)
        
        costmap_img = Image.fromarray(colored_costmap)
        run_id = f"costmap_{int(time.time())}"
        costmap_filename = f"{run_id}.png"
        costmap_path = os.path.join(RESULTS_FOLDER, "final_costmaps")
        os.makedirs(costmap_path, exist_ok=True)
        final_path = os.path.join(costmap_path, costmap_filename)
        
        costmap_img.save(final_path)

        costmap_filename = f"{run_id}_bw.png"
        final_path = os.path.join(costmap_path, costmap_filename)
        cv2.imwrite(final_path, (normalized_costmap * 255).astype(np.uint8))



        return jsonify({"costmap_url": f"/results/final_costmaps/{costmap_filename}"})

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

        tiff_path = os.path.join(TIFF_FOLDER, tiff_filename)
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

@app.route("/api/get-result-folders", methods=["GET"])
def get_result_folders():
    try:
        folders = [d for d in os.listdir(RESULTS_FOLDER) if os.path.isdir(os.path.join(RESULTS_FOLDER, d)) and d != "final_costmaps"]
        folders.sort(reverse=True)
        return jsonify({"folders": folders})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

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


if __name__ == "__main__":
    app.run(debug=True, port=5002, host='0.0.0.0')
