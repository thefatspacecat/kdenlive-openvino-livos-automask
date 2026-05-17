#!/usr/bin/env python3
# Combined MobileSAM + LiVOS (OpenVINO) Daemon

import os
import sys
import argparse
import numpy as np
import torch
import cv2
import time
from PIL import Image

# --- MOBILESAM & LIVOS IMPORTS ---
try:
    from mobile_sam import sam_model_registry, SamPredictor
    from livos.model.livos_wrapper import LIVOS
    from livos.eval import InferenceCore
    import openvino.torch
except ImportError as e:
    print(f"Missing required library: {e}", file=sys.stderr)
    sys.exit(1)

def process_list(list_string):
    return np.fromstring(list_string, dtype=int, sep=',')

def process_csv(array_data, csv_string, resize):
    vals_list = csv_string.split(';')
    for vals in vals_list:
        frame, csv_data = vals.split("=")
        np_array = np.fromstring(csv_data, dtype=int, sep=',')
        if resize > 1:
            cols = int((np.shape(np_array)[0])/resize)
            np_array = np_array.reshape(cols, resize)
        array_data[int(frame)] = np_array

def save_mask(mask, filename, borders, border_color, mask_color):
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * mask_color.reshape(1, 1, -1)
    if borders > 0:
        mask_uint8 = mask.astype(np.uint8)
        contours = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
        mask_image = cv2.drawContours(mask_image.astype(np.uint8), contours, -1, border_color.tolist(), borders)
    
    pil_img = Image.fromarray(np.uint8(mask_image))
    pil_img.save(filename)

if __name__ == "__main__":
    parser = argparse.ArgumentParser("MobileSAM + LiVOS Mask Creator")
    parser.add_argument("-P", "--point_coordinates", help="Points coordinates with frame")
    parser.add_argument("-F", "--preview_frame", help="The frame index for preview", default=-1)
    parser.add_argument("-L", "--labels", help="Points labels")
    parser.add_argument("-B", "--box_coordinates", help="Box coordinates with frame")
    parser.add_argument("-I", "--inputFolder", help="folder where input jpg files are stored", default="/tmp/src-frames")
    parser.add_argument("-O", "--output", help="folder for rendered png image", default="/tmp/")
    # Reusing model/config args for MobileSAM/LiVOS paths
    parser.add_argument("-M", "--model", help="path for MobileSAM model", default="mobile_sam.pt")
    parser.add_argument("-C", "--config", help="path for LiVOS weights", default="weights/livos-nomose-480p.pth")
    parser.add_argument("-D", "--device", help="enforce a device: cuda, cpu", default="cpu")
    parser.add_argument("--color", help="mask color", default="255,100,100,180")
    parser.add_argument("--bordercolor", help="mask border color", default="255,100,100,100")
    parser.add_argument("--border", help="mask border width", default="0")
    parser.add_argument('--offload', help="offload memory to CPU", action='store_true')
    
    # Pre-parse initial args
    args, unknown = parser.parse_known_args()

    # State variables
    box = {}
    points = {}
    labels = {}

    inputFolder = args.inputFolder
    output_folder = args.output
    
    # 1. Get the directory where this python script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # 2. Join the script directory with your filenames
    mobile_sam_chkpt = os.path.join(script_dir, "mobile_sam.pt")
    livos_weights = os.path.join(script_dir, "weights", "livos-nomose-480p.pth")
    PROCESS_HEIGHT = 384  # From Script 2

    device_str = "cuda" if torch.cuda.is_available() and args.device != "cpu" else "cpu"
    device = torch.device(device_str)

    # --- 1. INITIALIZE MOBILESAM (For Preview/Editing) ---
    print(f"Loading MobileSAM to {device}...", file=sys.stderr, flush=True)
    sam = sam_model_registry["vit_t"](checkpoint=mobile_sam_chkpt)
    sam.to(device=device)
    sam.eval()
    mobile_predictor = SamPredictor(sam)

    # --- 2. INITIALIZE LIVOS (For Video Tracking) ---
    print("Loading LiVOS and injecting OpenVINO compiler...", file=sys.stderr, flush=True)
    livos_net = LIVOS(model_type='base').cpu().eval()
    livos_net.load_weights(torch.load(livos_weights, map_location='cpu'))
    
    ov_options = {
        "device": "GPU" if args.device != "cpu" else "CPU",
        "model_caching": True,
        "cache_dir": "./ov_cache",
        "config": {
            "PERFORMANCE_HINT": "LATENCY",
            "INFERENCE_NUM_THREADS": "4",
            "NUM_STREAMS": "1",
            "INFERENCE_PRECISION_HINT": "f32"
        }
    }
    livos_net = torch.compile(livos_net, backend="openvino", options=ov_options)
    livos_processor = InferenceCore(livos_net)

    # Get frames list
    frame_names = [p for p in os.listdir(inputFolder) if p.lower().endswith(".jpg")]
    frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))
    framesCount = len(frame_names)

    cached_mask_for_render = None
    cached_frame_idx_for_render = None

    def generate_preview(inArgs):
        global cached_mask_for_render, cached_frame_idx_for_render
        prev_idx = int(inArgs.preview_frame)
        
        # Load frame
        image_path = os.path.join(inputFolder, frame_names[prev_idx])
        image = np.array(Image.open(image_path).convert("RGB"))
        mobile_predictor.set_image(image)

        pts = points.get(prev_idx, None)
        lbls = labels.get(prev_idx, None)
        bx = box.get(prev_idx, None)

        if pts is not None and lbls is not None:
            masks, scores, logits = mobile_predictor.predict(
                point_coords=pts,
                point_labels=lbls,
                box=bx[0] if bx is not None else None,
                multimask_output=False
            )
            current_mask = masks[0].astype(np.uint8)
        else:
            current_mask = np.zeros(image.shape[:2], dtype=np.uint8)

        # Cache this mask for LiVOS initialization later
        cached_mask_for_render = current_mask
        cached_frame_idx_for_render = prev_idx

        filename = os.path.join(output_folder, f'preview-{prev_idx:05d}.png')
        save_mask(
            current_mask, filename, 
            int(inArgs.border), 
            process_list(inArgs.bordercolor), 
            process_list(inArgs.color)
        )
        print(f"preview ok {prev_idx}", file=sys.stdout, flush=True)

    def render_video(out_folder, border_w, border_c, mask_c):
        if cached_mask_for_render is None:
            print("ERROR: No preview mask generated prior to render.", file=sys.stderr, flush=True)
            return

        start_idx = cached_frame_idx_for_render
        
        # Give a heads up that the freeze is normal
        print(f"INFO: Compiling OpenVINO for frame {start_idx} (takes a moment)...\n", file=sys.stdout, flush=True)

        first_frame_path = os.path.join(inputFolder, frame_names[start_idx])
        first_frame_rgb = np.array(Image.open(first_frame_path).convert("RGB"))
        orig_h, orig_w = first_frame_rgb.shape[:2]
        
        scale = PROCESS_HEIGHT / orig_h
        new_w = int(orig_w * scale)

        frame_resized = cv2.resize(first_frame_rgb, (new_w, PROCESS_HEIGHT))
        mask_resized = cv2.resize(cached_mask_for_render, (new_w, PROCESS_HEIGHT), interpolation=cv2.INTER_NEAREST)

        img_tensor = torch.from_numpy(frame_resized).permute(2, 0, 1).float() / 255.0
        mask_tensor = torch.from_numpy(mask_resized).float()
        objects = [1]

        with torch.no_grad():
            livos_processor.step(img_tensor, mask_tensor, objects, is_last_frame=False)
        
        filename = os.path.join(out_folder, f'{start_idx:05d}.png')
        save_mask(cached_mask_for_render, filename, border_w, border_c, mask_c)

        # Start tracking and timing
        print("INFO: Starting video propagation...\n", file=sys.stdout, flush=True)
        start_time = time.time()

        for out_frame_idx in range(start_idx + 1, framesCount):
            frame_path = os.path.join(inputFolder, frame_names[out_frame_idx])
            frame_rgb = np.array(Image.open(frame_path).convert("RGB"))
            frame_resized = cv2.resize(frame_rgb, (new_w, PROCESS_HEIGHT))
            img_tensor = torch.from_numpy(frame_resized).permute(2, 0, 1).float() / 255.0

            with torch.no_grad():
                prob_map = livos_processor.step(img_tensor, is_last_frame=(out_frame_idx == framesCount - 1))

            object_prob = prob_map[1] if prob_map.shape[0] > 1 else prob_map[0]
            prob_map_np = object_prob.cpu().numpy()
            
            prob_upscaled = cv2.resize(prob_map_np, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
            mask_upscaled = (prob_upscaled > 0.5).astype(np.uint8)

            filename = os.path.join(out_folder, f'{out_frame_idx:05d}.png')
            save_mask(mask_upscaled, filename, border_w, border_c, mask_c)

            # --- UI UPDATES ---
            frames_processed = out_frame_idx - start_idx
            elapsed_time = time.time() - start_time
            fps = frames_processed / elapsed_time if elapsed_time > 0 else 0.01
            
            frames_left = framesCount - out_frame_idx
            eta_seconds = int(frames_left / fps)
            
            # Format ETA into Minutes:Seconds
            eta_m, eta_s = divmod(eta_seconds, 60)
            eta_str = f"{eta_m}m {eta_s}s" if eta_m > 0 else f"{eta_s}s"

            # Update the Kdenlive Info Box
            print(f"INFO: Frame {out_frame_idx}/{framesCount} | {fps:.1f} FPS | ETA: {eta_str}\n", file=sys.stdout, flush=True)

            # Update the Kdenlive Progress Bar
            percent = int(100 * out_frame_idx / framesCount)
            print(f"Export {percent}%|\n", file=sys.stderr, flush=True)

    # --- MAIN STDIN LOOP ---
    while True:
        line = sys.stdin.readline()
        if not line:
            break
        line = line.rstrip()

        if line.startswith("edit="):
            inArgs = parser.parse_args(line[5:].split())
            continue

        if line.startswith("preview="):
            inArgs = parser.parse_args(line[8:].split())
            if inArgs.point_coordinates is not None:
                process_csv(points, inArgs.point_coordinates, 2)
                process_csv(labels, inArgs.labels, 1)
            if inArgs.box_coordinates is not None:
                process_csv(box, inArgs.box_coordinates, 4)
            generate_preview(inArgs)
            continue

        if line.startswith("render="):
            output_folder_cmd = line[7:].rstrip()
            # Grab current visual settings from the last args processed
            border_w = int(args.border)
            mask_c = process_list(args.color)
            border_c = process_list(args.bordercolor)
            
            render_video(output_folder_cmd, border_w, border_c, mask_c)
            time.sleep(0.5)
            print("mask ok", file=sys.stdout, flush=True)
            sys.exit()

        if line == "q":
            print("CLOSING...\n", file=sys.stdout, flush=True)
            sys.exit()
