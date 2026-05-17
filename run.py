import cv2
import time
import torch
import numpy as np
import openvino.torch
import urllib.request
import os

# --- MOBILESAM IMPORTS ---
from mobile_sam import sam_model_registry, SamPredictor

from livos.model.livos_wrapper import LIVOS
from livos.eval import InferenceCore

# --- CONFIGURATION ---
VIDEO_INPUT = "test_clip.mp4"
VIDEO_OUTPUT = "output_demo.mp4"
PROCESS_HEIGHT = 384
MOBILESAM_CHECKPOINT = "mobile_sam.pt"

def run_real_demo():
    print("Loading LiVOS and injecting OpenVINO compiler...")
    network = LIVOS(model_type='base').cpu().eval()
    network.load_weights(torch.load("weights/livos-nomose-480p.pth", weights_only=True, map_location='cpu'))
    
    # --- OPENVINO OPTIMIZATIONS ---
    ov_options = {
        "device": "GPU",
        "model_caching": True,
        "cache_dir": "./ov_cache",
        "config": {
            "PERFORMANCE_HINT": "LATENCY",
            "INFERENCE_NUM_THREADS": "4",
            "NUM_STREAMS": "1",
            "ENABLE_CPU_PINNING": "YES",       # Locks threads to physical cores (Linux only)
            "ENABLE_HYPER_THREADING": "NO",    # Prevents "ghost" threads from stealing cache
            "INFERENCE_PRECISION_HINT": "f32"  # 10th Gen doesn't have bf16; forcing f32 avoids conversion overhead
        }
    }
    
    network = torch.compile(network, backend="openvino", options=ov_options)
    processor = InferenceCore(network)

    cap = cv2.VideoCapture(VIDEO_INPUT)
    if not cap.isOpened():
        print(f"ERROR: Could not open {VIDEO_INPUT}.")
        return

    ret, first_frame = cap.read()
    orig_h, orig_w = first_frame.shape[:2]
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    scale = PROCESS_HEIGHT / orig_h
    new_w = int(orig_w * scale)
    first_frame_rgb = cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB)

    # --- 1. INITIAL MASK VIA MOBILESAM ---
    print("\n--- INITIALIZING MOBILESAM ---")
    if not os.path.exists(MOBILESAM_CHECKPOINT):
        print("Downloading MobileSAM weights (one-time setup)...")
        url = "https://raw.githubusercontent.com/ChaoningZhang/MobileSAM/master/weights/mobile_sam.pt"
        urllib.request.urlretrieve(url, MOBILESAM_CHECKPOINT)

    # Load MobileSAM
    sam = sam_model_registry["vit_t"](checkpoint=MOBILESAM_CHECKPOINT)
    sam.to(device="cuda" if torch.cuda.is_available() else "cpu")
    sam.eval()
    predictor = SamPredictor(sam)
    
    print("Encoding first frame (takes a second)...")
    predictor.set_image(first_frame_rgb)

    input_points = []
    input_labels = []
    current_mask = np.zeros((orig_h, orig_w), dtype=np.uint8)
    needs_update = False

    def mouse_click(event, x, y, flags, param):
        nonlocal needs_update
        if event == cv2.EVENT_LBUTTONDOWN:
            input_points.append([x, y])
            input_labels.append(1)  # 1 = Positive point (Include)
            needs_update = True
        elif event == cv2.EVENT_RBUTTONDOWN:
            input_points.append([x, y])
            input_labels.append(0)  # 0 = Negative point (Exclude)
            needs_update = True

    print("\n--- INTERACTIVE SEGMENTATION ---")
    print("LEFT CLICK: Add area (Positive)")
    print("RIGHT CLICK: Remove area (Negative)")
    print("Press 'r' to reset points. Press ENTER when finished.")
    
    cv2.namedWindow("MobileSAM - Edit Mask")
    cv2.setMouseCallback("MobileSAM - Edit Mask", mouse_click)

    while True:
        display_frame = first_frame.copy()

        # Only re-run the predictor if a new point was added
        if needs_update and len(input_points) > 0:
            pts = np.array(input_points)
            lbls = np.array(input_labels)
            
            masks, scores, logits = predictor.predict(
                point_coords=pts,
                point_labels=lbls,
                multimask_output=False,
            )
            # MobileSAM returns a boolean array, convert to uint8 (0 or 1)
            current_mask = masks[0].astype(np.uint8)
            needs_update = False

        # Overlay the current mask
        if np.max(current_mask) > 0:
            display_frame = overlay_mask(display_frame, current_mask)

        # Draw the points
        for i, pt in enumerate(input_points):
            color = (0, 255, 0) if input_labels[i] == 1 else (0, 0, 255)
            cv2.circle(display_frame, tuple(pt), 5, color, -1)
            cv2.circle(display_frame, tuple(pt), 6, (255, 255, 255), 1)

        cv2.putText(display_frame, "L-Click: Add | R-Click: Remove | 'r': Reset | ENTER: Done", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.imshow("MobileSAM - Edit Mask", display_frame)

        key = cv2.waitKey(20) & 0xFF
        if key == 13 or key == 32:  # ENTER or SPACE
            if len(input_points) > 0 and np.max(current_mask) > 0:
                break
            else:
                print("Click the subject at least once.")
        elif key == ord('r'):  # Reset
            input_points.clear()
            input_labels.clear()
            current_mask = np.zeros((orig_h, orig_w), dtype=np.uint8)
            needs_update = True
        elif key == 27:  # ESC
            print("Exited.")
            cv2.destroyAllWindows()
            return

    cv2.destroyAllWindows()
    mask_0 = current_mask

    print("\nMask locked in! Preparing LiVOS tracking...")
    
    # Resize frame and mask for the LiVOS AI
    frame_resized = cv2.resize(first_frame_rgb, (new_w, PROCESS_HEIGHT))
    mask_resized = cv2.resize(mask_0, (new_w, PROCESS_HEIGHT), interpolation=cv2.INTER_NEAREST)

    img_tensor = torch.from_numpy(frame_resized).permute(2, 0, 1).float() / 255.0
    mask_tensor = torch.from_numpy(mask_resized).float()
    objects = [1]

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(VIDEO_OUTPUT, fourcc, fps, (orig_w, orig_h))

    # 2. THE WARMUP
    print("Compiling Frame 0 (Expect a 30-60 second freeze)...")
    with torch.no_grad():
        processor.step(img_tensor, mask_tensor, objects, is_last_frame=False)
    
    out.write(overlay_mask(first_frame, mask_0))

    # 3. TRACKING LOOP
    print("\nTracking remaining frames...")
    frame_count = 1
    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, (new_w, PROCESS_HEIGHT))
        img_tensor = torch.from_numpy(frame_resized).permute(2, 0, 1).float() / 255.0

        with torch.no_grad():
            prob_map = processor.step(img_tensor, is_last_frame=False)

        object_prob = prob_map[1] if prob_map.shape[0] > 1 else prob_map[0]
        prob_map_np = object_prob.cpu().numpy()
        
        # Upscale soft probability map first, then threshold
        prob_upscaled = cv2.resize(prob_map_np, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
        mask_upscaled = (prob_upscaled > 0.5).astype(np.uint8)
        
        out.write(overlay_mask(frame, mask_upscaled))
        frame_count += 1
        print(f"\rProcessed frame {frame_count}... ({frame_count / (time.time() - start_time):.2f} FPS)", end="")

    total_time = time.time() - start_time
    print(f"\n\nDone! Tracked {frame_count} frames at {frame_count/total_time:.2f} FPS.")
    
    cap.release()
    out.release()

def overlay_mask(frame, mask):
    green_overlay = np.zeros_like(frame)
    green_overlay[:, :, 1] = 255
    colored_mask = cv2.bitwise_and(green_overlay, green_overlay, mask=mask)
    return cv2.addWeighted(frame, 1.0, colored_mask, 0.4, 0)

if __name__ == '__main__':
    run_real_demo()
