"""Línea base: tracking con BoT-SORT + GMC sobre cámara táctica dinámica."""

import csv
import json
import os
import random
import sys
import time
import tracemalloc

import cv2
import numpy as np
import psutil
from ultralytics import YOLO

SEED = 42

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.features.projection import estimate_camera_motion, compensate_point, pixels_to_meters

VIDEO_INPUT = os.path.join(os.path.dirname(__file__), "..", "..", "data", "raw", "video.mp4")
VIDEO_OUTPUT = os.path.join(os.path.dirname(__file__), "..", "..", "data", "processed", "baseline_tactical.mp4")
CSV_OUTPUT = os.path.join(os.path.dirname(__file__), "..", "..", "data", "processed", "baseline_distances.csv")
FRAME_CSV_OUTPUT = os.path.join(os.path.dirname(__file__), "..", "..", "data", "processed", "baseline_frame_data.csv")
TIMING_OUTPUT = os.path.join(os.path.dirname(__file__), "..", "..", "data", "processed", "baseline_timing.json")
MODEL_PATH = "yolov8n.pt"


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def main():
    set_seed(SEED)

    tracemalloc.start()
    process = psutil.Process()

    try:
        import torch
        has_cuda = torch.cuda.is_available()
        if has_cuda:
            torch.cuda.reset_peak_memory_stats()
    except ImportError:
        has_cuda = False

    t_start = time.time()

    model = YOLO(MODEL_PATH)
    cap = cv2.VideoCapture(VIDEO_INPUT)
    if not cap.isOpened():
        print("ERROR: No se pudo abrir el video en:", VIDEO_INPUT)
        return

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(VIDEO_OUTPUT, fourcc, fps, (w, h))

    print(f"Video: {w}x{h} @ {fps:.1f} FPS, {total_frames} frames")
    print("Procesando...")

    track_history = {}
    total_distances = {}
    prev_gray = None
    frame_data = []
    frame_times = []

    frame_num = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_num += 1
        t_frame = time.time()

        if frame_num % 50 == 0:
            print(f"  Frame {frame_num}/{total_frames} ({100 * frame_num / total_frames:.0f}%)")

        curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        results = model.track(frame, persist=True, classes=[0], tracker="botsort.yaml", verbose=False)
        boxes_xyxy = results[0].boxes.xyxy.cpu().numpy() if results[0].boxes.id is not None else np.empty((0, 4))
        ids = results[0].boxes.id.cpu().numpy().astype(int) if results[0].boxes.id is not None else np.empty(0, dtype=int)

        affine = estimate_camera_motion(prev_gray, curr_gray, boxes_xyxy) if prev_gray is not None else None

        for box, tid in zip(boxes_xyxy, ids):
            x1, y1, x2, y2 = box
            foot_u = (x1 + x2) / 2
            foot_v = y2
            current_pos = (foot_u, foot_v)

            dist = 0.0
            if affine is not None and tid in track_history:
                compensated = compensate_point(track_history[tid], affine)
                dist = pixels_to_meters(current_pos, compensated)
                total_distances[tid] = total_distances.get(tid, 0.0) + dist

            track_history[tid] = current_pos
            frame_data.append((frame_num, tid, foot_u, foot_v, dist))

            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            label = f"ID:{tid} {total_distances.get(tid, 0.0):.1f}m"
            cv2.putText(frame, label, (int(x1), int(y1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        prev_gray = curr_gray
        out.write(frame)
        frame_times.append(time.time() - t_frame)

    cap.release()
    out.release()

    total_time = time.time() - t_start
    peak_ram_mb = tracemalloc.get_traced_memory()[1] / (1024 * 1024)
    tracemalloc.stop()

    peak_vram_mb = 0.0
    if has_cuda:
        import torch
        peak_vram_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)

    with open(CSV_OUTPUT, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Track_ID", "Distancia_Metros"])
        for tid in sorted(total_distances):
            writer.writerow([tid, f"{total_distances[tid]:.2f}"])

    with open(FRAME_CSV_OUTPUT, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "track_id", "foot_u", "foot_v", "dist_meters"])
        for row in frame_data:
            writer.writerow([row[0], row[1], f"{row[2]:.2f}", f"{row[3]:.2f}", f"{row[4]:.4f}"])

    timing = {
        "seed": SEED,
        "total_time_s": round(total_time, 2),
        "total_frames": frame_num,
        "fps_processing": round(frame_num / total_time, 2) if total_time > 0 else 0,
        "avg_frame_ms": round(1000 * sum(frame_times) / len(frame_times), 2) if frame_times else 0,
        "peak_ram_mb": round(peak_ram_mb, 1),
        "peak_vram_mb": round(peak_vram_mb, 1),
        "video_fps": fps,
        "video_resolution": f"{w}x{h}",
    }
    with open(TIMING_OUTPUT, "w") as f:
        json.dump(timing, f, indent=2)

    print(f"\nVideo guardado en: {VIDEO_OUTPUT}")
    print(f"CSV resumen: {CSV_OUTPUT}")
    print(f"CSV por frame: {FRAME_CSV_OUTPUT}")
    print(f"Timing: {TIMING_OUTPUT}")
    print(f"Tracks: {len(total_distances)} | Tiempo: {total_time:.1f}s ({frame_num / total_time:.1f} FPS)")


if __name__ == "__main__":
    main()
