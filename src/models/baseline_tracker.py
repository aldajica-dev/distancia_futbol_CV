"""Línea base: tracking con BoT-SORT + GMC sobre cámara táctica dinámica."""

import csv
import os
import sys

import cv2
import numpy as np
from ultralytics import YOLO

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.features.projection import estimate_camera_motion, compensate_point, pixels_to_meters

VIDEO_INPUT = os.path.join(os.path.dirname(__file__), "..", "..", "data", "raw", "video.mp4")
VIDEO_OUTPUT = os.path.join(os.path.dirname(__file__), "..", "..", "data", "processed", "baseline_tactical.mp4")
CSV_OUTPUT = os.path.join(os.path.dirname(__file__), "..", "..", "data", "processed", "baseline_distances.csv")
MODEL_PATH = "yolov8n.pt"


def main():
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

    frame_num = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_num += 1
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

            if affine is not None and tid in track_history:
                compensated = compensate_point(track_history[tid], affine)
                dist = pixels_to_meters(current_pos, compensated)
                total_distances[tid] = total_distances.get(tid, 0.0) + dist

            track_history[tid] = current_pos

            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            label = f"ID:{tid} {total_distances.get(tid, 0.0):.1f}m"
            cv2.putText(frame, label, (int(x1), int(y1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        prev_gray = curr_gray
        out.write(frame)

    cap.release()
    out.release()

    with open(CSV_OUTPUT, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Track_ID", "Distancia_Metros"])
        for tid in sorted(total_distances):
            writer.writerow([tid, f"{total_distances[tid]:.2f}"])

    print(f"Video guardado en: {VIDEO_OUTPUT}")
    print(f"CSV guardado en: {CSV_OUTPUT}")
    print(f"Jugadores rastreados: {len(total_distances)}")


if __name__ == "__main__":
    main()
