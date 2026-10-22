"""Línea base: detección, tracking y acumulación de distancia cruda."""

import os
import sys
import cv2
import numpy as np
from ultralytics import YOLO

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.features.projection import project_point, calculate_distance

VIDEO_INPUT = os.path.join(os.path.dirname(__file__), "..", "..", "data", "raw", "video.mp4")
VIDEO_OUTPUT = os.path.join(os.path.dirname(__file__), "..", "..", "data", "processed", "output.mp4")
MODEL_PATH = "yolov8n.pt"
H = np.eye(3)


def main():
    model = YOLO(MODEL_PATH)
    cap = cv2.VideoCapture(VIDEO_INPUT)

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(VIDEO_OUTPUT, fourcc, fps, (w, h))

    distances = {}
    prev_positions = {}

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model.track(frame, classes=[0], persist=True, verbose=False)
        boxes = results[0].boxes

        if boxes.id is not None:
            for box, track_id in zip(boxes.xyxy.cpu().numpy(), boxes.id.cpu().numpy().astype(int)):
                x1, y1, x2, y2 = box
                foot_u = (x1 + x2) / 2
                foot_v = y2

                world_pos = project_point(H, foot_u, foot_v)
                tid = int(track_id)

                if tid in prev_positions:
                    d = calculate_distance(prev_positions[tid], world_pos)
                    distances[tid] = distances.get(tid, 0.0) + d

                prev_positions[tid] = world_pos

                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                label = f"ID:{tid} {distances.get(tid, 0.0):.1f}m"
                cv2.putText(frame, label, (int(x1), int(y1) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        out.write(frame)

    cap.release()
    out.release()
    print("Video guardado en:", VIDEO_OUTPUT)


if __name__ == "__main__":
    main()
