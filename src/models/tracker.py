"""Motor de evaluación modular para experimentos controlados de ablación."""

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

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.features.projection import (
    estimate_camera_motion, compensate_point, pixels_to_meters, get_perspective_scale,
)
from src.features.kinematics import smooth_point, filter_step_distance
from src.models.detector import get_device

SEED = 42
VIDEO_INPUT = os.path.join(os.path.dirname(__file__), "..", "..", "data", "raw", "video.mp4")
PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "processed")
MODEL_PATH = "yolov8n.pt"
EXPECTED_PLAYERS = 22


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


def _compute_metrics(total_distances, frame_data, frame_times, video_fps,
                     total_time, peak_ram_mb, peak_vram_mb, total_frames, config):
    """Calcula el diccionario de métricas compatible con baseline_metrics.json."""
    n_ids = len(total_distances)
    dists = np.array(list(total_distances.values())) if total_distances else np.array([0.0])

    metrics = {
        "phase": config["exp_name"],
        "config": config,
        "tracking": {
            "n_unique_ids": n_ids,
            "expected_players": EXPECTED_PLAYERS,
            "fragmentation_ratio": round(n_ids / EXPECTED_PLAYERS, 1) if n_ids > 0 else 0,
            "id_switches_estimated": max(0, n_ids - EXPECTED_PLAYERS),
        },
        "distances": {
            "total_m": round(float(dists.sum()), 1),
            "mean_per_id_m": round(float(dists.mean()), 2),
            "median_per_id_m": round(float(np.median(dists)), 2),
            "max_m": round(float(dists.max()), 2),
            "estimated_per_player_m": round(float(dists.sum()) / EXPECTED_PLAYERS, 1),
        },
    }

    # Tracking quality from frame data
    if frame_data:
        import pandas as pd
        df = pd.DataFrame(frame_data, columns=["frame", "track_id", "foot_u", "foot_v", "dist_meters", "frame_diff"])
        track_lengths = df.groupby("track_id").size()
        n_ghost_dist = int((dists < 0.1).sum())
        n_ghost_frames = int((track_lengths < 5).sum())

        metrics["tracking"]["n_ghost_tracks_by_dist"] = n_ghost_dist
        metrics["tracking"]["pct_ghost_tracks_by_dist"] = round(100 * n_ghost_dist / n_ids, 1) if n_ids > 0 else 0
        metrics["tracking"]["n_ghost_tracks_by_frames"] = n_ghost_frames
        metrics["tracking"]["avg_track_length_frames"] = round(float(track_lengths.mean()), 1)
        metrics["tracking"]["median_track_length_frames"] = round(float(track_lengths.median()), 0)

        # Kinematics — velocidad corregida por frame_diff
        valid = df.loc[(df["dist_meters"] > 0) & (df["frame_diff"] > 0)]
        speeds = (valid["dist_meters"] / valid["frame_diff"]) * video_fps
        if len(speeds) > 0:
            n_impossible = int((speeds > 12.0).sum())
            metrics["kinematics"] = {
                "mean_speed_ms": round(float(speeds.mean()), 2),
                "median_speed_ms": round(float(speeds.median()), 2),
                "max_speed_ms": round(float(speeds.max()), 2),
                "pct_impossible_speed": round(100 * n_impossible / len(speeds), 1),
            }

        # Smoothness — velocidad corregida por frame_diff
        frame_speeds = valid.copy()
        frame_speeds["speed"] = (frame_speeds["dist_meters"] / frame_speeds["frame_diff"]) * video_fps
        accel_vars = []
        for _, group in frame_speeds.groupby("track_id"):
            if len(group) > 2:
                accel_vars.append(group["speed"].diff().dropna().var())
        if accel_vars:
            accel_vars = np.array(accel_vars)
            raw_total = float(frame_speeds["dist_meters"].sum())
            filtered_total = float(frame_speeds.loc[frame_speeds["speed"] >= 0.3, "dist_meters"].sum())
            ratio = raw_total / filtered_total if filtered_total > 0 else float("inf")
            metrics["smoothness"] = {
                "mean_accel_variance": round(float(accel_vars.mean()), 4),
                "raw_distance_m": round(raw_total, 1),
                "filtered_distance_m": round(filtered_total, 1),
                "raw_filtered_ratio": round(ratio, 2),
                "noise_pct": round(100 * (1 - filtered_total / raw_total), 1) if raw_total > 0 else 0,
            }

    metrics["computational_cost"] = {
        "seed": SEED,
        "total_time_s": round(total_time, 2),
        "total_frames": total_frames,
        "fps_processing": round(total_frames / total_time, 2) if total_time > 0 else 0,
        "avg_frame_ms": round(1000 * sum(frame_times) / len(frame_times), 2) if frame_times else 0,
        "peak_ram_mb": round(peak_ram_mb, 1),
        "peak_vram_mb": round(peak_vram_mb, 1),
        "video_fps": video_fps,
    }

    return metrics


def run_controlled_experiment(config, save_artifacts_prefix=None):
    """Ejecuta un experimento controlado de tracking con las banderas de ablación.

    Args:
        config: Diccionario con las banderas de ablación:
            - exp_name: nombre del experimento
            - tracker_yaml: ruta al YAML del tracker ("botsort.yaml" por defecto)
            - conf_thresh: umbral de confianza de detección
            - smooth_window: tamaño de ventana de media móvil (1 = apagado)
            - apply_kinematic_filter: True/False
            - deadband_ms: velocidad mínima para considerar movimiento (m/s)
            - max_speed_ms: velocidad máxima físicamente posible (m/s)
            - max_gap_frames: máximo gap antes de descartar paso
            - scale_factor: factor de conversión píxeles → metros
            - use_perspective_scale: True/False
        save_artifacts_prefix: Si no es None, prefijo para guardar archivos en data/processed/.

    Returns:
        Diccionario de métricas compatible con baseline_metrics.json.
    """
    set_seed(SEED)

    exp_name = config.get("exp_name", "experiment")
    tracker_yaml = config.get("tracker_yaml", "botsort.yaml")
    conf_thresh = config.get("conf_thresh", 0.25)
    smooth_window = config.get("smooth_window", 1)
    apply_filter = config.get("apply_kinematic_filter", False)
    deadband_ms = config.get("deadband_ms", 0.4)
    max_speed_ms = config.get("max_speed_ms", 12.0)
    max_gap_frames = config.get("max_gap_frames", 5)
    scale_factor = config.get("scale_factor", 0.065)
    use_perspective = config.get("use_perspective_scale", False)
    imgsz = config.get("imgsz", 640)

    tracemalloc.start()
    try:
        import torch
        has_cuda = torch.cuda.is_available()
        if has_cuda:
            torch.cuda.reset_peak_memory_stats()
    except ImportError:
        has_cuda = False

    t_start = time.time()

    device = get_device()
    model = YOLO(MODEL_PATH)
    cap = cv2.VideoCapture(VIDEO_INPUT)
    if not cap.isOpened():
        print(f"ERROR: No se pudo abrir el video en: {VIDEO_INPUT}")
        return {}

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    save_video = config.get("save_video", True) and save_artifacts_prefix is not None
    out = None
    if save_video:
        video_path = os.path.join(PROCESSED_DIR, f"{save_artifacts_prefix}_tactical.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(video_path, fourcc, fps, (w, h))

    print(f"\n{'='*60}")
    print(f"Experimento: {exp_name}")
    print(f"Tracker: {tracker_yaml} | Smooth: {smooth_window} | Filter: {apply_filter} | Scale: {scale_factor}")
    print(f"Device: {device} | Video: {w}x{h} @ {fps:.1f} FPS, {total_frames} frames")
    print(f"{'='*60}")

    # Estado del tracking
    track_positions = {}      # tid -> (foot_u, foot_v) última posición compensada
    track_last_frame = {}     # tid -> último frame donde fue detectado
    track_smooth_buffers = {} # tid -> buffer de posiciones para suavizado
    total_distances = {}
    frame_data = []
    frame_times = []
    prev_gray = None

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
        results = model.track(frame, persist=True, classes=[0], conf=conf_thresh,
                              tracker=tracker_yaml, imgsz=imgsz, verbose=False)
        boxes_xyxy = results[0].boxes.xyxy.cpu().numpy() if results[0].boxes.id is not None else np.empty((0, 4))
        ids = results[0].boxes.id.cpu().numpy().astype(int) if results[0].boxes.id is not None else np.empty(0, dtype=int)

        affine = estimate_camera_motion(prev_gray, curr_gray, boxes_xyxy) if prev_gray is not None else None

        for box, tid in zip(boxes_xyxy, ids):
            x1, y1, x2, y2 = box
            foot_u = (x1 + x2) / 2
            foot_v = y2
            current_pos = (foot_u, foot_v)

            dist = 0.0
            frame_diff = frame_num - track_last_frame.get(tid, frame_num)

            if affine is not None and tid in track_positions:
                compensated = compensate_point(track_positions[tid], affine)

                # Suavizado
                buf = track_smooth_buffers.get(tid, [])
                buf, smoothed = smooth_point(buf, compensated, window_size=smooth_window)
                track_smooth_buffers[tid] = buf

                sf = get_perspective_scale(foot_v, h, scale_factor) if use_perspective else scale_factor
                raw_dist = pixels_to_meters(current_pos, smoothed, scale_factor=sf)

                dist, _ = filter_step_distance(
                    raw_dist, frame_diff, fps,
                    apply_filter=apply_filter,
                    deadband_ms=deadband_ms,
                    max_speed_ms=max_speed_ms,
                    max_gap_frames=max_gap_frames,
                )
                total_distances[tid] = total_distances.get(tid, 0.0) + dist

            track_positions[tid] = current_pos
            track_last_frame[tid] = frame_num
            frame_data.append((frame_num, tid, foot_u, foot_v, dist, frame_diff))

            if save_video:
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                label = f"ID:{tid} {total_distances.get(tid, 0.0):.1f}m"
                cv2.putText(frame, label, (int(x1), int(y1) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        prev_gray = curr_gray
        if out is not None:
            out.write(frame)
        frame_times.append(time.time() - t_frame)

    cap.release()
    if out is not None:
        out.release()

    total_time = time.time() - t_start
    peak_ram_mb = tracemalloc.get_traced_memory()[1] / (1024 * 1024)
    tracemalloc.stop()

    peak_vram_mb = 0.0
    if has_cuda:
        import torch
        peak_vram_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)

    metrics = _compute_metrics(
        total_distances, frame_data, frame_times, fps,
        total_time, peak_ram_mb, peak_vram_mb, frame_num, config,
    )

    # Guardar artefactos
    if save_artifacts_prefix:
        csv_path = os.path.join(PROCESSED_DIR, f"{save_artifacts_prefix}_distances.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Track_ID", "Distancia_Metros"])
            for tid in sorted(total_distances):
                writer.writerow([tid, f"{total_distances[tid]:.2f}"])

        frame_csv_path = os.path.join(PROCESSED_DIR, f"{save_artifacts_prefix}_frame_data.csv")
        with open(frame_csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["frame", "track_id", "foot_u", "foot_v", "dist_meters", "frame_diff"])
            for row in frame_data:
                writer.writerow([row[0], row[1], f"{row[2]:.2f}", f"{row[3]:.2f}", f"{row[4]:.4f}", row[5]])

        metrics_path = os.path.join(PROCESSED_DIR, f"{save_artifacts_prefix}_metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)

        print(f"\nArtefactos guardados con prefijo: {save_artifacts_prefix}")

    print(f"[{exp_name}] Tracks: {len(total_distances)} | Tiempo: {total_time:.1f}s ({frame_num / total_time:.1f} FPS)")
    return metrics
