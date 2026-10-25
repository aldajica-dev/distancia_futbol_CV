"""
Experimentos 5, 6 y 7 del estudio de ablación (GPU recomendado).

Uso:
  python src/models/gpu_tracker.py           # corre Exp 5, 6 y 7 en secuencia
  python src/models/gpu_tracker.py --exp 5   # solo Exp 5: OC-SORT + YOLO11s
  python src/models/gpu_tracker.py --exp 6   # solo Exp 6: BoT-SORT + YOLO11s + ReID
  python src/models/gpu_tracker.py --exp 7   # solo Exp 7: YOLO fine-tuned + ReID

Requisitos:
  - data/raw/video.mp4 debe existir.
  - GPU con ≥ 4 GB VRAM recomendada.  CPU funciona pero es lento (~10× más).
  - PyTorch con CUDA instalado (ver requirements.txt).

Salidas en data/processed/:
  exp5_distances.csv, exp5_frame_data.csv, exp5_metrics.json, exp5_tactical.mp4
  exp6_distances.csv, exp6_frame_data.csv, exp6_metrics.json, exp6_tactical.mp4
  exp7_distances.csv, exp7_frame_data.csv, exp7_metrics.json, exp7_tactical.mp4
  ablation_results.json (actualizado con Exp 5-7, preserva Exp 1-4 existentes)
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.features.projection import calibrate_scale_factor
from src.models.detector import (
    create_botsort_yaml,
    create_ocsort_yaml,
    get_device,
)
from src.models.tracker import run_controlled_experiment

ROOT_DIR    = os.path.join(os.path.dirname(__file__), "..", "..")
VIDEO_INPUT = os.path.join(ROOT_DIR, "data", "raw", "video.mp4")
PROCESSED   = os.path.join(ROOT_DIR, "data", "processed")
MODELS_DIR  = os.path.join(ROOT_DIR, "models", "fine_tuned")
INTERIM_DIR = os.path.join(ROOT_DIR, "data", "interim", "pseudo_labels")


def _check_video():
    path = os.path.abspath(VIDEO_INPUT)
    if not os.path.exists(path):
        print("\n" + "=" * 60)
        print("ERROR: video no encontrado.")
        print(f"  Ruta esperada: {path}")
        print("  Coloca el video de entrada en esa ruta y vuelve a ejecutar.")
        print("=" * 60)
        sys.exit(1)
    return path


def _load_existing_results():
    """Carga ablation_results.json si existe; retorna dict vacío si no."""
    path = os.path.join(PROCESSED, "ablation_results.json")
    if os.path.exists(path):
        with open(path) as f:
            data = json.load(f)
        print(f"Resultados previos cargados: {path}")
        return data
    return {}


def _save_results(results):
    path = os.path.join(PROCESSED, "ablation_results.json")
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResultados guardados en: {path}")


def _calibrated_scale():
    ref_points_px = [
        ((335, 265), (1020, 265)),   # ancho área penal: 40.32 m → 685 px
        ((575, 345), (855, 345)),    # diámetro círculo: 18.30 m → 280 px
    ]
    real_distances_m = [40.32, 18.30]
    scale = calibrate_scale_factor(ref_points_px, real_distances_m)
    print(f"Scale factor calibrado: {scale:.4f} m/px")
    return scale


def _field_polygon():
    return [(60, 125), (1300, 125), (1364, 600), (1364, 764), (0, 764), (0, 600)]


# ── Pseudo-etiquetado ─────────────────────────────────────────────────────────

def generate_pseudo_labels(video_path, output_dir, conf=0.50, min_dets=4):
    """Genera pseudo-etiquetas YOLO desde detecciones de alta confianza.

    Corre YOLO11s sobre el video con conf=0.50 y guarda los frames donde
    se detectan >= min_dets jugadoras. Las etiquetas se guardan en formato
    YOLO (clase cx cy w h normalizados). Retorna la ruta al dataset.yaml.

    Args:
        video_path: Ruta al video fuente.
        output_dir: Directorio base donde crear images/ y labels/.
        conf: Umbral de confianza mínima para generar pseudo-etiqueta.
        min_dets: Número mínimo de detecciones por frame para incluirlo.

    Returns:
        Ruta al dataset.yaml para usar en model.train().
    """
    import cv2
    import numpy as np
    from ultralytics import YOLO

    imgs_dir   = os.path.join(output_dir, "images", "train")
    lbls_dir   = os.path.join(output_dir, "labels", "train")
    val_imgs   = os.path.join(output_dir, "images", "val")
    val_lbls   = os.path.join(output_dir, "labels", "val")
    for d in (imgs_dir, lbls_dir, val_imgs, val_lbls):
        os.makedirs(d, exist_ok=True)

    model = YOLO("yolo11s.pt")
    cap   = cv2.VideoCapture(video_path)
    w     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"\nGenerando pseudo-etiquetas (conf>{conf}, min_dets={min_dets})...")
    saved_frames = []
    frame_num = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_num += 1

        results = model(frame, classes=[0], conf=conf, imgsz=1280, verbose=False)
        boxes = results[0].boxes.xyxy.cpu().numpy() if len(results[0].boxes) > 0 else []
        if len(boxes) < min_dets:
            continue

        img_name = f"frame_{frame_num:05d}.jpg"
        cv2.imwrite(os.path.join(imgs_dir, img_name), frame)

        lbl_lines = []
        for box in boxes:
            x1, y1, x2, y2 = box
            cx = ((x1 + x2) / 2) / w
            cy = ((y1 + y2) / 2) / h
            bw = (x2 - x1) / w
            bh = (y2 - y1) / h
            lbl_lines.append(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

        lbl_name = img_name.replace(".jpg", ".txt")
        with open(os.path.join(lbls_dir, lbl_name), "w") as f:
            f.write("\n".join(lbl_lines))

        saved_frames.append(img_name)
        if frame_num % 100 == 0:
            print(f"  Frame {frame_num}/{total} — guardados: {len(saved_frames)}")

    cap.release()

    # Split 80/20 train/val
    import random
    random.seed(42)
    n_val = max(1, len(saved_frames) // 5)
    val_set = set(random.sample(saved_frames, n_val))
    moved = 0
    for name in val_set:
        src_img = os.path.join(imgs_dir, name)
        src_lbl = os.path.join(lbls_dir, name.replace(".jpg", ".txt"))
        if os.path.exists(src_img):
            os.replace(src_img, os.path.join(val_imgs, name))
            os.replace(src_lbl, os.path.join(val_lbls, name.replace(".jpg", ".txt")))
            moved += 1

    n_train = len(saved_frames) - moved
    print(f"  Total frames guardados: {len(saved_frames)} | train: {n_train} | val: {moved}")

    dataset_yaml = os.path.join(output_dir, "dataset.yaml")
    with open(dataset_yaml, "w") as f:
        f.write(f"path: {os.path.abspath(output_dir)}\n")
        f.write("train: images/train\n")
        f.write("val:   images/val\n")
        f.write("nc: 1\n")
        f.write("names: ['person']\n")

    print(f"Dataset YAML creado: {dataset_yaml}")
    return dataset_yaml


def finetune_yolo(dataset_yaml, base_model="yolo11s.pt", epochs=30):
    """Fine-tunea YOLO sobre el dataset de pseudo-etiquetas.

    Args:
        dataset_yaml: Ruta al dataset.yaml generado por generate_pseudo_labels().
        base_model: Modelo base a fine-tunear.
        epochs: Número de épocas de entrenamiento.

    Returns:
        Ruta al modelo fine-tuneado (best.pt copiado a models/fine_tuned/).
    """
    import shutil
    from ultralytics import YOLO

    device = get_device()
    print(f"\nFine-tuning {base_model} por {epochs} épocas en {device}...")

    model = YOLO(base_model)
    model.train(
        data=dataset_yaml,
        epochs=epochs,
        imgsz=640,
        device=device,
        patience=10,
        batch=16 if device != "cpu" else 4,
        project="runs/train",
        name="yolo_finetuned",
        exist_ok=True,
        verbose=False,
    )

    best_pt = os.path.join("runs", "train", "yolo_finetuned", "weights", "best.pt")
    if not os.path.exists(best_pt):
        # Ultralytics puede generar el path de forma distinta; buscar recursivo
        import glob
        candidates = glob.glob(os.path.join("runs", "**", "best.pt"), recursive=True)
        if candidates:
            best_pt = candidates[-1]
        else:
            print("ADVERTENCIA: no se encontró best.pt; usando el modelo base.")
            return base_model

    dest = os.path.join(MODELS_DIR, "yolo11s_finetuned.pt")
    os.makedirs(MODELS_DIR, exist_ok=True)
    shutil.copy2(best_pt, dest)
    print(f"Modelo fine-tuneado guardado en: {dest}")
    return dest


# ── Configuración de experimentos ─────────────────────────────────────────────

def build_configs(video_path):
    scale      = _calibrated_scale()
    polygon    = _field_polygon()

    ocsort_yaml = create_ocsort_yaml(
        output_path=os.path.join(MODELS_DIR, "ocsort_gpu.yaml"),
        track_high_thresh=0.25,
        track_low_thresh=0.10,
        new_track_thresh=0.60,
        track_buffer=600,
        match_thresh=0.80,
        delta_t=5,
        inertia=0.3,
        use_byte=True,
    )

    botsort_reid_yaml = create_botsort_yaml(
        output_path=os.path.join(MODELS_DIR, "botsort_reid.yaml"),
        track_high_thresh=0.25,
        track_low_thresh=0.10,
        new_track_thresh=0.60,
        track_buffer=600,
        match_thresh=0.80,
        gmc_method="sparseOptFlow",
        with_reid=True,
        appearance_thresh=0.25,
    )

    base_cfg = {
        "video_path": video_path,
        "conf_thresh": 0.10,
        "imgsz": 1536,
        "smooth_window": 9,
        "apply_kinematic_filter": True,
        "deadband_ms": 1.5,
        "max_speed_ms": 12.0,
        "max_gap_frames": 9,
        "scale_factor": scale,
        "use_perspective_scale": True,
        "field_polygon": polygon,
        "apply_color_clustering": True,
        "apply_track_merge": True,
        "merge_max_gap_frames": 450,
        "merge_max_dist_px": 300.0,
        "use_homography_gmc": True,
        "save_video": True,
    }

    exp5 = {**base_cfg,
        "exp_name": "Exp 5: OC-SORT + YOLO11s",
        "tracker_yaml": ocsort_yaml,
        "model_path": "yolo11s.pt",
    }
    exp6 = {**base_cfg,
        "exp_name": "Exp 6: BoT-SORT + YOLO11s + ReID (OSNet)",
        "tracker_yaml": botsort_reid_yaml,
        "model_path": "yolo11s.pt",
    }

    return ocsort_yaml, botsort_reid_yaml, exp5, exp6


# ── Main ──────────────────────────────────────────────────────────────────────

def main(exps_to_run):
    video_path = _check_video()
    results = _load_existing_results()

    _, botsort_reid_yaml, exp5_cfg, exp6_cfg = build_configs(video_path)

    run_map = {
        "exp5": ("exp5", exp5_cfg),
        "exp6": ("exp6", exp6_cfg),
    }

    for key in exps_to_run:
        if key == "exp7":
            # Exp 7 requiere pseudo-etiquetado + fine-tuning antes del tracking
            finetuned_path = os.path.join(MODELS_DIR, "yolo11s_finetuned.pt")
            if not os.path.exists(finetuned_path):
                dataset_yaml = generate_pseudo_labels(
                    video_path, output_dir=INTERIM_DIR, conf=0.50, min_dets=4,
                )
                finetuned_path = finetune_yolo(dataset_yaml, base_model="yolo11s.pt", epochs=30)
            else:
                print(f"\n[Exp 7] Modelo fine-tuneado encontrado en {finetuned_path} — omitiendo reentrenamiento.")

            exp7_cfg = {
                **build_configs(video_path)[2],   # hereda base de exp5
                "exp_name": "Exp 7: YOLO fine-tuned + BoT-SORT + ReID",
                "tracker_yaml": botsort_reid_yaml,
                "model_path": finetuned_path,
            }
            metrics = run_controlled_experiment(exp7_cfg, save_artifacts_prefix="exp7")
            results["exp7"] = metrics

        elif key in run_map:
            prefix, cfg = run_map[key]
            metrics = run_controlled_experiment(cfg, save_artifacts_prefix=prefix)
            results[prefix] = metrics

    _save_results(results)

    # Tabla comparativa en consola
    keys_order = ["baseline", "exp1", "exp2", "exp3", "exp4", "exp5", "exp6", "exp7"]
    present    = [k for k in keys_order if k in results]
    print(f"\n{'='*60}")
    print("RESULTADOS ACUMULADOS")
    print(f"{'='*60}")
    for k in present:
        r = results[k]
        phase = r.get("phase", k)
        ids   = r.get("tracking", {}).get("n_unique_ids", "?")
        frag  = r.get("tracking", {}).get("fragmentation_ratio", "?")
        tmed  = r.get("tracking", {}).get("median_track_length_frames", "?")
        dist  = r.get("distances",  {}).get("estimated_per_player_m", "?")
        fps   = r.get("computational_cost", {}).get("fps_processing", "?")
        print(f"  {phase[:40]:<40} IDs:{ids:>4}  frag:{frag:>4}×  "
              f"tmed:{tmed:>5}fr  dist:{dist:>6.1f}m  fps:{fps:>5.1f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Experimentos 5-7 del pipeline de análisis cinemático de fútbol."
    )
    parser.add_argument(
        "--exp",
        choices=["5", "6", "7"],
        help="Experimento a correr (5, 6 o 7). Sin argumento: corre los 3.",
    )
    args = parser.parse_args()

    if args.exp:
        exps = [f"exp{args.exp}"]
    else:
        exps = ["exp5", "exp6", "exp7"]

    main(exps)
