"""Etapa 2B: Estudio de Ablación — 3 Experimentos Controlados Incrementales."""

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.features.projection import calibrate_scale_factor
from src.models.detector import create_bytetrack_yaml, create_ocsort_yaml
from src.models.tracker import run_controlled_experiment

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "processed")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "models", "fine_tuned")


def _safe_get(d, *keys, default="N/A"):
    current = d
    for k in keys:
        if isinstance(current, dict):
            current = current.get(k, default)
        else:
            return default
    return current


def print_comparison_table(results):
    """Imprime tabla comparativa de ablación en consola."""
    keys = ["baseline", "exp1", "exp2", "exp3", "exp4", "exp5"]
    labels = ["Baseline", "Exp 1", "Exp 2", "Exp 3", "Exp 4", "Exp 5"]
    available = [(k, l) for k, l in zip(keys, labels) if k in results]
    keys = [a[0] for a in available]
    labels = [a[1] for a in available]

    rows = [
        ("IDs únicos", "tracking", "n_unique_ids"),
        ("Fragmentación (x)", "tracking", "fragmentation_ratio"),
        ("ID switches est.", "tracking", "id_switches_estimated"),
        ("Long. media track (fr)", "tracking", "avg_track_length_frames"),
        ("Vel. máxima (m/s)", "kinematics", "max_speed_ms"),
        ("% vel. imposible", "kinematics", "pct_impossible_speed"),
        ("Var. aceleración", "smoothness", "mean_accel_variance"),
        ("% ruido (jitter)", "smoothness", "noise_pct"),
        ("Distancia total (m)", "distances", "total_m"),
        ("Dist/jugador (m)", "distances", "estimated_per_player_m"),
        ("FPS procesamiento", "computational_cost", "fps_processing"),
    ]

    col_w = 16
    header = f"{'Métrica':<25}" + "".join(f"{l:>{col_w}}" for l in labels)
    sep = "=" * len(header)

    print(f"\n{sep}")
    print("TABLA COMPARATIVA — ESTUDIO DE ABLACIÓN")
    print(sep)
    print(header)
    print("-" * len(header))

    for label, *path in rows:
        vals = []
        for k in keys:
            v = _safe_get(results[k], *path)
            if isinstance(v, float):
                vals.append(f"{v:>{col_w}.2f}")
            else:
                vals.append(f"{str(v):>{col_w}}")
        print(f"{label:<25}" + "".join(vals))

    # Extrapolación a 90 min
    print("-" * len(header))
    extrap_vals = []
    for k in keys:
        dist = _safe_get(results[k], "distances", "estimated_per_player_m", default=0)
        cost = _safe_get(results[k], "computational_cost", default={})
        total_frames = cost.get("total_frames", 0) if isinstance(cost, dict) else 0
        video_fps = cost.get("video_fps", 30) if isinstance(cost, dict) else 30
        duration_min = (total_frames / video_fps / 60) if total_frames > 0 and video_fps > 0 else 1
        km_90 = (dist / duration_min * 90 / 1000) if isinstance(dist, (int, float)) else "N/A"
        if isinstance(km_90, float):
            extrap_vals.append(f"{km_90:>{col_w}.1f}")
        else:
            extrap_vals.append(f"{str(km_90):>{col_w}}")
    print(f"{'Extrap. 90min (km)':<25}" + "".join(extrap_vals))
    print(sep)


def main(only=None):
    """
    Args:
        only: lista de prefijos a ejecutar, p.ej. ["exp5"].
              None = ejecutar todos.
    """
    # =================================================================
    # 1. Preparación de artefactos de configuración
    # =================================================================

    # ByteTrack: dos pasadas de asociación (alta+baja confianza) para recuperar IDs perdidos
    # conf_thresh en los configs debe ser igual a track_low_thresh (0.10) para que
    # ByteTrack reciba las detecciones débiles y las use en la 2ª pasada
    tuned_yaml = create_bytetrack_yaml(
        output_path=os.path.join(MODELS_DIR, "bytetrack_tuned.yaml"),
        track_high_thresh=0.25,  # primera asociación: detecciones sólidas
        track_low_thresh=0.10,   # segunda asociación: detecciones débiles (ocluidas)
        new_track_thresh=0.60,   # umbral para crear track nuevo (evita falsos positivos)
        track_buffer=450,        # 15 seg — mantiene IDs perdidos vivos más tiempo
        match_thresh=0.80,       # IoU para matching
    )
    print(f"ByteTrack tuneado generado: {tuned_yaml}")

    ocsort_yaml = create_ocsort_yaml(
        output_path=os.path.join(MODELS_DIR, "ocsort_tuned.yaml"),
        track_high_thresh=0.25,
        track_low_thresh=0.10,
        new_track_thresh=0.60,
        track_buffer=600,  # 20 seg — reduce IDs nuevos cuando jugadora sale de cuadro
        match_thresh=0.80,
        delta_t=5,    # ventana de velocidad observada (~0.17 seg a 30fps)
        inertia=0.3,  # más peso a velocidad observada que Kalman puro
        use_byte=True,
    )
    print(f"OC-SORT tuneado generado: {ocsort_yaml}")

    # Calibrar scale_factor usando solo medidas horizontales (resolución 1364x764)
    # La medida vertical (profundidad área penal) se excluye: perspectiva la distorsiona
    ref_points_px = [
        ((335, 265), (1020, 265)),  # ancho área penal: 40.32 m → 685 px (horizontal)
        ((575, 345), (855, 345)),   # diámetro círculo central: 18.30 m → 280 px (horizontal)
    ]
    real_distances_m = [40.32, 18.30]

    calibrated_scale = calibrate_scale_factor(ref_points_px, real_distances_m)
    print(f"Scale factor calibrado: {calibrated_scale:.4f} m/px (baseline: 0.065)")

    # =================================================================
    # 2. Grilla de ablación (3 experimentos incrementales)
    # =================================================================

    # Polígono de cancha (1364x764): excluye graderías superiores y bordes laterales
    # y=125 (antes 145) para capturar jugadoras de la línea del fondo
    field_polygon = [(60, 125), (1300, 125), (1364, 600), (1364, 764), (0, 764), (0, 600)]

    exp1_config = {
        "exp_name": "Exp 1: ByteTrack Tuneado",
        "tracker_yaml": tuned_yaml,
        "model_path": "yolov8s.pt",
        "conf_thresh": 0.10,
        "imgsz": 1536,          # resolución mayor → mejor detección jugadoras lejanas
        "smooth_window": 1,
        "apply_kinematic_filter": False,
        "scale_factor": 0.065,
        "use_perspective_scale": False,
        "field_polygon": field_polygon,
        "save_video": False,
    }

    exp2_config = {
        "exp_name": "Exp 2: + Filtro Anti-Jitter",
        "tracker_yaml": tuned_yaml,
        "model_path": "yolov8s.pt",
        "conf_thresh": 0.10,
        "imgsz": 1536,
        "smooth_window": 9,
        "apply_kinematic_filter": True,
        "deadband_ms": 0.4,
        "max_speed_ms": 12.0,
        "max_gap_frames": 9,
        "scale_factor": 0.065,
        "use_perspective_scale": False,
        "field_polygon": field_polygon,
        "save_video": False,
    }

    exp3_config = {
        "exp_name": "Exp 3: + Scale Factor Calibrado",
        "tracker_yaml": tuned_yaml,
        "model_path": "yolov8s.pt",
        "conf_thresh": 0.10,
        "imgsz": 1536,
        "smooth_window": 9,
        "apply_kinematic_filter": True,
        "deadband_ms": 0.4,
        "max_speed_ms": 12.0,
        "max_gap_frames": 9,
        "scale_factor": calibrated_scale,
        "use_perspective_scale": True,
        "field_polygon": field_polygon,
        "save_video": False,
    }

    exp4_config = {
        "exp_name": "Exp 4: + Color Clustering + Track Merge",
        "tracker_yaml": tuned_yaml,
        "model_path": "yolov8s.pt",
        "conf_thresh": 0.10,
        "imgsz": 1536,
        "smooth_window": 9,
        "apply_kinematic_filter": True,
        "deadband_ms": 1.0,
        "max_speed_ms": 12.0,
        "max_gap_frames": 9,
        "scale_factor": calibrated_scale,
        "use_perspective_scale": True,
        "field_polygon": field_polygon,
        "apply_color_clustering": True,
        "apply_track_merge": True,
        "merge_max_gap_frames": 450,   # 15 seg (= track_buffer) — fusiona fragmentos lejanos
        "merge_max_dist_px": 300.0,    # tolerancia espacial ampliada (+ predicción por velocidad)
        "save_video": True,
    }

    # =================================================================
    # 3. Ejecución y registro consolidado
    # =================================================================

    # Cargar baseline como referencia
    baseline_path = os.path.join(PROCESSED_DIR, "baseline_metrics.json")
    if not os.path.exists(baseline_path):
        print(f"ERROR: No se encontró {baseline_path}")
        print("Ejecute primero: python src/models/baseline_tracker.py")
        print("Luego el notebook: notebooks/03_baseline_metrics.ipynb")
        return

    with open(baseline_path) as f:
        baseline = json.load(f)
    print(f"\nBaseline cargado: {baseline_path}")

    results = {"baseline": baseline}

    exp5_config = {
        "exp_name": "Exp 5: OC-SORT + YOLO11s",
        "tracker_yaml": ocsort_yaml,
        "model_path": "yolo11s.pt",       # detector más moderno, mejor recall
        "conf_thresh": 0.10,
        "imgsz": 1536,
        "smooth_window": 9,
        "apply_kinematic_filter": True,
        "deadband_ms": 1.5,    # 1.0→1.5: filtra residual GMC (~1.2 m/s) sin eliminar trote lento
        "max_speed_ms": 12.0,
        "max_gap_frames": 9,
        "scale_factor": calibrated_scale,
        "use_perspective_scale": True,
        "field_polygon": field_polygon,
        "apply_color_clustering": True,
        "apply_track_merge": True,
        "merge_max_gap_frames": 450,
        "merge_max_dist_px": 300.0,
        "save_video": True,
    }

    all_experiments = [
        ("exp1", exp1_config),
        ("exp2", exp2_config),
        ("exp3", exp3_config),
        ("exp4", exp4_config),
        ("exp5", exp5_config),
    ]

    # Filtrar experimentos si se especificó --only / only=
    if only:
        run_set = set(only)
        experiments_to_run = [(p, c) for p, c in all_experiments if p in run_set]
        # Cargar resultados previos de los que se omiten
        for prefix, _ in all_experiments:
            if prefix not in run_set:
                cached = os.path.join(PROCESSED_DIR, f"{prefix}_metrics.json")
                if os.path.exists(cached):
                    with open(cached) as f:
                        results[prefix] = json.load(f)
                    print(f"[{prefix}] cargado de caché: {cached}")
                else:
                    print(f"[{prefix}] sin caché — omitido de la tabla")
    else:
        experiments_to_run = all_experiments

    for prefix, config in experiments_to_run:
        metrics = run_controlled_experiment(config, save_artifacts_prefix=prefix)
        results[prefix] = metrics

    # Consolidar resultados
    ablation_path = os.path.join(PROCESSED_DIR, "ablation_results.json")
    with open(ablation_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResultados consolidados en: {ablation_path}")

    # =================================================================
    # 4. Tabla comparativa en consola
    # =================================================================

    print_comparison_table(results)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--only", nargs="+", metavar="EXP",
        help="Prefijos a ejecutar, p.ej. --only exp5  o  --only exp4 exp5"
    )
    args = parser.parse_args()
    main(only=args.only)
