"""Etapa 2B: Estudio de Ablación — 3 Experimentos Controlados Incrementales."""

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.features.projection import calibrate_scale_factor
from src.models.detector import create_botsort_yaml
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
    keys = ["baseline", "exp1", "exp2", "exp3"]
    labels = ["Baseline", "Exp 1", "Exp 2", "Exp 3"]
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


def main():
    # =================================================================
    # 1. Preparación de artefactos de configuración
    # =================================================================

    # BoT-SORT tuneado para reducir fragmentación
    tuned_yaml = create_botsort_yaml(
        output_path=os.path.join(MODELS_DIR, "botsort_tuned.yaml"),
        track_high_thresh=0.40,
        track_low_thresh=0.10,
        new_track_thresh=0.58,
        track_buffer=120,
        match_thresh=0.80,
        gmc_method="sparseOptFlow",
    )
    print(f"BoT-SORT tuneado generado: {tuned_yaml}")

    # Calibrar scale_factor con referencias FIFA sobre resolución 1548x898
    ref_points_px = [
        ((324, 430), (1224, 430)),  # ancho área penal: 40.32 m → ~900 px
        ((540, 280), (540, 645)),   # largo área penal: 16.50 m → ~365 px
        ((572, 350), (977, 350)),   # diámetro círculo central: 18.30 m → ~405 px
    ]
    real_distances_m = [40.32, 16.50, 18.30]

    calibrated_scale = calibrate_scale_factor(ref_points_px, real_distances_m)
    print(f"Scale factor calibrado: {calibrated_scale:.4f} m/px (baseline: 0.065)")

    # =================================================================
    # 2. Grilla de ablación (3 experimentos incrementales)
    # =================================================================

    exp1_config = {
        "exp_name": "Exp 1: BoT-SORT Tuneado",
        "tracker_yaml": tuned_yaml,
        "conf_thresh": 0.35,
        "imgsz": 1280,
        "smooth_window": 1,
        "apply_kinematic_filter": False,
        "scale_factor": 0.065,
        "use_perspective_scale": False,
        "save_video": False,
    }

    exp2_config = {
        "exp_name": "Exp 2: + Filtro Anti-Jitter",
        "tracker_yaml": tuned_yaml,
        "conf_thresh": 0.35,
        "imgsz": 1280,
        "smooth_window": 5,
        "apply_kinematic_filter": True,
        "deadband_ms": 0.4,
        "max_speed_ms": 12.0,
        "max_gap_frames": 5,
        "scale_factor": 0.065,
        "use_perspective_scale": False,
        "save_video": False,
    }

    exp3_config = {
        "exp_name": "Exp 3: Pipeline Avanzado Completo",
        "tracker_yaml": tuned_yaml,
        "conf_thresh": 0.35,
        "imgsz": 1280,
        "smooth_window": 5,
        "apply_kinematic_filter": True,
        "deadband_ms": 0.4,
        "max_speed_ms": 12.0,
        "max_gap_frames": 5,
        "scale_factor": calibrated_scale,
        "use_perspective_scale": True,
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

    for prefix, config in [("exp1", exp1_config), ("exp2", exp2_config), ("advanced", exp3_config)]:
        metrics = run_controlled_experiment(config, save_artifacts_prefix=prefix)
        key = prefix if prefix != "advanced" else "exp3"
        results[key] = metrics

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
    main()
