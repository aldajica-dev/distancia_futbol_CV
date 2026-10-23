# CLAUDE.md - Contexto del Proyecto

## Proyecto
**Análisis Cinemático de Fútbol** — Estimar distancias recorridas por jugadores a partir de video de cámara táctica dinámica usando YOLO + BoT-SORT + GMC.

## Contexto académico
Trabajo final de la materia "Técnicas Avanzadas de Modelado IA" en la Maestría de la Universidad de la Sabana.

## Arquitectura del proyecto

```
├── data/raw/            → Videos originales (video.mp4)
├── data/interim/        → Datos intermedios
├── data/processed/      → CSVs, JSONs de métricas, videos anotados
├── models/homographies/ → Matrices H (.npy)
├── models/fine_tuned/   → botsort_tuned.yaml (BoT-SORT tuneado)
├── notebooks/           → Notebooks de exploración y homografía
├── src/features/        → projection.py (GMC + calibración) + kinematics.py (filtros)
├── src/models/          → baseline_tracker.py, tracker.py, detector.py, advanced_tracker.py
├── src/visualization/   → annotator.py (Fase 2)
├── reports/             → Documentación y reportes (.docx)
```

## Paradigma de desarrollo
- **Funcional**: funciones puras, sin clases (OOP) salvo requerimiento técnico.
- **"Menos es más"**: código mínimo funcional, sin validaciones excesivas ni abstracciones prematuras.
- Entorno conda: `futbol` (Python 3.10.22).

## Bitácora de progreso

### Fase 1 — Línea Base (completada)
1. **Estructura creada**: todos los directorios y archivos según el árbol definido.
2. **Configuración raíz**:
   - `requirements.txt`: ultralytics, opencv-python, numpy, jupyter.
   - `.gitignore`: ignora __pycache__/, .mp4, .avi, .pt, data/raw/*, data/interim/*.
   - `README.md`: descripción, alcance e instalación.
3. **`src/features/projection.py`** (funcional, refactorizado para cámara táctica):
   - `estimate_camera_motion(prev_gray, curr_gray, boxes)` — flujo óptico sparse con máscara de exclusión de jugadores + estimateAffinePartial2D (RANSAC) para obtener la transformación afín de la cámara.
   - `compensate_point(point, affine_matrix)` — proyecta punto del frame anterior al actual cancelando el paneo de cámara.
   - `pixels_to_meters(p1, p2, scale_factor)` — distancia euclidiana en píxeles × factor de escala → metros.
4. **`src/models/baseline_tracker.py`** (funcional, refactorizado):
   - Carga YOLOv8n con tracker BoT-SORT (`botsort.yaml`).
   - Extrae punto base (centro inferior del bbox) como contacto con suelo.
   - Compensa movimiento de cámara frame a frame con GMC.
   - Acumula distancia cruda (sin filtros anti-jitter) por cada track_id.
   - Dibuja cajas, ID y distancia acumulada en el video.
   - Guarda video en `data/processed/baseline_tactical.mp4`.
   - Exporta CSV en `data/processed/baseline_distances.csv`.
   - Indicador de progreso en consola (cada 50 frames).
5. **`notebooks/02_homography_builder.ipynb`** (funcional):
   - Define 4 correspondencias imagen↔cancha, calcula H con cv2.findHomography.
   - Guarda la matriz en `models/homographies/H_matrix.npy`.
6. **Archivos placeholder** (`# TODO: Implementar en Fase 2`):
   - video_loader.py, kinematics.py, detector.py, tracker.py, annotator.py.
   - Notebooks 01 (EDA) y 03 (métricas).
7. **Primera ejecución**:
   - Dependencias instaladas correctamente en entorno conda `futbol`.
   - YOLOv8n descargado (v8.4.0), dependencia `lap` auto-instalada.
8. **Ejecución exitosa sobre video real** — resultados en `baseline_distances.csv`:
   - ~460 track IDs generados (debería haber ~22 jugadores + árbitros).
   - **Problema principal: fragmentación masiva de IDs** — BoT-SORT pierde y reasigna identidades constantemente.
   - ~150 IDs con < 0.1 m (tracks fantasma de 1-2 frames).
   - ~170 IDs entre 0.1–1.0 m (fragmentos muy cortos).
   - ~90 IDs entre 1.0–5.0 m (fragmentos parciales).
   - ~30 IDs entre 5.0–10.0 m (tracks más estables).
   - ~10 IDs con > 10.0 m (mejores tracks; máximo: ID 824 con 15.38 m).
   - **Interpretación**: la GMC funciona (valores no explotan), pero las distancias son muy bajas porque la distancia real de cada jugador está repartida entre múltiples IDs fragmentados. La Línea Base cumple su rol como punto de referencia.

### Métricas de evaluación (implementadas)
- **`baseline_tracker.py`** actualizado: exporta `baseline_frame_data.csv` (datos por frame) y `baseline_timing.json` (costo computacional).
- **`03_baseline_metrics.ipynb`** implementado con 7 secciones:
  1. Calidad del tracking (IDs únicos, fragmentación, tracks fantasma, longitud de tracks).
  2. ID switches estimados (nuevos IDs por segundo, acumulados vs esperados).
  3. Plausibilidad cinemática (distribución de velocidades, % imposibles >12 m/s).
  4. Distribución de distancias (histograma, top tracks, extrapolación a 90 min vs literatura 8–13 km).
  5. Suavidad de trayectoria (varianza de aceleración, ratio cruda/filtrada con deadband).
  6. Costo computacional (tiempo total, FPS, RAM, VRAM).
  7. Tabla resumen + exportación a `baseline_metrics.json` (para comparar con Fase 2).
- Dependencias agregadas: `matplotlib`, `pandas`, `psutil`.

### Fase 2 — Etapa 2A: Módulos Core (implementada)
**Diseño de ablación con 3 experimentos incrementales:**
- Baseline (Fase 1): BoT-SORT por defecto + sin filtro + scale_factor=0.065 fijo.
- Exp 1 (Tracker Tuneado): Hiperparámetros BoT-SORT ajustados para reducir fragmentación.
- Exp 2 (+ Filtro Anti-Jitter): Exp 1 + media móvil + deadband + control de frame_diff.
- Exp 3 (+ Scale Factor Calibrado): Exp 2 + calibración empírica con medidas FIFA.

**Módulos implementados:**
1. **`src/features/projection.py`** — Agregadas `calibrate_scale_factor()` y `get_perspective_scale()` sin romper compatibilidad.
2. **`src/features/kinematics.py`** — `smooth_point()` (media móvil componible) + `filter_step_distance()` (deadband, max_speed, max_gap con flag `apply_filter` para aislar variables).
3. **`src/models/detector.py`** — `get_device()` (cuda>mps>cpu) + `create_botsort_yaml()` (genera YAML dinámico en `models/fine_tuned/`).
4. **`src/models/tracker.py`** — `run_controlled_experiment(config)` motor de evaluación modular que recibe banderas de ablación, ejecuta el bucle de tracking y retorna métricas compatibles con `baseline_metrics.json`.

**Dependencia agregada:** `pyyaml`.

### Fase 2 — Etapa 2B: Experimentos (completada)
**`src/models/advanced_tracker.py`** — Orquestador del estudio de ablación con 3 experimentos incrementales.

**Hiperparámetros BoT-SORT tuneados (finales):**
- `track_high_thresh=0.40`, `track_low_thresh=0.10`, `new_track_thresh=0.58`
- `track_buffer=120`, `match_thresh=0.80`, `conf_thresh=0.35`
- `imgsz=1280` (resolución de inferencia YOLO aumentada)

**Calibración del scale factor con referencias FIFA (resolución 1548x898):**
- Ancho área penal: (324,430)→(1224,430) = 900 px → 40.32 m
- Largo área penal: (540,280)→(540,645) = 365 px → 16.50 m
- Diámetro círculo central: (572,350)→(977,350) = 405 px → 18.30 m
- **Scale factor calibrado: 0.0451 m/px** (vs 0.065 de la Línea Base)

**Resultados del estudio de ablación:**

| Métrica | Baseline | Exp 1 | Exp 2 | Exp 3 |
|---|---|---|---|---|
| IDs únicos | 470 | 300 | 300 | 300 |
| Fragmentación (x) | 21.4 | 13.6 | 13.6 | 13.6 |
| Vel. máxima (m/s) | 205.19 | 24.36 | 11.98 | 11.98 |
| % vel. imposible | 1.1% | 0.3% | 0.0% | 0.0% |
| Var. aceleración | 159.41 | 11.24 | 6.04 | 4.99 |
| % ruido (jitter) | 3.7% | 1.8% | 0.0% | 0.0% |
| Dist/jugador (m) | 29.3 | 85.1 | 62.1 | 57.8 |
| Extrap. 90 min (km) | 5.3 | 15.3 | 11.2 | **10.4** |
| FPS procesamiento | 4.0 | 2.23 | 2.36 | 2.33 |

**Mejoras logradas (Baseline → Exp 3):**
- Fragmentación: ↓ 36.2% (470 → 300 IDs)
- Velocidades imposibles: ↓ 100% (eliminadas)
- Varianza de aceleración: ↓ 96.9%
- Ruido (jitter): ↓ 100% (eliminado)
- Extrapolación: 5.3 → 10.4 km/90min (dentro del rango FIFA 9-13 km)

**Artefactos generados en `data/processed/`:**
- `exp1_distances.csv`, `exp1_frame_data.csv`, `exp1_metrics.json`
- `exp2_distances.csv`, `exp2_frame_data.csv`, `exp2_metrics.json`
- `advanced_distances.csv`, `advanced_frame_data.csv`, `advanced_metrics.json`
- `ablation_results.json` — Consolidado de todos los experimentos
- `advanced_tactical.mp4` — Video anotado del pipeline avanzado (Exp 3)
- `models/fine_tuned/botsort_tuned.yaml` — Configuración BoT-SORT tuneada

**Reporte detallado:** `reports/reporte_fase2_ajustes.docx`

### Fase 2 — Etapa 2C: Evaluación y Reporte (pendiente)
- Notebook de comparación con gráficas lado a lado.
- Visualización avanzada (annotator.py).

## Comandos útiles

```bash
# Activar entorno
conda activate futbol

# Instalar dependencias
pip install -r requirements.txt

# Ejecutar línea base
python src/models/baseline_tracker.py

# Ejecutar estudio de ablación (Fase 2)
python src/models/advanced_tracker.py

# Abrir notebooks
jupyter notebook notebooks/
```

## Notas técnicas
- El video de entrada debe estar en `data/raw/video.mp4` (configurable en baseline_tracker.py línea 15).
- Video de salida baseline: `data/processed/baseline_tactical.mp4`.
- Video de salida avanzado: `data/processed/advanced_tactical.mp4`.
- CSV resumen baseline: `data/processed/baseline_distances.csv`.
- CSV por frame baseline: `data/processed/baseline_frame_data.csv` (frame, track_id, foot_u, foot_v, dist_meters).
- CSV por frame avanzado: `data/processed/advanced_frame_data.csv` (incluye frame_diff).
- Timing: `data/processed/baseline_timing.json`.
- Métricas baseline: `data/processed/baseline_metrics.json` (generado por notebook 03).
- Métricas ablación: `data/processed/ablation_results.json` (generado por advanced_tracker.py).
- Scale factor baseline: 0.065 m/px (estimación manual).
- Scale factor calibrado (Fase 2): 0.0451 m/px (referencias FIFA sobre 1548x898).
- Semilla de reproducibilidad: SEED=42 (random, numpy, torch, cudnn).
- La Línea Base usa suma cruda compensada por cámara, SIN filtros anti-jitter, para tener referencia de comparación con Fase 2.
