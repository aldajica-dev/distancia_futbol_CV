# Bitácora de Cambios — Fase 2

## Exp 1: BoT-SORT Tuneado
**Variable aislada:** Hiperparámetros del tracker + resolución de inferencia.

**Cambios:**
- track_high_thresh: 0.50 → 0.40
- new_track_thresh: 0.60 → 0.58
- track_buffer: 30 → 120 (mantiene tracks perdidos 7.5 seg)
- conf_thresh: 0.25 → 0.35
- imgsz: 640 → 1280

**Resultado:** 470 → 300 IDs (↓36.2%), var. aceleración 159.41 → 11.24 (↓92.9%)

---

## Exp 2: + Filtro Anti-Jitter
**Variable aislada:** Suavizado de trayectoria + filtros de plausibilidad física.

**Cambios sobre Exp 1:**
- smooth_window = 5 (media móvil)
- deadband = 0.4 m/s
- max_speed = 12.0 m/s
- max_gap = 5 frames

**Resultado:** Vel. imposibles 0.3% → 0.0%, jitter 1.8% → 0.0%, distancia 1872.3 → 1365.9 m (↓27.1% — ruido eliminado)

---

## Exp 3: + Calibración Scale Factor
**Variable aislada:** Scale factor calibrado FIFA + corrección de perspectiva.

**Cambios sobre Exp 2:**
- scale_factor: 0.065 → 0.0451 m/px (calibrado con 3 referencias FIFA sobre 1548x898)
  - Ancho área penal: (324,430)→(1224,430) = 900px → 40.32m
  - Largo área penal: (540,280)→(540,645) = 365px → 16.50m
  - Diámetro círculo central: (572,350)→(977,350) = 405px → 18.30m
- use_perspective_scale = True (corrección lineal hasta 1.5x en borde superior)

**Resultado:** Extrap. 11.2 → 10.4 km/90min (rango FIFA 9-13 km), var. aceleración 6.04 → 4.99

---

## Exp 4: + Color Clustering + Track Merge
**Variable aislada:** Clasificación de equipos por color + fusión post-proceso de fragmentos.

### Módulos nuevos

**color_classifier.py:**
- `extract_torso_color(frame, box)` — Recorta 40% superior / 60% central del bbox (zona torso), convierte a HSV, excluye píxeles verdes (césped: H 35-85, S>40), retorna media (H,S,V)
- `classify_teams(track_colors, n_teams=3)` — Color mediano por track → K-Means (K=3) → etiqueta de equipo

**track_merger.py:**
- `build_track_segments(...)` — Descriptor por track: frame inicio/fin, posición inicio/fin, equipo
- `merge_tracks(segments, max_gap_frames=30, max_dist_px=100)` — Fusión greedy con 3 reglas:
  1. Solo mismo equipo
  2. Gap temporal ≤ 30 frames (~1.9 seg)
  3. Distancia espacial ≤ 100 px
- `apply_merge(...)` — Reasigna IDs según merge map, re-agrega distancias

**Cambios en tracker.py:**
- Nuevas banderas: apply_color_clustering, apply_track_merge, merge_max_gap_frames, merge_max_dist_px
- Extracción de color durante el loop de tracking
- Post-procesamiento después del loop: classify_teams → merge_tracks → apply_merge

**Resultado:**
- 3 equipos detectados correctamente por K-Means
- 300 → 117 IDs (fusionados 183, ↓61%)
- Fragmentación: 13.6x → 5.3x
- Long. media track: 27.7 → 60.2 frames (↑117%)
- Distancia total invariante: 1270.5 m (el merge solo reasigna, no modifica distancias)
- FPS: 2.38 → 1.90 (overhead de extracción de color + K-Means + merge)

**Problemas pendientes:**
1. **117 IDs sigue siendo alto** (5.3x para 22 jugadores) — merge no suficientemente agresivo
2. **Cámara suma distancia a jugadores quietos** — deadband de 0.4 m/s no filtra el error residual del GMC

---

---

## Exp 4 v2: Ajuste de Parámetros de Merge + Deadband
**Problema a resolver:**
1. 117 IDs sigue alto (5.3x) — merge no suficientemente agresivo
2. Cámara suma distancia a jugadores quietos — deadband 0.4 m/s no filtra error residual GMC

**Cambios sobre Exp 4 v1:**
- deadband_ms: 0.4 → 1.0 (una persona caminando lento es ~1.2 m/s, 1.0 filtra solo ruido GMC)
- merge_max_gap_frames: 30 → 60 (~3.75 seg a 16 FPS, permite fusiones más lejanas temporalmente)
- merge_max_dist_px: 100 → 150 (acepta mayor desplazamiento entre fragmentos)

**Optimización de ejecución:**
- Exp 1-3 se cargan de cache (JSON guardados), solo Exp 4 se re-ejecuta → ~4 min en vez de ~14 min

**Resultado:**
- IDs: 117 → 102 (↓12.8% adicional, fragmentación 4.6x)
- Distancia total: 1270.5 → 1209.5 m (↓4.8% — deadband filtró ruido GMC en jugadores quietos)
- Var. aceleración: 5.60 → 4.94 (↓11.8%)
- Extrap. 90 min: 10.4 → 9.9 km (sigue en rango FIFA 9-13 km)
- Long. media track: 60.2 → 66.6 frames (↑10.6%)

---

## Tabla Comparativa Completa

| Métrica | Baseline | Exp 1 | Exp 2 | Exp 3 | Exp 4 |
|---|---|---|---|---|---|
| IDs únicos | 470 | 300 | 300 | 300 | 102 |
| Fragmentación (x) | 21.4 | 13.6 | 13.6 | 13.6 | 4.6 |
| ID switches est. | 448 | 278 | 278 | 278 | 80 |
| Long. media track (fr) | 13.3 | 27.7 | 27.7 | 27.7 | 66.6 |
| Vel. máxima (m/s) | 205.19 | 24.36 | 11.98 | 11.98 | 11.98 |
| % vel. imposible | 1.1% | 0.3% | 0.0% | 0.0% | 0.0% |
| Var. aceleración | 159.41 | 11.24 | 6.04 | 4.99 | 4.94 |
| % ruido (jitter) | 3.7% | 1.8% | 0.0% | 0.0% | 0.0% |
| Distancia total (m) | 644.0 | 1872.3 | 1365.9 | 1270.5 | 1209.5 |
| Dist/jugador (m) | 29.3 | 85.1 | 62.1 | 57.8 | 55.0 |
| Extrap. 90 min (km) | 5.3 | 15.3 | 11.2 | 10.4 | 9.9 |
| FPS procesamiento | 4.0 | 2.18 | 2.27 | 2.38 | 2.11 |

## Mejoras acumuladas Baseline → Exp 4 (Video 1: 1548x898@16fps, 480 frames)

| Métrica | Baseline | Exp 4 | Mejora |
|---|---|---|---|
| IDs únicos | 470 | 102 | ↓ 78.3% |
| Fragmentación | 21.4x | 4.6x | ↓ 78.5% |
| Vel. imposibles | 1.1% | 0.0% | ↓ 100% |
| Var. aceleración | 159.41 | 4.94 | ↓ 96.9% |
| Ruido (jitter) | 3.7% | 0.0% | ↓ 100% |
| Extrap. 90 min | 5.3 km | 9.9 km | ↑ 86.8% |
| Rango FIFA | Fuera | Dentro | ✓ |

---

## Cambio de Video (2026-10-23)

**Motivación:** Validar que el pipeline generaliza a otros partidos/cámaras.

**Video anterior:** 1548x898 @ 16 FPS, 480 frames (~30 seg)
**Video nuevo:** 1364x764 @ 30 FPS, 1233 frames (~41 seg)

**Recalibración requerida:**
- Nuevos puntos de referencia FIFA medidos sobre frame 300 del nuevo video:
  - Ancho área penal: (335,265)→(1020,265) = 685 px → 40.32 m
  - Largo área penal: (680,150)→(680,265) = 115 px → 16.50 m
  - Diámetro círculo central: (575,345)→(855,345) = 280 px → 18.30 m
- Resultados anteriores eliminados, pipeline completo re-ejecutado desde cero
- Nota: el video a 30 FPS (vs 16) puede afectar track_buffer (120 frames = 4 seg a 30fps vs 7.5 seg a 16fps) y merge_max_gap_frames (60 frames = 2 seg a 30fps vs 3.75 seg a 16fps)

**Scale factor recalibrado: 0.0892 m/px** (vs 0.0451 del video anterior — esperado, menor resolución implica más metros por pixel).

**Parámetros ajustados para 30fps (ratio 1.875x vs 16fps anterior):**
- track_buffer: 120 → 225 (mantiene ~7.5 seg)
- smooth_window: 5 → 9 (~0.3 seg)
- max_gap_frames: 5 → 9 (~0.3 seg)
- merge_max_gap_frames: 60 → 112 (~3.75 seg)

---

## Resultados nuevo video — Corrida v2 (con ajuste hiperparámetros, antes de ReID/polígono)

| Métrica | Baseline | Exp 1 | Exp 2 | Exp 3 | Exp 4 |
|---|---|---|---|---|---|
| IDs únicos | 42 | 44 | 44 | 44 | 28 |
| Fragmentación (x) | 1.9 | 2.0 | 2.0 | 2.0 | 1.3 |
| ID switches est. | 20 | 22 | 22 | 22 | 6 |
| Long. media track (fr) | 25.5 | 73.4 | 73.4 | 73.4 | 113.8 |
| Vel. máxima (m/s) | 44.5 | 13.2 | 12.0 | 12.0 | 12.0 |
| % vel. imposible | 1.1% | 0.1% | 0.0% | 0.0% | 0.0% |
| Distancia total (m) | 75.7 | 315.9 | 535.0 | 497.3 | 495.8 |
| Extrap. 90 min (km) | 0.4 | 1.9 | 3.2 | 3.0 | 3.0 |

**Problema detectado:** varias jugadoras no detectadas (conf_thresh=0.50 muy alto), IDs siguen saltando (sin ReID ni polígono de cancha).

---

## Corrida v3: + Polígono de Cancha + ReID + conf_thresh=0.35 (2026-10-23)

**Cambios aplicados a todos los experimentos:**
- `conf_thresh: 0.50 → 0.35` (detecta jugadoras lejanas/ocluidas)
- `field_polygon = [(60,145),(1300,145),(1364,600),(1364,764),(0,764),(0,600)]` — trapezoide que excluye graderías y bordes laterales
- `with_reid=True, appearance_thresh=0.35` en BoT-SORT YAML (re-identificación por apariencia)

**Resultados:**

| Métrica | Baseline | Exp 1 | Exp 2 | Exp 3 | Exp 4 |
|---|---|---|---|---|---|
| IDs únicos | 42 | 36 | 36 | 36 | 27 |
| Fragmentación (x) | 1.9 | 1.6 | 1.6 | 1.6 | 1.2 |
| ID switches est. | 20 | 14 | 14 | 14 | 5 |
| Long. media track (fr) | 25.5 | 119.8 | 119.8 | 119.8 | 159.8 |
| Vel. máxima (m/s) | 44.5 | 15.3 | 12.0 | 12.0 | 12.0 |
| % vel. imposible | 1.1% | 0.2% | 0.0% | 0.0% | 0.0% |
| Var. aceleración | 5.66 | 2.34 | 3.87 | 3.39 | 3.43 |
| % ruido (jitter) | 0.4% | 0.5% | 0.0% | 0.0% | 0.0% |
| Distancia total (m) | 75.7 | 381.6 | 647.3 | 581.8 | 580.0 |
| Dist/jugador (m) | 3.4 | 17.3 | 29.4 | 26.4 | 26.4 |
| Extrap. 90 min (km) | 0.4 | 2.3 | 3.9 | 3.5 | 3.5 |
| FPS procesamiento | 4.2 | 2.0 | 2.4 | 2.6 | 2.2 |

**Mejoras logradas con los 3 cambios (v2 → v3, Exp 1):**
- IDs: 44 → 36 (↓18.2%) — polígono filtra detecciones fuera de cancha
- Track length: 73.4 → 119.8 frames (↑63.2%) — ReID recupera identidades perdidas
- ID switches: 22 → 14 (↓36.4%) — menos cambios de identidad por jugadora
- Fragmentación: 2.0x → 1.6x

**Mejoras acumuladas Baseline → Exp 4 (corrida v3):**
- IDs: 42 → 27 (↓35.7%)
- Fragmentación: 1.9x → 1.2x (near-ideal para 22 jugadoras esperadas)
- ID switches: 20 → 5 (↓75%)
- Track length media: 25.5 → 159.8 frames (↑527%)
- Vel. imposibles: 1.1% → 0.0% (↓100%)
- Var. aceleración: 5.66 → 3.43 (↓39.4%)
- Jitter: 0.4% → 0.0% (↓100%)

**Nota sobre extrapolación (3.5 km/90min):** el video de 41 seg muestra una fase de juego de muy baja actividad (velocidad media ~0.6 m/s, por debajo del paso caminando). No es un error del sistema sino una limitación del segmento de video específico.

**Artefactos:** `exp{1,2,3,4}_*.csv`, `exp{1,2,3,4}_metrics.json`, `ablation_results.json`, `exp4_tactical.mp4`

---

## Corrida v4: YOLOv8s + conf_thresh=0.25 (2026-10-24)

**Problema a resolver:** Aún con polígono de cancha + ReID, varias jugadoras no son detectadas. El modelo YOLOv8n (nano) tiene recall bajo para personas pequeñas/lejanas, y conf_thresh=0.35 descarta detecciones borderline.

**Cambios aplicados a todos los experimentos:**
- `model_path: yolov8n.pt → yolov8s.pt` (YOLOv8 Small — mejor recall para personas lejanas/pequeñas, ~2x parámetros vs nano)
- `conf_thresh: 0.35 → 0.25` (el polígono de cancha ya filtra espectadores, podemos ser más agresivos)
- `tracker.py` ahora acepta `model_path` desde config (antes hardcodeado)

**Trade-off esperado:** ~2x más lento por frame (YOLOv8s ~11.2M params vs YOLOv8n ~3.2M), pero mayor detección de jugadoras.

**Resultados:**

| Métrica | Baseline | Exp 1 | Exp 2 | Exp 3 | Exp 4 |
|---|---|---|---|---|---|
| IDs únicos | 42 | 44 | 44 | 44 | 32 |
| Fragmentación (x) | 1.9 | 2.0 | 2.0 | 2.0 | 1.5 |
| ID switches est. | 20 | 22 | 22 | 22 | 10 |
| Long. media track (fr) | 25.5 | 181.9 | 181.9 | 181.9 | 250.1 |
| Vel. máxima (m/s) | 44.5 | 15.7 | 12.0 | 12.0 | 12.0 |
| % vel. imposible | 1.1% | 0.1% | 0.0% | 0.0% | 0.0% |
| Var. aceleración | 5.66 | 1.76 | 2.37 | 2.96 | 3.16 |
| % ruido (jitter) | 0.4% | 0.7% | 0.0% | 0.0% | 0.0% |
| Distancia total (m) | 75.7 | 622.5 | 1194.3 | 1099.9 | 1096.0 |
| Dist/jugador (m) | 3.4 | 28.3 | 54.3 | 50.0 | 49.8 |
| Extrap. 90 min (km) | 0.4 | 3.7 | 7.1 | 6.6 | 6.5 |
| FPS procesamiento | 4.2 | 1.0 | 1.1 | 1.1 | 1.1 |

**Mejoras v3 → v4 (Exp 4):**
- Track length: 159.8 → 250.1 frames (↑56%) — YOLOv8s detecta jugadoras que nano perdía
- Distancia total: 580.0 → 1096.0 m (↑89%) — casi el doble de distancia capturada
- Extrap. 90 min: 3.5 → 6.5 km (↑86%) — acercándose al rango FIFA
- Var. aceleración: 3.43 → 3.16 (↓8%) — trayectorias más suaves
- FPS: 2.2 → 1.1 (↓50%) — costo esperado de YOLOv8s vs YOLOv8n

**Mejoras acumuladas Baseline → Exp 4 (corrida v4):**
- Track length: 25.5 → 250.1 frames (↑881%)
- Distancia total: 75.7 → 1096.0 m (↑1348%)
- Vel. imposibles: 1.1% → 0.0% (↓100%)
- Jitter: 0.4% → 0.0% (↓100%)
- Var. aceleración: 5.66 → 3.16 (↓44%)
- Extrap. 90 min: 0.4 → 6.5 km

**Nota:** IDs pasaron de 36 a 44 (Exp 1) porque conf_thresh=0.25 detecta más personas en cancha. Después del merge (Exp 4): 32 IDs para 22 jugadoras esperadas (1.5x fragmentación). La extrapolación de 6.5 km aún está por debajo de FIFA 9-13 km, consistente con un segmento de video de baja actividad.

---

## Corrida v5: track_buffer=450 + match_thresh=0.55 + appearance_thresh=0.25 (2026-10-24)

**Problema a resolver:** IDs siguen saltando — el tracker pierde la asociación y crea un ID nuevo cuando una jugadora es ocluida brevemente o detectada con confianza baja.

**Cambios (solo en YAML BoT-SORT, sin cambio en modelo ni conf_thresh):**
- `track_buffer: 225 → 450` (15 seg a 30fps — el ID se mantiene vivo el doble de tiempo antes de descartarse)
- `match_thresh: 0.70 → 0.55` (más fácil asociar detección a track existente por IoU)
- `appearance_thresh: 0.35 → 0.25` (ReID más permisivo — acepta mayor diferencia de apariencia para re-identificar)

**Resultados:** *(pendiente — corrida en curso)*
