# CLAUDE.md - Contexto del Proyecto

## Proyecto
**Análisis Cinemático de Fútbol** — Estimar distancias recorridas por jugadores a partir de video de cámara táctica dinámica usando YOLO + BoT-SORT + GMC.

## Contexto académico
Trabajo final de la materia "Técnicas Avanzadas de Modelado IA" en la Maestría de la Universidad de la Sabana.

## Arquitectura del proyecto

```
├── data/raw/          → Videos originales (video.mp4)
├── data/interim/      → Datos intermedios
├── data/processed/    → baseline_tactical.mp4 + baseline_distances.csv
├── models/homographies/ → Matrices H (.npy)
├── notebooks/         → Notebooks de exploración y homografía
├── src/features/      → projection.py (GMC + conversión a metros)
├── src/models/        → baseline_tracker.py (pipeline principal)
├── src/visualization/ → annotator.py (Fase 2)
├── reports/           → Documentación y figuras
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

### Fase 2 — Pendiente
- Mejorar persistencia de IDs (reducir fragmentación del tracker).
- Implementar filtro anti-jitter y deadband (suavizado de trayectorias).
- Calibrar scale_factor con medidas reales de la cancha.
- Implementar kinematics.py (velocidad, aceleración).
- EDA de calidad de video (notebook 01).
- Métricas de evaluación: comparar Fase 1 (cruda) vs Fase 2 (filtrada) (notebook 03).
- Visualización avanzada (annotator.py).

## Comandos útiles

```bash
# Activar entorno
conda activate futbol

# Instalar dependencias
pip install -r requirements.txt

# Ejecutar línea base
python src/models/baseline_tracker.py

# Abrir notebooks
jupyter notebook notebooks/
```

## Notas técnicas
- El video de entrada debe estar en `data/raw/video.mp4` (configurable en baseline_tracker.py línea 15).
- Video de salida: `data/processed/baseline_tactical.mp4`.
- CSV de salida: `data/processed/baseline_distances.csv`.
- El scale_factor por defecto es 0.065 (píxeles → metros). Ajustar según resolución y ángulo de la cámara.
- La Línea Base usa suma cruda compensada por cámara, SIN filtros anti-jitter, para tener referencia de comparación con Fase 2.
