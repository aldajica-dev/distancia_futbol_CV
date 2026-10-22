# Análisis Cinemático de Fútbol - Línea Base

Estimación de distancias recorridas por jugadores de fútbol a partir de video de cámara táctica dinámica (paneos laterales y zooms suaves), usando YOLOv8 + BoT-SORT + compensación global de movimiento de cámara (GMC).

## Alcance

- Detección de jugadores con YOLOv8n (clase persona).
- Tracking con BoT-SORT (IDs persistentes cuadro a cuadro).
- Compensación de movimiento de cámara mediante flujo óptico sparse con máscara de exclusión de jugadores.
- Conversión de desplazamiento en píxeles a metros (factor de escala configurable).
- Exportación de distancias acumuladas por jugador en CSV.

## Instalación

```bash
pip install -r requirements.txt
```

## Uso rápido

```bash
# Colocar video en data/raw/video.mp4
python src/models/baseline_tracker.py
```

Salidas:
- `data/processed/baseline_tactical.mp4` — video anotado con IDs y distancias.
- `data/processed/baseline_distances.csv` — tabla Track_ID, Distancia_Metros.
