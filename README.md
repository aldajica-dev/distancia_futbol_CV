# Análisis Cinemático de Fútbol - Línea Base

Estimación de distancias recorridas por jugadores de fútbol a partir de video táctico, usando detección con YOLO, tracking por ID y proyección homográfica del plano de la cancha.

## Alcance

- Detección de jugadores con YOLOv8n (clase persona).
- Tracking con IDs persistentes cuadro a cuadro.
- Proyección de coordenadas imagen → coordenadas reales de cancha mediante homografía.
- Cálculo de distancia euclidiana acumulada por jugador.

## Instalación

```bash
pip install -r requirements.txt
```

## Uso rápido

```bash
python src/models/baseline_tracker.py
```
