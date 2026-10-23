"""Filtros cinemáticos: suavizado de trayectoria y control de plausibilidad física."""

import numpy as np


def smooth_point(history_buffer, current_pos, window_size=5):
    """Suaviza la posición actual usando media móvil sobre un buffer de posiciones.

    Args:
        history_buffer: Lista de posiciones previas [(u, v), ...].
        current_pos: Posición actual (u, v).
        window_size: Tamaño de ventana. Si <= 1, retorna sin suavizar.

    Returns:
        (updated_buffer, smoothed_pos): Buffer actualizado y posición suavizada.
    """
    updated = history_buffer + [current_pos]
    if window_size <= 1:
        return updated[-1:], current_pos

    updated = updated[-window_size:]
    arr = np.array(updated)
    smoothed = (float(arr[:, 0].mean()), float(arr[:, 1].mean()))
    return updated, smoothed


def filter_step_distance(raw_dist_m, frame_diff, fps, apply_filter=True,
                         deadband_ms=0.4, max_speed_ms=12.0, max_gap_frames=5):
    """Filtra un paso de distancia por plausibilidad física.

    Args:
        raw_dist_m: Distancia cruda en metros para este paso.
        frame_diff: Frames transcurridos desde la última detección de este track.
        fps: FPS del video.
        apply_filter: Si False, retorna valores crudos (modo Baseline/Exp1).
        deadband_ms: Velocidad mínima en m/s para considerar movimiento real.
        max_speed_ms: Velocidad máxima físicamente posible en m/s.
        max_gap_frames: Máximo gap de frames antes de descartar el paso.

    Returns:
        (filtered_dist_m, speed_ms): Distancia filtrada y velocidad instantánea.
    """
    if frame_diff <= 0:
        return 0.0, 0.0

    speed_ms = (raw_dist_m / frame_diff) * fps

    if not apply_filter:
        return raw_dist_m, speed_ms

    if frame_diff > max_gap_frames:
        return 0.0, 0.0

    if speed_ms < deadband_ms or speed_ms > max_speed_ms:
        return 0.0, 0.0

    return raw_dist_m, speed_ms
