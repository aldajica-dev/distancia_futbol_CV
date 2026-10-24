"""Clasificación de equipos por color dominante del torso."""

import cv2
import numpy as np


def extract_torso_color(frame, box):
    """Extrae el color HSV medio del torso (40% superior, 60% central del bbox).

    Args:
        frame: Frame BGR completo.
        box: Bounding box (x1, y1, x2, y2).

    Returns:
        Tupla (H, S, V) promedio del torso, o None si el recorte es inválido.
    """
    x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
    bw = x2 - x1
    bh = y2 - y1
    if bw < 4 or bh < 8:
        return None

    # Torso: 40% superior en vertical, 60% central en horizontal
    margin_x = int(bw * 0.2)
    torso_y2 = y1 + int(bh * 0.4)
    crop = frame[y1:torso_y2, x1 + margin_x:x2 - margin_x]
    if crop.size == 0:
        return None

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    # Excluir píxeles verdes (césped): H entre 35-85 y S > 40
    mask = ~((hsv[:, :, 0] >= 35) & (hsv[:, :, 0] <= 85) & (hsv[:, :, 1] > 40))
    valid_pixels = hsv[mask]
    if len(valid_pixels) < 5:
        return None

    return tuple(valid_pixels.mean(axis=0).astype(float))


def classify_teams(track_colors, n_teams=3):
    """Clasifica tracks en equipos usando K-Means sobre colores HSV.

    Args:
        track_colors: Dict {tid: lista de tuplas (H, S, V)}.
        n_teams: Número de clusters (default 3: local, visitante, árbitro).

    Returns:
        Dict {tid: team_label (int 0..n_teams-1)}.
    """
    # Calcular color mediano por track
    tid_list = []
    color_list = []
    for tid, colors in track_colors.items():
        if len(colors) >= 2:
            median_color = np.median(colors, axis=0)
            tid_list.append(tid)
            color_list.append(median_color)

    if len(color_list) < n_teams:
        return {tid: 0 for tid in track_colors}

    data = np.float32(color_list)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 1.0)
    _, labels, _ = cv2.kmeans(data, n_teams, None, criteria, 10, cv2.KMEANS_PP_CENTERS)

    team_map = {}
    for tid, label in zip(tid_list, labels.ravel()):
        team_map[tid] = int(label)

    # Tracks sin suficientes observaciones de color → equipo -1
    for tid in track_colors:
        if tid not in team_map:
            team_map[tid] = -1

    return team_map
