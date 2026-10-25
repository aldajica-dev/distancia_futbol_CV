"""Post-procesamiento: fusión de fragmentos de track del mismo jugador."""

import numpy as np


def build_track_segments(frame_data, total_distances, team_labels):
    """Construye segmentos de track con posición inicial/final, velocidad y equipo.

    Args:
        frame_data: Lista de tuplas (frame, tid, foot_u, foot_v, dist, frame_diff).
        total_distances: Dict {tid: distancia_total}.
        team_labels: Dict {tid: team_label}.

    Returns:
        Lista de dicts con info de cada segmento de track.
    """
    track_info = {}
    # Para velocidad: guardamos las últimas N posiciones con sus frames
    track_tail = {}

    for row in frame_data:
        frame, tid, foot_u, foot_v = row[0], row[1], row[2], row[3]
        if tid not in track_info:
            track_info[tid] = {
                "tid": tid,
                "first_frame": frame,
                "last_frame": frame,
                "first_pos": (foot_u, foot_v),
                "last_pos": (foot_u, foot_v),
                "n_detections": 0,
            }
            track_tail[tid] = []
        track_info[tid]["last_frame"] = frame
        track_info[tid]["last_pos"] = (foot_u, foot_v)
        track_info[tid]["n_detections"] += 1
        track_tail[tid].append((frame, foot_u, foot_v))
        if len(track_tail[tid]) > 10:
            track_tail[tid].pop(0)

    segments = []
    for tid, info in track_info.items():
        info["distance"] = total_distances.get(tid, 0.0)
        info["team"] = team_labels.get(tid, -1)
        # Velocidad media (px/frame) de las últimas observaciones del track
        tail = track_tail[tid]
        if len(tail) >= 2:
            df = tail[-1][0] - tail[0][0]
            if df > 0:
                info["vel_u"] = (tail[-1][1] - tail[0][1]) / df
                info["vel_v"] = (tail[-1][2] - tail[0][2]) / df
            else:
                info["vel_u"] = 0.0
                info["vel_v"] = 0.0
        else:
            info["vel_u"] = 0.0
            info["vel_v"] = 0.0
        segments.append(info)

    segments.sort(key=lambda s: s["first_frame"])
    return segments


def merge_tracks(segments, max_gap_frames=30, max_dist_px=100.0):
    """Fusiona fragmentos de track que probablemente pertenecen al mismo jugador.

    Reglas:
    - Solo fusiona tracks del MISMO equipo (team label).
    - El gap temporal entre el final de A y el inicio de B debe ser <= max_gap_frames.
    - La distancia entre la posición PREDICHA de A (proyectada con su velocidad) y
      la primera posición de B debe ser <= max_dist_px.
    - Greedy: una vez que B se fusiona con A, B no puede fusionarse con otro.

    Args:
        segments: Lista de dicts de build_track_segments().
        max_gap_frames: Máximo gap temporal para considerar fusión.
        max_dist_px: Máxima distancia espacial para considerar fusión.

    Returns:
        Dict {old_tid: merged_tid} donde merged_tid es el tid del segmento raíz.
    """
    merge_map = {}
    for seg in segments:
        merge_map[seg["tid"]] = seg["tid"]

    absorbed = set()

    # Índice por tid para acceder rápido a velocidad/posición actualizadas del grupo
    seg_by_tid = {seg["tid"]: seg for seg in segments}

    for i, seg_a in enumerate(segments):
        if seg_a["tid"] in absorbed:
            continue
        if seg_a["team"] == -1:
            continue

        root_tid = merge_map[seg_a["tid"]]
        current_last_frame = seg_a["last_frame"]
        current_last_pos = seg_a["last_pos"]
        current_vel_u = seg_a.get("vel_u", 0.0)
        current_vel_v = seg_a.get("vel_v", 0.0)

        for j in range(i + 1, len(segments)):
            seg_b = segments[j]
            if seg_b["tid"] in absorbed:
                continue
            # Permitir merge: mismo equipo, o alguno sin clasificar (team=-1)
            if seg_b["team"] != seg_a["team"] and seg_a["team"] != -1 and seg_b["team"] != -1:
                continue

            gap = seg_b["first_frame"] - current_last_frame
            if gap < 0 or gap > max_gap_frames:
                continue

            # Distancia estática (fallback seguro)
            dx_s = seg_b["first_pos"][0] - current_last_pos[0]
            dy_s = seg_b["first_pos"][1] - current_last_pos[1]
            static_dist = np.sqrt(dx_s * dx_s + dy_s * dy_s)

            # Distancia con posición proyectada por velocidad
            predicted_u = current_last_pos[0] + current_vel_u * gap
            predicted_v = current_last_pos[1] + current_vel_v * gap
            dx_p = seg_b["first_pos"][0] - predicted_u
            dy_p = seg_b["first_pos"][1] - predicted_v
            predicted_dist = np.sqrt(dx_p * dx_p + dy_p * dy_p)

            # Tomar la mejor de las dos (nunca peor que estático puro)
            spatial_dist = min(static_dist, predicted_dist)
            if spatial_dist > max_dist_px:
                continue

            # Fusionar B → raíz
            merge_map[seg_b["tid"]] = root_tid
            absorbed.add(seg_b["tid"])
            current_last_frame = seg_b["last_frame"]
            current_last_pos = seg_b["last_pos"]
            current_vel_u = seg_b.get("vel_u", 0.0)
            current_vel_v = seg_b.get("vel_v", 0.0)

    return merge_map


def apply_merge(total_distances, frame_data, merge_map):
    """Aplica el merge map para re-agregar distancias y frame_data.

    Args:
        total_distances: Dict {tid: distancia_total} original.
        frame_data: Lista de tuplas original.
        merge_map: Dict {old_tid: merged_tid}.

    Returns:
        (merged_distances, merged_frame_data, n_original, n_merged):
            - merged_distances: Dict {merged_tid: distancia_total}.
            - merged_frame_data: Lista de tuplas con tids actualizados.
            - n_original: Número de IDs originales.
            - n_merged: Número de IDs después del merge.
    """
    merged_distances = {}
    for tid, dist in total_distances.items():
        mtid = merge_map.get(tid, tid)
        merged_distances[mtid] = merged_distances.get(mtid, 0.0) + dist

    merged_frame_data = []
    for row in frame_data:
        frame, tid, foot_u, foot_v, dist, frame_diff = row
        mtid = merge_map.get(tid, tid)
        merged_frame_data.append((frame, mtid, foot_u, foot_v, dist, frame_diff))

    return merged_distances, merged_frame_data, len(total_distances), len(merged_distances)
