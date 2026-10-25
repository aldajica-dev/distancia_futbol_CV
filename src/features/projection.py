"""Geometría: compensación de movimiento de cámara y conversión a metros."""

import cv2
import numpy as np

AFFINE_IDENTITY = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)
HOMOGRAPHY_IDENTITY = np.eye(3, dtype=np.float32)


def estimate_camera_motion(prev_gray, curr_gray, boxes=None, use_homography=False):
    """Estima la transformación de cámara entre dos frames consecutivos.

    Usa flujo óptico sparse sobre puntos estáticos del fondo (excluye jugadores).

    Args:
        prev_gray: Frame anterior en escala de grises.
        curr_gray: Frame actual en escala de grises.
        boxes: Lista de bounding boxes [[x1,y1,x2,y2], ...] para enmascarar jugadores.
        use_homography: Si True, estima homografía completa 3x3 (8 DOF) en lugar de
            la transformación afín 2x3 (4 DOF). La homografía modela mejor cámaras
            con pan+tilt+zoom (perspectiva variable), reduciendo el error residual
            que genera distancia fantasma en jugadores quietos.

    Returns:
        Matriz 2x3 (afín) o 3x3 (homografía) del movimiento de cámara.
    """
    mask = np.ones_like(prev_gray) * 255
    if boxes is not None:
        for (x1, y1, x2, y2) in boxes:
            mask[int(y1):int(y2), int(x1):int(x2)] = 0

    pts = cv2.goodFeaturesToTrack(prev_gray, mask=mask, maxCorners=200,
                                  qualityLevel=0.01, minDistance=30)
    if pts is None or len(pts) < 4:
        return HOMOGRAPHY_IDENTITY.copy() if use_homography else AFFINE_IDENTITY.copy()

    matched, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, pts, None)
    valid = status.ravel() == 1
    src = pts[valid]
    dst = matched[valid]

    min_pts = 4
    if len(src) < min_pts:
        return HOMOGRAPHY_IDENTITY.copy() if use_homography else AFFINE_IDENTITY.copy()

    if use_homography:
        H, _ = cv2.findHomography(src, dst, method=cv2.RANSAC, ransacReprojThreshold=3.0)
        if H is None:
            return HOMOGRAPHY_IDENTITY.copy()
        return H.astype(np.float32)
    else:
        affine, _ = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC)
        if affine is None:
            return AFFINE_IDENTITY.copy()
        return affine.astype(np.float32)


def compensate_point(point, matrix, use_homography=False):
    """Proyecta un punto del frame anterior al frame actual cancelando el movimiento de cámara.

    Args:
        point: Tupla (u, v) del punto en el frame anterior.
        matrix: Matriz 2x3 (afín) o 3x3 (homografía) del movimiento de cámara.
        use_homography: Debe coincidir con el flag usado en estimate_camera_motion.

    Returns:
        Tupla (u', v') compensada en el frame actual.
    """
    if use_homography:
        pt = np.array([[[point[0], point[1]]]], dtype=np.float32)
        result = cv2.perspectiveTransform(pt, matrix)
        return float(result[0][0][0]), float(result[0][0][1])
    else:
        pt = np.array([point[0], point[1], 1.0], dtype=np.float64)
        result = matrix @ pt
        return float(result[0]), float(result[1])


def pixels_to_meters(p1, p2, scale_factor=0.065):
    """Calcula el desplazamiento en metros entre un punto actual y su previo compensado.

    Args:
        p1: Punto actual (u, v).
        p2: Punto previo compensado (u', v').
        scale_factor: Factor de conversión píxeles → metros.

    Returns:
        Distancia en metros.
    """
    dx = p1[0] - p2[0]
    dy = p1[1] - p2[1]
    return np.sqrt(dx * dx + dy * dy) * scale_factor


def calibrate_scale_factor(ref_points_px, real_distances_m):
    """Calcula scale_factor calibrado a partir de referencias conocidas de la cancha.

    Args:
        ref_points_px: Lista de pares de puntos [((u1,v1),(u2,v2)), ...] medidos en píxeles.
        real_distances_m: Lista de distancias reales en metros para cada par.

    Returns:
        scale_factor calibrado (metros/píxel).
    """
    scales = []
    for (p1, p2), real_m in zip(ref_points_px, real_distances_m):
        dx = p1[0] - p2[0]
        dy = p1[1] - p2[1]
        dist_px = np.sqrt(dx * dx + dy * dy)
        if dist_px > 0:
            scales.append(real_m / dist_px)
    return float(np.mean(scales))


def get_perspective_scale(foot_v, img_height, base_scale):
    """Corrección de escala por perspectiva vertical.

    Objetos más arriba en la imagen (lejos de la cámara) tienen mayor escala
    porque cada píxel representa más metros.

    Args:
        foot_v: Coordenada vertical del pie del jugador.
        img_height: Altura de la imagen en píxeles.
        base_scale: Scale factor base calibrado.

    Returns:
        Scale factor ajustado por perspectiva.
    """
    # ratio 0→1 de arriba a abajo; arriba = lejos = mayor escala
    t = 1.0 - (foot_v / img_height)
    # factor de corrección lineal: hasta 1.5x en la parte superior
    correction = 1.0 + 0.5 * t
    return base_scale * correction
