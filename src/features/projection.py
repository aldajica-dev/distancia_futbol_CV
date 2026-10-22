"""Geometría: compensación de movimiento de cámara y conversión a metros."""

import cv2
import numpy as np

AFFINE_IDENTITY = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)


def estimate_camera_motion(prev_gray, curr_gray, boxes=None):
    """Estima la transformación afín de la cámara entre dos frames consecutivos.

    Usa flujo óptico sparse sobre puntos estáticos del fondo (excluye jugadores).

    Args:
        prev_gray: Frame anterior en escala de grises.
        curr_gray: Frame actual en escala de grises.
        boxes: Lista de bounding boxes [[x1,y1,x2,y2], ...] para enmascarar jugadores.

    Returns:
        Matriz afín 2x3 del movimiento de cámara.
    """
    mask = np.ones_like(prev_gray) * 255
    if boxes is not None:
        for (x1, y1, x2, y2) in boxes:
            mask[int(y1):int(y2), int(x1):int(x2)] = 0

    pts = cv2.goodFeaturesToTrack(prev_gray, mask=mask, maxCorners=200,
                                  qualityLevel=0.01, minDistance=30)
    if pts is None or len(pts) < 4:
        return AFFINE_IDENTITY.copy()

    matched, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, pts, None)
    valid = status.ravel() == 1
    src = pts[valid]
    dst = matched[valid]

    if len(src) < 4:
        return AFFINE_IDENTITY.copy()

    affine, inliers = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC)
    if affine is None:
        return AFFINE_IDENTITY.copy()

    return affine.astype(np.float32)


def compensate_point(point, affine_matrix):
    """Proyecta un punto del frame anterior al frame actual usando la matriz afín.

    Cancela el movimiento de cámara: retorna dónde estaría el punto si el jugador
    hubiera permanecido quieto.

    Args:
        point: Tupla (u, v) del punto en el frame anterior.
        affine_matrix: Matriz afín 2x3 del movimiento de cámara.

    Returns:
        Tupla (u', v') compensada en el frame actual.
    """
    pt = np.array([point[0], point[1], 1.0], dtype=np.float64)
    result = affine_matrix @ pt
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
