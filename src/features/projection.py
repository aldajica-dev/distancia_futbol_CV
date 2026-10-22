"""Funciones de proyección homográfica y cálculo de distancia."""

import numpy as np


def project_point(H, u, v):
    """Proyecta un punto (u, v) de la imagen al plano real usando la homografía H.

    Args:
        H: Matriz de homografía 3x3.
        u: Coordenada horizontal en píxeles.
        v: Coordenada vertical en píxeles.

    Returns:
        Tupla (X, Y) en coordenadas del plano real (metros).
    """
    pt = np.array([u, v, 1.0])
    projected = H @ pt
    X = projected[0] / projected[2]
    Y = projected[1] / projected[2]
    return X, Y


def calculate_distance(p1, p2):
    """Calcula la distancia euclidiana entre dos puntos 2D.

    Args:
        p1: Tupla (X, Y) del primer punto.
        p2: Tupla (X, Y) del segundo punto.

    Returns:
        Distancia euclidiana en las mismas unidades de los puntos.
    """
    return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)
