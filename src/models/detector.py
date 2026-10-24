"""Configuración de dispositivo y generación de archivos YAML para trackers."""

import os

import yaml


def get_device():
    """Detecta el mejor dispositivo disponible para PyTorch.

    Returns:
        'cuda', 'mps' o 'cpu'.
    """
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


def create_botsort_yaml(output_path, track_high_thresh=0.3, track_low_thresh=0.05,
                        new_track_thresh=0.6, track_buffer=90, match_thresh=0.85,
                        gmc_method="sparseOptFlow", with_reid=False, appearance_thresh=0.25):
    """Genera un archivo YAML de configuración de BoT-SORT para Ultralytics.

    Args:
        output_path: Ruta donde guardar el YAML.
        track_high_thresh: Umbral alto de detección para primera asociación.
        track_low_thresh: Umbral bajo para segunda asociación.
        new_track_thresh: Umbral mínimo para inicializar un track nuevo.
        track_buffer: Frames que un track perdido se mantiene vivo.
        match_thresh: Umbral de matching IoU/apariencia.
        gmc_method: Método de compensación de movimiento global.
        with_reid: Activar ReID por apariencia para re-identificar jugadoras perdidas.
        appearance_thresh: Umbral de distancia de apariencia para re-identificación.

    Returns:
        Ruta absoluta al archivo YAML generado.
    """
    config = {
        "tracker_type": "botsort",
        "track_high_thresh": track_high_thresh,
        "track_low_thresh": track_low_thresh,
        "new_track_thresh": new_track_thresh,
        "track_buffer": track_buffer,
        "match_thresh": match_thresh,
        "fuse_score": True,
        "gmc_method": gmc_method,
        "proximity_thresh": 0.5,
        "appearance_thresh": appearance_thresh,
        "with_reid": with_reid,
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    return os.path.abspath(output_path)


def create_bytetrack_yaml(output_path, track_high_thresh=0.25, track_low_thresh=0.10,
                          new_track_thresh=0.60, track_buffer=450, match_thresh=0.80):
    """Genera YAML de ByteTrack para Ultralytics.

    ByteTrack hace dos pasadas de asociación:
      1ª pasada: detecciones con conf > track_high_thresh → tracks activos
      2ª pasada: detecciones con track_low_thresh < conf < track_high_thresh → recupera tracks perdidos

    Para que la 2ª pasada funcione, conf_thresh en model.track() debe ser igual a track_low_thresh.

    Returns:
        Ruta absoluta al archivo YAML generado.
    """
    config = {
        "tracker_type": "bytetrack",
        "track_high_thresh": track_high_thresh,
        "track_low_thresh": track_low_thresh,
        "new_track_thresh": new_track_thresh,
        "track_buffer": track_buffer,
        "match_thresh": match_thresh,
        "fuse_score": True,
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    return os.path.abspath(output_path)
