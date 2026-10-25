# Análisis Cinemático de Fútbol

Estimación de distancias recorridas por jugadores a partir de video de cámara táctica dinámica, usando YOLO + BoT-SORT/OC-SORT + compensación global de movimiento de cámara (GMC).

**Trabajo final** — Materia "Técnicas Avanzadas de Modelado IA", Maestría Universidad de La Sabana.

---

## Estructura del proyecto

```
├── data/
│   ├── raw/             → video.mp4 (video de entrada)
│   ├── interim/         → pseudo-etiquetas Exp 7
│   └── processed/       → CSVs, JSONs de métricas, videos anotados
├── models/
│   ├── homographies/    → H_matrix.npy (generado por notebook 02)
│   └── fine_tuned/      → YAMLs de trackers + yolo11s_finetuned.pt (Exp 7)
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_homography_builder.ipynb
│   ├── 03_baseline_metrics.ipynb
│   ├── 04_colab_runner.ipynb
│   └── 04_exp6_exp7_colab.ipynb   ← experimentos 6 y 7 en Google Colab
├── src/
│   ├── features/
│   │   ├── projection.py    → GMC + calibración de escala
│   │   └── kinematics.py    → filtros anti-jitter
│   └── models/
│       ├── baseline_tracker.py   → Línea Base
│       ├── detector.py           → get_device(), create_*_yaml()
│       ├── tracker.py            → run_controlled_experiment()
│       ├── advanced_tracker.py   → Experimentos 1–5
│       └── gpu_tracker.py        → Experimentos 5, 6 y 7 (GPU local)
└── reports/
    ├── informe_final.docx           → Informe técnico completo (Baseline + Exp 1-7)
    └── reporte_fase2_ajustes.docx   → Reporte Fase 2 (Exp 1-7)
```

---

## Instalación

```bash
conda create -n futbol python=3.10
conda activate futbol
pip install -r requirements.txt
```

**GPU (recomendado para Exp 5-7):** instala PyTorch con CUDA antes de correr `gpu_tracker.py`:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

---

## Uso
el video a ser analizado debe estar en data/raw/ y llamarse video.mp4
### Línea Base
Para correr la linea base:
```bash
# Requiere: data/raw/video.mp4
python src/models/baseline_tracker.py
```

Salidas en `data/processed/`:
- `baseline_tactical.mp4` — video anotado
- `baseline_distances.csv` — distancias por track ID
- `baseline_frame_data.csv` — datos por frame
- `baseline_timing.json` — métricas de rendimiento

### FASE 2- Experimentos 1–4 
Para correr la fase 2, la cual consiste en ajuste de hiperparametros con respecto a la linea base:
```bash
python src/models/advanced_tracker.py           # todos
python src/models/advanced_tracker.py --only 3  # solo Exp 3
```

### Experimentos 5, 6 y 7 (GPU recomendado)

```bash
python src/models/gpu_tracker.py           # corre los tres en secuencia
python src/models/gpu_tracker.py --exp 5   # OC-SORT + YOLO11s
python src/models/gpu_tracker.py --exp 6   # BoT-SORT + YOLO11s + ReID (OSNet)
python src/models/gpu_tracker.py --exp 7   # YOLO fine-tuned + BoT-SORT + ReID
```

> **Exp 7** genera pseudo-etiquetas y fine-tunea YOLO automáticamente si no existe `models/fine_tuned/yolo11s_finetuned.pt`.

Salidas en `data/processed/`:
- `exp{5,6,7}_tactical.mp4`, `exp{5,6,7}_distances.csv`, `exp{5,6,7}_metrics.json`
- `ablation_results.json` — consolidado de todos los experimentos

---

## Estudio de ablación — resultados

| Experimento | IDs únicos | Frag. | Switches | Track mediana | Dist./jugadora |
|---|---|---|---|---|---|
| Baseline (YOLOv8n + BoT-SORT) | 460 | 20.9× | — | 3 fr | ~0.8 m |
| Exp 1: BoT-SORT tuneado | 47 | 2.1× | 31 | 52 fr | 42.7 m |
| Exp 2: + filtro anti-jitter | 47 | 2.1× | 31 | 52 fr | 40.1 m |
| Exp 3: + scale calibrado | 47 | 2.1× | 31 | 52 fr | 42.9 m |
| Exp 4: + color clustering + merge | 32 | 1.5× | 18 | 312 fr | 92.4 m |
| Exp 5: OC-SORT + YOLO11s | 31 | 1.4× | 12 | 520 fr | 138.1 m |
| **Exp 6: + ReID OSNet** | **37** | **1.7×** | **15** | **1030 fr** | **165.2 m** |
| **Exp 7: + YOLO fine-tuned** | **29** | **1.3×** | **7** | **691 fr** | **101.2 m** |

**Exp 6** ofrece la mayor cobertura continua (1030 fr/jugadora).  
**Exp 7** ofrece la mayor pureza de identidad (mínima fragmentación y switches).

---

## Notebooks de Colab

Los experimentos 6 y 7 fueron ejecutados en Google Colab por sus requerimientos de VRAM:

- `notebooks/04_exp6_exp7_colab.ipynb` — versión completa con resultados
- `notebooks/04_colab_runner.ipynb` — versión simplificada de referencia

Estos notebooks son evidencia del proceso; no es necesario re-ejecutarlos si ya existe `ablation_results.json`.
