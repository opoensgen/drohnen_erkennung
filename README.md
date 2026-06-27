# Drohnen-Erkennung mit YOLO11

Objekterkennung von Drohnen/UAVs in Bildern mit einem auf YOLO11n feinabgestimmten Modell.
Abschlussprojekt im Modul *Data Science with Python* (BHT).

## Erkannte Klassen (5)

| ID | Klasse        |
|----|---------------|
| 0  | shahed_136    |
| 1  | shahed_238    |
| 2  | mq9_reaper    |
| 3  | dji_mavic     |
| 4  | mohajer_6     |

## Projektstruktur

```
.
├── 7_DataCleaning_exercise.ipynb      # Metadaten bereinigen (metadata.csv -> metadata_cleaned.csv)
├── 8_DataVisualization_exercise.ipynb # Explorative Datenanalyse / Visualisierung
├── 9_YoloTraining.ipynb               # YOLO-Training, Validierung, Inferenz
├── dataset.yaml                       # YOLO-Dataset-Konfiguration (Klassen + Splits)
├── metadata.csv / metadata_cleaned.csv
├── yolo11n.pt                         # Pretrained YOLO11n (Ausgangsgewichte)
├── images/   {train, val, test}       # Bild-Splits
├── labels/   {train, val, test}       # YOLO-Labels (.txt, eine Box pro Zeile)
├── own_images/                        # Eigene Testbilder (drone / no_drone)
└── runs/detect/
    ├── train-2/   # GENUTZTES Training -> weights/best.pt
    └── train-5/   # Plots (Confusion-Matrix, PR/F1-Kurven)
```

## Setup

```bash
git clone git@github.com:opoensgen/drohnen_erkennung.git
cd drohnen_erkennung

# Python 3.10-3.12 empfohlen (conda-Env o. venv)
pip install -r requirements.txt
```

> Trainiert wurde im conda-Env `yolo_env` (ultralytics + OpenCV). Die `.venv` im
> Repo war nicht enthalten — Abhaengigkeiten kommen aus `requirements.txt`.

## Inferenz mit dem trainierten Modell

```python
from ultralytics import YOLO

model = YOLO("runs/detect/train-2/weights/best.pt")   # bestes Modell
model.predict("own_images/drone.jpg", save=True, conf=0.25)
```

## Training reproduzieren

Konfiguration des genutzten Laufs (`runs/detect/train-2/args.yaml`):

| Parameter | Wert        |
|-----------|-------------|
| model     | yolo11n.pt  |
| data      | dataset.yaml|
| epochs    | 30          |
| batch     | 4           |
| imgsz     | 640         |
| device    | cpu         |

```python
from ultralytics import YOLO

model = YOLO("yolo11n.pt")
model.train(data="dataset.yaml", epochs=30, batch=4, imgsz=640, device="cpu")
```

Ergebnisse/Metriken: `runs/detect/train-2/results.csv` & `results.png`,
Plots (Confusion-Matrix, PR/F1) in `runs/detect/train-5/`.
