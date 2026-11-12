# Детекция ноутбуков и телефонов: YOLO26m vs RF-DETR Medium

Учебный проект по курсу «Введение в ИИ»: дообучение и сравнение двух детекторов
на собственном датасете из 2 классов — `laptop` и `cell phone`.

## Что сделано

1. **Разметка.** 200 фото были предварительно размечены моделью **Grounding DINO**
   (`batch_grounding.py`), результат — `results/grounding/coco_preannotation.json`.
   Далее разметка проверена и доведена вручную (`instances_Train1.json`, `instances_Train2.json`).
2. **Подготовка датасета.** `prepare_dataset.py` конвертирует COCO-разметку в YOLO-формат
   и делит данные 70/15/15 (seed=42).
3. **QA разметки.** `verify_mapping.py` проверял, что `file_name ↔ bbox` согласованы
   (файл существует, размеры кадра совпадают с json, боксы не вылезают за кадр).
4. **Обучение.** Две модели, по два прогона каждая (без аугментаций и с аугментациями).
5. **Оценка.** Сравнение по mAP на валидации + качественное сравнение предсказаний
   на тестовой выборке (`results/yolo/`, `results/detr/`, `results/*_detections.csv`).

## Датасет

| Сплит | Изображений | Laptop | Cell phone | Всего боксов |
|---|---|---|---|---|
| train | 138 | 199 | 128 | 327 |
| valid | 29 | 28 | 29 | 57 |
| test | 31 | 40 | 22 | 62 |
| **итого** | **198** | **267** | **179** | **446** |

- Формат: YOLO (`label` — `class cx cy w h`, нормированные координаты).
- Исходные фото — 5712×4284 (смартфон). В репозитории изображения перекодированы
  до длинной стороны 1600 px, чтобы держать размер репозитория разумным
  (`_build/make_light_dataset.py`); разметка при этом не меняется, т.к. координаты нормированные.
- `dataset/data.yaml` — конфиг с относительными путями.

## Модели и конфигурации

| | YOLO26m | RF-DETR Medium |
|---|---|---|
| База | ultralytics `yolo26m.pt` | `rf-detr-medium.pth` (DINOv2 encoder) |
| Скрипт | `train_yolo.py` | `train_rfdetr.py` |
| Эпохи | 60 (patience=20) | 40 |
| Batch / LR | 8 / 5e-4 (AdamW) | 4 / 1e-4 (AdamW) |
| Разрешение | 640 | 576 |
| Аугментации (прогон `_aug`) | mosaic, mixup, HSV, affine (deg 20, scale 0.7), erasing 0.3, close_mosaic=20 | встроенные аугментации RF-DETR + `scale_jitter` |
| Заморозка | freeze=10 | — |

## Результаты (валидация, 29 изображений / 57 боксов)

| Модель | Прогон | Эпох | Лучшая эпоха | mAP@50 | mAP@50-95 | Precision | Recall |
|---|---|---|---|---|---|---|---|
| YOLO26m | без аугм. (`yolo26m_ft-3`) | 35 | 25 | 0.733 | 0.457 | 0.729 | 0.704 |
| YOLO26m | с аугм. (`yolo26m_ft_aug`) | 41 | 21 | 0.878 | **0.538** | 0.776 | 0.882 |
| RF-DETR Medium | без аугм. (`rfdetr_medium_ft`) | 40 | 15 | 0.928 | 0.865 | 0.930 | 0.896 |
| RF-DETR Medium | с аугм. (`rfdetr_medium_ft_aug`) | 40 | 34 | 0.916 | **0.872** | 0.868 | 0.913 |

> Метрики YOLO — из `results.csv` (ultralytics), RF-DETR — из `metrics.csv` (собственный
> val-эвалюатор тренера). Абсолютные значения между трекерами сравнимы лишь приблизительно;
> для отчёта важнее относительная динамика «до/после дообучения» и поведение на реальных кадрах.

**Выводы**
- Дообучение дало основной прирост обеим моделям; аугментации заметно помогли YOLO
  (mAP@50-95: 0.457 → 0.538), для RF-DETR — в пределах шума (0.865 → 0.872).
- RF-DETR Medium даёт более высокие и «уверенные» детекции: на одних и тех же кадрах
  уверенность выше (0.974 vs 0.642 на `001.jpg`), нет пропусков из 16 кадров,
  тогда как YOLO пропустил `004.jpg` и выдал дублирующиеся рамки на `001/008/012`
  (см. `results/yolo_detections.csv` vs `results/detr_detections.csv`).
- Стоимость: RF-DETR Medium существенно тяжелее (≈130 MB весов, 576 px вход) против
  ≈42 MB у YOLO26m при 640 px.

## Структура репозитория

```
.
├── dataset/                     # датасет в YOLO-формате (train/valid/test + data.yaml)
├── instances_Train1.json        # ручная COCO-разметка (часть 1)
├── instances_Train2.json        # ручная COCO-разметка (часть 2)
├── results/
│   ├── grounding/coco_preannotation.json   # предразметка Grounding DINO
│   ├── yolo/001-016.jpg                    # предсказания YOLO на тестовых кадрах
│   ├── detr/001-016.jpg                    # предсказания RF-DETR на тех же кадрах
│   ├── yolo_detections.csv                 # детекции YOLO (класс, conf, bbox)
│   └── detr_detections.csv                 # детекции RF-DETR (класс, conf, bbox)
├── runs/
│   ├── detect/runs/finetune/yolo26m_ft_aug/    # кривые, матрица ошибок, батчи, results.csv
│   ├── detect/runs/finetune/yolo26m_ft-3/      # то же для прогона без аугментаций
│   └── finetune/rfdetr_medium_ft{,_aug}/       # metrics.csv, training_config.json, hparams.yaml
├── train_yolo.py / train_rfdetr.py # обучение
├── prepare_dataset.py             # COCO → YOLO + сплиты
├── verify_mapping.py              # QA разметки
├── evaluate_test.py               # mAP на тесте (torchmetrics), до/после
├── batch_yolo.py / batch_detr.py / batch_grounding.py   # инференс на пачках фото
├── heic2jpeg.py / rename_photos.py                      # подготовка фото
└── README.md
```

Веса (`*.pt`, `*.pth`, `*.ckpt`), `.venv/`, `mlruns/` и исходные фото в полном разрешении
в репозиторий не включены — см. `.gitignore`.

## Воспроизведение

```bash
pip install -r requirements.txt

python prepare_dataset.py        # собрать data/det_dataset из json
python verify_mapping.py         # проверить разметку
python train_yolo.py             # -> runs/finetune/yolo26m_ft_aug
python train_rfdetr.py           # -> runs/finetune/rfdetr_medium_ft_aug
python evaluate_test.py          # mAP на тестовой выборке
```

## Инструменты

Python 3.13, PyTorch 2.14 (CUDA 12.6), ultralytics 8.4, rfdetr, Grounding DINO,
torchmetrics, OpenCV, Pillow. Обучение — одна RTX 4060 Laptop (8 GB).
