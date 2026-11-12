import csv
import cv2
import time
import torch
from collections import Counter
from pathlib import Path
import supervision as sv
from rfdetr import RFDETRMedium
from rfdetr.assets.coco_classes import COCO_CLASSES

print(f"Загрузка модели...")
model = RFDETRMedium    ()
model_name = type(model).__name__

# Оптимизация: включаем FP16 для Tensor Cores на GPU
model.inference(dtype=torch.float16)

EXT = {".jpg", ".jpeg", ".png"}
out_dir = Path("results/detr")
out_dir.mkdir(parents=True, exist_ok=True)

# ОРИГИНАЛЬНЫЕ ID классов MS COCO, которые использует RF-DETR:
# 73 = laptop, 77 = cell phone
TARGET_CLASS_IDS = {73, 77}

rows = []
processing_times = []

# Warmup: прогреваем модель перед замерами
print("\nПрогрев модели (warmup)...")
_ = model.predict("data/photos/001.jpg", threshold=0.25)
print("Модель загружена в GPU, начинаем замеры...\n")

print("Начинаем пакетную обработку (ищем только laptop и cell phone)...")
for p in sorted(Path("data/photos").iterdir()):
    if p.suffix.lower() not in EXT:
        continue
    
    # Замер времени
    start_time = time.time()
    
    # Предикт
    detections = model.predict(str(p), threshold=0.25)
    
    elapsed = time.time() - start_time
    processing_times.append(elapsed)
    
    # Фильтрация: оставляем только нужные нам оригинальные ID
    valid_indices = [
        i for i, class_id in enumerate(detections.class_id)
        if class_id in TARGET_CLASS_IDS
    ]
    filtered_detections = detections[valid_indices]
    
    # Аннотация только отфильтрованных объектов
    labels = [
        f"{COCO_CLASSES[class_id]} {filtered_detections.confidence[i]:.2f}" 
        for i, class_id in enumerate(filtered_detections.class_id)
    ]
    annotated_image = sv.BoxAnnotator().annotate(
        filtered_detections.metadata["source_image"], 
        filtered_detections
    )
    annotated_image = sv.LabelAnnotator().annotate(
        annotated_image, 
        filtered_detections, 
        labels
    )
    
    # Конвертируем RGB в BGR перед сохранением
    bgr_image = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(out_dir / p.name), bgr_image)
    
    # Сбор данных для CSV
    for i, class_id in enumerate(filtered_detections.class_id):
        class_name = COCO_CLASSES[class_id]
        conf = round(float(filtered_detections.confidence[i]), 3)
        x1, y1, x2, y2 = map(round, filtered_detections.xyxy[i])
        rows.append([p.name, "rfdetr-nano", class_name, conf, x1, y1, x2, y2])
        
    print(f"  {p.name}: {len(filtered_detections.class_id)} объектов за {elapsed:.3f} сек")

# Сохранение CSV
Path("results").mkdir(exist_ok=True)
with open("results/detr_detections.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["file", "model", "class", "conf", "x1", "y1", "x2", "y2"])
    w.writerows(rows)

# Статистика по времени
avg_time = sum(processing_times) / len(processing_times) if processing_times else 0
fps = 1 / avg_time if avg_time > 0 else 0

print(f"\n=== ИТОГИ {model_name} ===")
print(f"Всего детекций (только laptop/cell phone): {len(rows)}")
print("По классам:", dict(Counter(r[2] for r in rows)))
print(f"\n️  Время обработки:")
print(f"  Среднее на изображение: {avg_time:.3f} сек")
print(f"  FPS: {fps:.1f}")
print(f"  Всего изображений: {len(processing_times)}")