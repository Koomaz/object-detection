import csv
import time
from collections import Counter
from pathlib import Path
from ultralytics import YOLO

MODEL_NAME = "YOLO26 Medium"

# Загрузка модели
print(f"Загрузка модели {MODEL_NAME}...")
model = YOLO("yolo26m.pt")

EXT = {".jpg", ".jpeg", ".png"}
out_dir = Path("results/yolo")
out_dir.mkdir(parents=True, exist_ok=True)

rows = []
processing_times = []

# Warmup: прогреваем модель перед замерами
print("\nПрогрев модели (warmup)...")
_ = model.predict(
    source="data/photos/001.jpg",
    conf=0.25,
    imgsz=640,
    classes=[63, 67],
    verbose=False
)
print("Модель загружена в GPU, начинаем замеры...\n")

print(f"Начинаем пакетную обработку (ищем только laptop и cell phone)...")
for p in sorted(Path("data/photos").iterdir()):
    if p.suffix.lower() not in EXT:
        continue
    
    # Замер времени
    start_time = time.time()
    
    # Предикт с FP16 оптимизацией
    results = model.predict(
        source=str(p),
        conf=0.25,
        imgsz=1280,
        classes=[63, 67],  # 63=laptop, 67=cell phone в сжатом списке Ultralytics
        save=True,
        save_dir=out_dir,
        exist_ok=True,
        verbose=False
    )
    
    elapsed = time.time() - start_time
    processing_times.append(elapsed)
    
    # Сбор данных для CSV
    for r in results:
        for b in r.boxes:
            cid = int(b.cls)
            x1, y1, x2, y2 = b.xyxy.tolist()[0]
            rows.append([
                Path(r.path).name,
                MODEL_NAME,
                r.names[cid],
                round(float(b.conf), 3),
                round(x1), round(y1), round(x2), round(y2)
            ])
    
    print(f"  {p.name}: {len(results[0].boxes)} объектов за {elapsed:.3f} сек")

# Сохранение CSV
Path("results").mkdir(exist_ok=True)
with open("results/yolo_detections.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["file", "model", "class", "conf", "x1", "y1", "x2", "y2"])
    w.writerows(rows)

# Статистика по времени
avg_time = sum(processing_times) / len(processing_times) if processing_times else 0
fps = 1 / avg_time if avg_time > 0 else 0

print(f"\n=== ИТОГИ {MODEL_NAME} ===")
print(f"Всего детекций (только laptop/cell phone): {len(rows)}")
print("По классам:", dict(Counter(r[2] for r in rows)))
print(f"\n⏱️  Время обработки:")
print(f"  Среднее на изображение: {avg_time:.3f} сек")
print(f"  FPS: {fps:.1f}")
print(f"  Всего изображений: {len(processing_times)}")