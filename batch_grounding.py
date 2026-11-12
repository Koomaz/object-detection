import json
import sys
from pathlib import Path
from collections import Counter   # к импортам наверх

import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

CKPT = "IDEA-Research/grounding-dino-base"   # без GPU / медленно — возьми -tiny
TEXT_PROMPT = "laptop. cell phone."          # точка = конец фразы, важна для токенизации BERT
BOX_THR, TEXT_THR = 0.25, 0.20               # порог занижен: нужен recall, FP вычистим в CVAT

# метки grounding -> классы твоего CVAT-проекта
LABEL_MAP = {"laptop": "laptop", "cell phone": "cell phone"}

processor = AutoProcessor.from_pretrained(CKPT)
try:
    model = AutoModelForZeroShotObjectDetection.from_pretrained(CKPT)
except Exception as e:
    sys.exit(f"Auto-класс не распознал {CKPT}: {e}\n"
             f"Лечение: pip install \"transformers<5\" и возврат v4-импортов.")

if not hasattr(processor, "post_process_grounded_object_detection"):
    sys.exit(f"У процессора нет post_process_grounded_object_detection. Есть: "
             f"{[m for m in dir(processor) if m.startswith('post_process')]}")

device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device).eval()

EXT = {".jpg", ".jpeg", ".png"}
TRAIN_DIR = Path("data/photos")   # ТОЛЬКО картинки train-кандидатов; val/test не прогоняем (GOLD)
photos = sorted(p for p in TRAIN_DIR.iterdir() if p.suffix.lower() in EXT)

categories = [{"id": i, "name": n} for i, n in enumerate(LABEL_MAP.values(), start=1)]
name_to_id = {c["name"]: c["id"] for c in categories}
coco = {"images": [], "annotations": [], "categories": categories}
ann_id = 1
counts = Counter()   
for img_id, path in enumerate(photos, start=1):
    image = Image.open(path).convert("RGB")
    w, h = image.size
    coco["images"].append({"id": img_id, "file_name": path.name, "width": w, "height": h})

    inputs = processor(images=image, text=TEXT_PROMPT, return_tensors="pt").to(device)
    amp = torch.autocast(device_type="cuda", enabled=(device == "cuda"))  # FP16 на Tensor Cores
    with torch.no_grad(), amp:
        outputs = model(**inputs)

    res = processor.post_process_grounded_object_detection(
        outputs, inputs.input_ids,
        threshold=BOX_THR, text_threshold=TEXT_THR,  # было box_threshold=BOX_THR
        target_sizes=[(h, w)],
    )[0]

    n_kept = 0
    for box, label, score in zip(res["boxes"], res["labels"], res["scores"]):
        cls = LABEL_MAP.get(label)
        if cls is None:
            continue
        x1, y1, x2, y2 = box.tolist()
        coco["annotations"].append({
            "id": ann_id, "image_id": img_id, "category_id": name_to_id[cls],
            "bbox": [round(x1, 2), round(y1, 2), round(x2 - x1, 2), round(y2 - y1, 2)],
            "area": round((x2 - x1) * (y2 - y1), 2), "iscrowd": 0,
        })
        ann_id += 1
        n_kept += 1
        counts[cls] += 1          # новая строка
    print(f"[{img_id}/{len(photos)}] {path.name}: {n_kept} boxes kept")

             # новая строка

out = Path("results/grounding")
out.mkdir(parents=True, exist_ok=True)
(out / "coco_preannotation.json").write_text(json.dumps(coco, ensure_ascii=False), encoding="utf-8")
print(f"images: {len(coco['images'])}, boxes: {len(coco['annotations'])}")
print(f"laptops: {counts['laptop']}, cell phones: {counts['cell phone']}")