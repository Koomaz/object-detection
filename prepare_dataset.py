import json
import random
import shutil
from pathlib import Path

COCO_FILES = [Path("instances_Train1.json"), Path("instances_Train2.json")]
SRC_IMAGES = Path("data/photos")          # все 200 кадров (без 032)
OUT = Path("data/det_dataset")
SEED = 42

# COCO category_id -> наш класс: 0=laptop, 1=cell phone
CAT_MAP = {2: 0, 1: 1}

records = {}
for cf in COCO_FILES:
    data = json.loads(cf.read_text(encoding="utf-8"))
    img_meta = {i["id"]: i for i in data["images"]}
    for ann in data["annotations"]:
        img = img_meta[ann["image_id"]]
        fn, W, H = img["file_name"].strip(), img["width"], img["height"]
        x, y, w, h = ann["bbox"]
        cls = CAT_MAP[ann["category_id"]]
        records.setdefault(fn, []).append(
            f"{cls} {(x + w/2)/W:.6f} {(y + h/2)/H:.6f} {w/W:.6f} {h/H:.6f}"
        )

print(f"Изображений с аннотациями: {len(records)}")
print(f"Всего боксов: {sum(len(v) for v in records.values())}")

missing = [fn for fn in records if not (SRC_IMAGES / fn).exists()]
if missing:
    print(f"⚠️ Нет файлов на диске: {missing[:10]}")
    exit(1)
print("✅ Все файлы найдены в data/photos")

# Разбиение 70/15/15
random.seed(SEED)
files = sorted(records)
random.shuffle(files)
n = len(files)
n_tr, n_va = int(n * 0.70), int(n * 0.15)
splits = {"train": files[:n_tr], "valid": files[n_tr:n_tr+n_va], "test": files[n_tr+n_va:]}

for split, fns in splits.items():
    (OUT / split / "images").mkdir(parents=True, exist_ok=True)
    (OUT / split / "labels").mkdir(parents=True, exist_ok=True)
    n_lap = n_ph = 0
    for fn in fns:
        shutil.copy2(SRC_IMAGES / fn, OUT / split / "images" / fn)
        lines = records[fn]
        (OUT / split / "labels" / (Path(fn).stem + ".txt")).write_text("\n".join(lines), encoding="utf-8")
        n_lap += sum(1 for l in lines if l.startswith("0 "))
        n_ph  += sum(1 for l in lines if l.startswith("1 "))
    print(f"📁 {split}: {len(fns)} фото | laptop={n_lap} | phone={n_ph}")

(OUT / "data.yaml").write_text(f"""path: {OUT.resolve()}
train: train/images
val: valid/images
test: test/images

names:
  0: laptop
  1: cell phone
""", encoding="utf-8")
print("\n🎉 Готово: data/det_dataset + data.yaml")