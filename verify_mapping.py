import json
import random
from pathlib import Path
import cv2

COCO_FILES = ["instances_Train1.json", "instances_Train2.json"]
IMG_DIR = Path("data/photos")
OUT = Path("verify")
SAMPLE = 12
SEED = 42
TOLERANCE = 5   # px: допуск на вылет рамки за край (_people рисуют ±1-2 px, это норма)

# --- Собираем из json: имя -> (ширина, высота) и имя -> [боксы] ---
images, anns = {}, {}
for cf in COCO_FILES:
    data = json.loads(Path(cf).read_text(encoding="utf-8"))
    id2fn = {im["id"]: im["file_name"].strip() for im in data["images"]}
    for im in data["images"]:
        fn = im["file_name"].strip()
        images[fn] = (im["width"], im["height"])
        anns.setdefault(fn, [])
    for a in data["annotations"]:
        anns[id2fn[a["image_id"]]].append(a["bbox"])

print(f"Изображений в json: {len(images)}")
print(f"Боксов в json: {sum(len(v) for v in anns.values())}")

# --- Сито 1: файл существует? Сито 2: размеры совпадают? ---
missing, dim_bad, oob = [], [], []
for fn, (w, h) in sorted(images.items()):
    p = IMG_DIR / fn
    if not p.exists():
        missing.append(fn)
        continue
    img = cv2.imread(str(p))
    if img is None:
        missing.append(fn + " (не читается)")
        continue
    H, W = img.shape[:2]
    if (W, H) != (w, h):
        dim_bad.append((fn, f"json={w}x{h}", f"disk={W}x{H}"))
    # --- Сито 3: боксы влезают в кадр? ---
    for (x, y, bw, bh) in anns[fn]:
        if x < -TOLERANCE or y < -TOLERANCE or x + bw > W + TOLERANCE or y + bh > H + TOLERANCE:
            oob.append((fn, [round(v) for v in (x, y, bw, bh)]))

print(f"\nНе найдено на диске: {missing if missing else '— все на месте'}")
print(f"Несовпадений размеров: {len(dim_bad)}")
for row in dim_bad[:10]:
    print("  ", row)
print(f"Боксов за границами кадра: {len(oob)}")
for row in oob[:10]:
    print("  ", row)

# --- Сито 4: визуальная выборка с рамками поверх ---
random.seed(SEED)
forced = [fn for fn in ["031.jpg", "033.jpg", "046.jpg", "060.jpg", "075.jpg", "089.jpg"] if fn in images]
pool = [fn for fn in sorted(images) if fn not in forced]
sample = forced + random.sample(pool, SAMPLE - len(forced))

OUT.mkdir(exist_ok=True)
for fn in sample:
    img = cv2.imread(str(IMG_DIR / fn))
    if img is None:
        continue
    for (x, y, bw, bh) in anns[fn]:
        cv2.rectangle(img, (int(x), int(y)), (int(x + bw), int(y + bh)), (0, 255, 0), 3)
    cv2.putText(img, fn, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
    cv2.imwrite(str(OUT / fn), img)
print(f"\nСохранено {len(sample)} картинок с рамками в {OUT}/ — проверь глазами")

# --- Вердикт ---
if not missing and not dim_bad and not oob:
    print("✅ ВЕРДИКТ: связность 'имя ↔ разметка' целостна, можно обучаться")
else:
    print("⚠️ ВЕРДИКТ: найдены расхождения, смотри строки выше")