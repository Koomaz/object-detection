"""HEIC -> JPEG для датасета.
Запуск:  python convert_heic.py [src_dir] [dst_dir]
По умолчанию: data/raw -> data/photos
"""
import sys
from pathlib import Path

from PIL import Image, ImageOps
import pillow_heif

pillow_heif.register_heif_opener()          # учит Pillow открывать .heic/.heif

SRC = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/raw")
DST = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("data/photos")
QUALITY = 92

DST.mkdir(parents=True, exist_ok=True)
heics = sorted(p for p in SRC.iterdir() if p.suffix.lower() in {".heic", ".heif"})
converted, skipped = 0, 0

for src in heics:
    dst = DST / (src.stem + ".jpg")         # 017.HEIC -> 017.jpg, имя сохраняется
    if dst.exists():
        print(f"[skip] {dst.name} уже существует")
        skipped += 1
        continue

    img = Image.open(src)
    img = ImageOps.exif_transpose(img)      # поворот из EXIF применяем к пикселям
    img = img.convert("RGB")                # HEIC часто 10-bit -> приводим к uint8
    img.save(dst, "JPEG", quality=QUALITY)

    print(f"[ok]   {src.name} -> {dst.name}  {img.size[0]}x{img.size[1]}")
    converted += 1

print(f"done: converted={converted}, skipped={skipped}, found_heic={len(heics)}")