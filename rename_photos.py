import re
import sys
from pathlib import Path

PHOTOS_DIR = Path("data/photos")
EXT = {".jpg", ".jpeg", ".png"}
PAD = 3
FMT = re.compile(r"^\d{3}$")


def is_image(p: Path) -> bool:
    return p.is_file() and p.suffix.lower() in EXT


def main() -> None:
    apply = "--apply" in sys.argv

    files = [p for p in PHOTOS_DIR.iterdir() if is_image(p)]
    n = len(files)
    if n == 0:
        print("В папке нет картинок.")
        return
    if n > 10**PAD - 1:
        sys.exit(f"Файлов {n}: не хватит {PAD} цифр для нумерации.")

    expected = {f"{i:0{PAD}d}" for i in range(1, n + 1)}
    actual = {p.stem for p in files}
    if actual == expected:
        print(f"Всё правильно: {n} файлов, 001..{n:0{PAD}d} без пропусков и дублей.")
        return

    # диагностика: что именно не так
    nums = [int(p.stem) for p in files if FMT.match(p.stem)]
    bad_fmt = [p.name for p in files if not FMT.match(p.stem)]
    missing = [i for i in range(1, max(nums, default=0) + 1) if i not in set(nums)]
    dupes = sorted({s for s in (p.stem for p in files) if sum(p.stem == s for p in files) > 1})
    print(f"Файлов: {n}. Вне формата: {bad_fmt or 'нет'}. Пропуски: {missing or 'нет'}. Дубли: {dupes or 'нет'}.")

    # план: ВСЕ файлы перенумеровываем непрерывно, порядок = время съёмки
    files.sort(key=lambda p: (p.stat().st_mtime, p.name))
    plan = [(p, PHOTOS_DIR / f"{i:0{PAD}d}{p.suffix.lower()}")
            for i, p in enumerate(files, start=1)]
    plan = [(s, d) for s, d in plan if s.name != d.name]

    print(f"К переименованию: {len(plan)} из {n}.")
    for src, dst in plan:
        print(f"  {src.name}  ->  {dst.name}")

    if not apply:
        print("\nЭто ПРЕВЬЮ. Чтобы применить: python rename_photos.py --apply")
        return

    # двухфазно: сначала все во временные имена, иначе 010->009 может столкнуться с ещё не уехавшим 009-соседом
    tmp = []
    for src, dst in plan:
        t = src.with_name(src.name + ".renaming")
        src.rename(t)
        tmp.append((t, dst))
    for t, dst in tmp:
        t.rename(dst)
    print(f"Готово, переименовано: {len(plan)}.")


if __name__ == "__main__":
    main()