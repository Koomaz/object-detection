from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO("yolo26m.pt")

    model.train(
        data="data/det_dataset/data.yaml",
        epochs=60,
        patience=20,
        imgsz=640,
        batch=8,               # если OOM: 4
        device=0,
        seed=42,
        project="runs/finetune",
        name="yolo26m_ft_aug",
        # --- бережный fine-tuning: бережём COCO-признаки ---
        optimizer="AdamW",
        lr0=0.0005,            # вместо авто-0.001667, который всё сломал в v1
        lrf=0.01,
        warmup_epochs=1.0,
        freeze=10,             # бэкбон заморожен, учим neck+head
        # --- усиленная онлайн-аугментация ---
        degrees=20.0,          # повороты ±20° (имитация кривой съёмки с руки)
        translate=0.2,         # сдвиги кадра
        scale=0.7,             # зум ±70% (объекты ближе/дальше)
        shear=2.0,             # лёгкий сдвиг перспективы
        perspective=0.0,       # тяжёлую перспективу не включаем
        fliplr=0.5,
        flipud=0.0,            # ноутбуки вверх ногами не бывают
        hsv_h=0.02, hsv_s=0.8, hsv_v=0.5,   # освещение/цвет
        erasing=0.3,           # случайное затирание кусков (учит при occlusion)
        mosaic=1.0,
        close_mosaic=20,
        mixup=0.1,             # лёгкое смешивание кадров
    )
    print("Лучшие веса: runs/finetune/yolo26m_ft_aug/weights/best.pt")