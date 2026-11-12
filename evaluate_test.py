import cv2
import torch
import numpy as np
from pathlib import Path
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from ultralytics import YOLO
from rfdetr import RFDETRMedium

TEST_IMG = Path("data/det_dataset/test/images")
TEST_LBL = Path("data/det_dataset/test/labels")
NAMES = {0: "laptop", 1: "cell phone"}
YOLO_COCO = {63: 0, 67: 1}
DETR_COCO = {73: 0, 77: 1}
CONF = 0.001

# Ground truth
GT = {}
for txt in sorted(TEST_LBL.glob("*.txt")):
    img = next(TEST_IMG / (txt.stem + e) for e in (".jpg", ".jpeg", ".png")
               if (TEST_IMG / (txt.stem + e)).exists())
    h, w = cv2.imread(str(img)).shape[:2]
    boxes, labels = [], []
    for line in txt.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        c, cx, cy, nw, nh = map(float, line.split())
        boxes.append([(cx - nw/2)*w, (cy - nh/2)*h, (cx + nw/2)*w, (cy + nh/2)*h])
        labels.append(int(c))
    
    # КРИТИЧНО: если боксов нет, создаем тензоры правильной формы (0, 4) и (0,)
    if boxes:
        GT[txt.stem] = (str(img), torch.tensor(boxes, dtype=torch.float32),
                        torch.tensor(labels, dtype=torch.int))
    else:
        GT[txt.stem] = (str(img), torch.zeros((0, 4), dtype=torch.float32),
                        torch.zeros((0,), dtype=torch.int))

print(f"Test: {len(GT)} фото, {sum(b.shape[0] for _, b, _ in GT.values())} боксов")

EMPTY = (torch.zeros((0, 4), dtype=torch.float32), 
         torch.zeros((0,), dtype=torch.float32), 
         torch.zeros((0,), dtype=torch.int))

def make_yolo(weights, cls_map):
    model = YOLO(weights)
    valid_classes = torch.tensor(list(cls_map.keys()))
    def fn(path):
        r = model.predict(path, conf=CONF, verbose=False)[0].boxes
        if len(r) == 0:
            return EMPTY
        # Фильтруем только нужные классы (laptop/phone)
        mask = torch.isin(r.cls.cpu().int(), valid_classes)
        r = r[mask]
        if len(r) == 0:
            return EMPTY
        return (r.xyxy.cpu(), r.conf.cpu(),
                torch.tensor([cls_map[int(c)] for c in r.cls], dtype=torch.int))
    return fn

def make_rfdetr(weights, cls_map):
    model = RFDETRMedium(pretrain_weights=weights) if weights else RFDETRMedium()
    def fn(path):
        d = model.predict(path, threshold=CONF)
        if len(d.class_id) == 0:
            return EMPTY
        mask = np.isin(d.class_id, list(cls_map.keys()))
        d = d[mask]
        if len(d.class_id) == 0:
            return EMPTY
        return (torch.tensor(d.xyxy, dtype=torch.float32),
                torch.tensor(d.confidence, dtype=torch.float32),
                torch.tensor([cls_map[int(c)] for c in d.class_id], dtype=torch.int))
    return fn

def evaluate(pred_fn, title):
    metric = MeanAveragePrecision(iou_type="bbox", class_metrics=True)
    preds, targets = [], []
    for stem, (path, boxes, labels) in GT.items():
        pb, ps, pl = pred_fn(path)
        preds.append({"boxes": pb, "scores": ps, "labels": pl})
        targets.append({"boxes": boxes, "labels": labels})
    
    # Правильный вызов update и compute
    metric.update(preds, targets)
    r = metric.compute()
    
    print(f"\n=== {title} ===")
    print(f"  mAP50-95: {r['map']:.4f} | mAP50: {r['map_50']:.4f}")
    if 'map_per_class' in r:
        for cid, name in NAMES.items():
            val = r['map_per_class'][cid] if cid < len(r['map_per_class']) else -1.0
            print(f"  AP50-95 {name}: {val:.4f}")
    else:
        print("  (Per-class metrics not available)")

# Веса
y_coco = "yolo26m.pt"
y_ft = Path("runs/detect/runs/finetune/yolo26m_ft_aug/weights/best.pt")
d_coco = None
d_ft = Path("runs/finetune/rfdetr_medium_ft_aug/checkpoint_best_ema.pth")

print("YOLO COCO:", y_coco)
print("YOLO FT:", y_ft)
print("RF-DETR COCO:", d_coco)
print("RF-DETR FT:", d_ft)

evaluate(make_yolo(y_coco, YOLO_COCO), "YOLO26m ДО (COCO)")
evaluate(make_yolo(str(y_ft), {0: 0, 1: 1}), "YOLO26m ПОСЛЕ (fine-tune)")
evaluate(make_rfdetr(d_coco, DETR_COCO), "RF-DETR Medium ДО (COCO)")
evaluate(make_rfdetr(str(d_ft), {0: 0, 1: 1}), "RF-DETR Medium ПОСЛЕ (fine-tune)")