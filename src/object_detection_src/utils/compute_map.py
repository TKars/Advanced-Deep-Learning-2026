"""mAP / mAP@0.75 на eval через torchmetrics (две метрики в одном проходе по батчам)."""

from __future__ import annotations

import torch
from tqdm.auto import tqdm

from prior_boxes import detect_objects
from torchmetrics.detection import MeanAveragePrecision

# метки как в voc_dataset (0 = background, в GT обычно только 1 и 2)
DEFAULT_CLASS_NAMES = {1: "car", 2: "license_plate"}


def _batch_preds_targets(
    images: torch.Tensor,
    box_ss: list,
    label_ss: list,
    pred_box_ss: list,
    pred_label_ss: list,
    pred_score_ss: list,
    device: torch.device,
):
    preds, targets = [], []
    for i in range(images.shape[0]):
        b, s, lab = pred_box_ss[i], pred_score_ss[i], pred_label_ss[i]
        if b.numel() == 0:
            b = torch.zeros(0, 4, device=device)
            s = torch.zeros(0, device=device)
            lab = torch.zeros(0, dtype=torch.long, device=device)
        else:
            b = b.reshape(-1, 4).float()
            s = s.reshape(-1).float()
            lab = lab.reshape(-1).long()
        gt_b = box_ss[i].to(device).float().reshape(-1, 4)
        gt_l = label_ss[i].to(device).long().reshape(-1)
        preds.append({"boxes": b.cpu(), "scores": s.cpu(), "labels": lab.cpu()})
        targets.append({"boxes": gt_b.cpu(), "labels": gt_l.cpu()})
    return preds, targets


@torch.no_grad()
def compute_map(
    model,
    data_loader,
    priors,
    num_classes,
    device,
    overlap_threshold=0.5,
    conf_threshold=0.4,
):
    """Возвращает dict: map, map_75, per_class (mAP COCO-style и mAP@0.75 по классам)."""
    model.eval()
    m = MeanAveragePrecision(box_format="xyxy", class_metrics=True)
    m75 = MeanAveragePrecision(box_format="xyxy", class_metrics=True, iou_thresholds=[0.75])

    for images, box_ss, label_ss in tqdm(data_loader, desc="mAP (eval)", leave=False):
        if images.numel() == 0:
            continue
        images = images.to(device)
        loc_pred, conf_pred = model(images)
        pb, pl, ps = detect_objects(
            loc_pred,
            conf_pred,
            priors,
            num_classes,
            overlap_threshold,
            conf_threshold,
        )
        preds, targets = _batch_preds_targets(images, box_ss, label_ss, pb, pl, ps, device)
        m.update(preds, targets)
        m75.update(preds, targets)

    o, o75 = m.compute(), m75.compute()
    m.reset()
    m75.reset()

    cls_ids = o["classes"].tolist()
    maps = o["map_per_class"].tolist()
    maps75 = o75["map_per_class"].tolist()
    per_class = []
    for c, a, b in zip(cls_ids, maps, maps75):
        per_class.append(
            {
                "cls": int(c),
                "map": float(a) if a >= 0 else float("nan"),
                "map_75": float(b) if b >= 0 else float("nan"),
            }
        )

    return {
        "map": float(o["map"]),
        "map_75": float(o["map_75"]),
        "per_class": per_class,
    }


def log_eval_map(tag: str, d: dict, class_names: dict[int, str] | None = None) -> None:
    """Краткий вывод в лог / ноутбук."""
    names = class_names or DEFAULT_CLASS_NAMES
    print(f"{tag}  mAP={d['map']:.4f}  mAP@0.75={d['map_75']:.4f}")
    for row in d["per_class"]:
        nm = names.get(row["cls"], str(row["cls"]))
        print(f"    cls {row['cls']} ({nm}): mAP={row['map']:.4f}  mAP@0.75={row['map_75']:.4f}")
