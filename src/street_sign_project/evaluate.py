from ultralytics.utils.metrics import bbox_iou
from torch import tensor
from loguru import logger
from street_sign_project.utils import project_root
import yaml
import typer

app = typer.Typer()

MODELS_PATH = project_root() / "models"
MODELS_QUALITY_YAML = project_root() / "models" / "models_quality.yaml"
TEST_IMAGES = project_root() / "data" / "preprocessed" / "test" / "images"
TEST_LABELS = project_root() / "data" / "preprocessed" / "test" / "labels"


def evaluate_tp_fp_fn(preds, labels, iou_threshold: float = 0.5) -> float:
    if len(labels) == 0:
        return 0, len(preds.boxes), 0
    img_h, img_w = preds.orig_shape
    pred_boxes = preds.boxes.xyxy
    xywh = [line[1:5] for line in labels]
    if xywh == [[]]:
        return 0, len(preds.boxes), 0
    label_boxes = [
        [
            xywh[i][0] * img_w - 0.5 * xywh[i][2] * img_w,
            xywh[i][1] * img_h - 0.5 * xywh[i][3] * img_h,
            xywh[i][0] * img_w + 0.5 * xywh[i][2] * img_w,
            xywh[i][1] * img_h + 0.5 * xywh[i][3] * img_h,
        ]
        for i in range(len(xywh))
    ]
    pred_map = []
    matched_labels = []
    pred_cls = preds.boxes.cls
    labels_cls = [line[0] for line in labels]
    count_labels = len(labels_cls)
    fp = 0
    tp = 0
    for i, pred_box in enumerate(pred_boxes):
        best_iou = 0
        best_idx = -1
        for j, label_box in enumerate(label_boxes):
            if j in matched_labels:
                continue
            label_box_tensor = tensor(label_box, device=pred_box.device)
            iou = bbox_iou(pred_box.unsqueeze(0), label_box_tensor.unsqueeze(0), xywh=False).item()
            if iou > best_iou:
                best_iou = iou
                best_idx = j
        if best_iou > iou_threshold:
            pred_map.append([i, best_idx])
            matched_labels.append(best_idx)
        else:
            fp += 1
    for idx_p, idx_l in pred_map:
        if int(pred_cls[idx_p]) == int(labels_cls[idx_l]):
            tp += 1
        else:
            fp += 1
    return tp, fp, count_labels - tp


@app.command()
def models_quality_yaml():
    from street_sign_project.model import YOLOv26  # darf nicht oben stehen, da sonst import zyklus

    logger.info("Creating...")
    quality_dict = {}
    for model_path in MODELS_PATH.glob("*.pt"):
        try:
            model = YOLOv26(model_path.name)
        except Exception as e:
            raise ValueError(f"Fehler beim Laden von {model_path.stem}: {e}")
        tp_global, fp_global = 0, 0
        for image in TEST_IMAGES.glob("*.jpg"):
            label_path = TEST_LABELS / f"{image.stem}.txt"
            labels = []
            if label_path.exists():
                with open(label_path, "r") as f:
                    for line in f.readlines():
                        labels.append([float(x) for x in line.strip().split()])
            preds = model.predict(image)
            tp, fp, _ = evaluate_tp_fp_fn(preds, labels, iou_threshold=0.5)
            tp_global += tp
            fp_global += fp
            print(tp)
        if (tp_global + fp_global) > 0:
            precision_50 = tp_global / (tp_global + fp_global)
        else:
            precision_50 = 0.0
        quality_dict[model_path.name] = precision_50
    while len(quality_dict) > 3:
        current_models = list(quality_dict.keys())
        current_values = list(quality_dict.values())
        min_idx = tensor(current_values).argmin().item()
        worst_model = current_models[min_idx]
        logger.warning(f"Due to too many models {worst_model} is not included in the yaml an should be deleted")
        del quality_dict[worst_model]

    with open(str(MODELS_QUALITY_YAML), "w") as f:
        yaml.safe_dump(quality_dict, f)


@app.command()
def Lueckenfüller():
    print("Lückenfüller, da typer mehrere app commands pro Datei braucht")


if __name__ == "__main__":
    app()
