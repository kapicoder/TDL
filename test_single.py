import argparse
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO  # type: ignore
from utils import tiff2png
from utils.config import CONFIG
from utils.showtiff_withLabel import visual_img

def class_color(index: int) -> Tuple[int, int, int]:
    """Return a deterministic color for the given class index."""
    palette = [
        (255, 59, 48),
        (0, 122, 255),
        (88, 86, 214),
        (255, 149, 0),
        (52, 199, 135),
        (255, 204, 0),
    ]
    return palette[index % len(palette)]


def measure_text(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> Tuple[int, int]:
    """Return (width, height) for the provided text."""
    if hasattr(font, "getbbox"):
        left, top, right, bottom = font.getbbox(text)
        return right - left, bottom - top
    return draw.textsize(text, font=font)


def draw_boxes(
    image: Image.Image,
    boxes: Sequence[Tuple[float, float, float, float, float, int]],
    class_names: Dict[int, str],
) -> Image.Image:
    """Draw detections on the image."""
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    for x1, y1, x2, y2, score, cls_idx in boxes:
        color = class_color(cls_idx)
        draw.rectangle([(x1, y1), (x2, y2)], outline=color, width=3)

        label = class_names.get(cls_idx, str(cls_idx))
        caption = f"{label} {score:.2f}" if score >= 0 else label
        text_w, text_h = measure_text(draw, caption, font)
        text_x1 = x1
        text_y1 = max(0.0, y1 - text_h - 6)
        text_x2 = text_x1 + text_w + 6
        text_y2 = text_y1 + text_h + 6

        draw.rectangle([(text_x1, text_y1), (text_x2, text_y2)], fill=color)
        draw.text((text_x1 + 3, text_y1 + 3), caption, fill="white", font=font)

    return image


def prediction_boxes(result) -> List[Tuple[float, float, float, float, float, int]]:
    """Convert Ultralytics result boxes to a uniform tuple format."""
    boxes: List[Tuple[float, float, float, float, float, int]] = []
    if not getattr(result, "boxes", None):
        return boxes

    xyxy = result.boxes.xyxy.cpu().numpy()
    confs = result.boxes.conf.cpu().numpy()
    classes = result.boxes.cls.cpu().numpy().astype(int)

    for (x1, y1, x2, y2), conf, cls_idx in zip(xyxy, confs, classes):
        boxes.append((float(x1), float(y1), float(x2), float(y2), float(conf), int(cls_idx)))
    return boxes


def load_ground_truth_boxes(
    label_path: Path,
    width: int,
    height: int,
) -> List[Tuple[float, float, float, float, float, int]]:
    """Load YOLO-format annotations and convert them to absolute xyxy coordinates."""
    boxes: List[Tuple[float, float, float, float, float, int]] = []
    if not label_path.exists():
        return boxes

    with label_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            try:
                cls_idx = int(parts[0])
                xc, yc, w, h = map(float, parts[1:])
            except ValueError:
                continue

            box_w = w * width
            box_h = h * height
            x_center = xc * width
            y_center = yc * height

            x1 = max(0.0, x_center - box_w / 2)
            y1 = max(0.0, y_center - box_h / 2)
            x2 = min(float(width), x_center + box_w / 2)
            y2 = min(float(height), y_center + box_h / 2)

            boxes.append((x1, y1, x2, y2, -1.0, cls_idx))
    return boxes


def find_label_file(image_path: Path) -> Optional[Path]:
    """Infer the corresponding label file for an image within a YOLO dataset layout."""
    if image_path.parent.name == "images":
        candidate = image_path.parent.parent / "labels" / f"{image_path.stem}.txt"
        if candidate.exists():
            return candidate

    candidate = image_path.with_suffix(".txt")
    if candidate.exists():
        return candidate

    for parent in image_path.parents:
        labels_dir = parent / "labels"
        if labels_dir.is_dir():
            candidate = labels_dir / f"{image_path.stem}.txt"
            if candidate.exists():
                return candidate
    return None

def _normalize_box(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    width: int,
    height: int,
) -> Tuple[float, float, float, float]:
    x1 = max(0.0, min(float(width), x1))
    x2 = max(0.0, min(float(width), x2))
    y1 = max(0.0, min(float(height), y1))
    y2 = max(0.0, min(float(height), y2))
    if x2 <= x1 or y2 <= y1:
        return 0.0, 0.0, 0.0, 0.0
    xc = ((x1 + x2) / 2.0) / width
    yc = ((y1 + y2) / 2.0) / height
    bw = (x2 - x1) / width
    bh = (y2 - y1) / height
    return xc, yc, bw, bh

def write_label_file(
    label_path: Path, boxes: List[dict], width: int, height: int
) -> int:
    label_path.parent.mkdir(parents=True, exist_ok=True)
    lines: List[str] = []
    for box in boxes:
        xc, yc, bw, bh = _normalize_box(*box["xyxy"], width, height)
        if bw <= 0 or bh <= 0:
            continue
        lines.append(f"{box['cls']} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")
    label_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"已写入 {len(lines)} 个预测框 -> {label_path}")
    return len(lines)

def run_single_test(config: CONFIG, visualize: bool = True) -> Dict[str, object]:
    """Run single-image test, save composite PNG/labels, and optionally visualize."""
    image_path_raw = config["test_img_path"]
    if image_path_raw is None:
        raise ValueError("未找到 test_img_path，请在 config.json 或命令行中提供。")
    image_path = Path(image_path_raw)
    weights_path = Path(config["test_weights_path"] or "model/best.pt")
    output_root = Path(config["test_output_path"] or "./result/result_single")
    conf_thres = float(config["test_conf"] if config["test_conf"] is not None else 0.25)
    iou_thres = float(config["test_iou"] if config["test_iou"] is not None else 0.6)
    device = config["test_device"] or config["device"] or "cuda"
    use_label = config["use_label"] if config["use_label"] is not None else True
    augment = config["test_augment"] if config["test_augment"] is not None else False
    patch_root = Path(config["visualization_patch_output_dir"])
    patch_size_raw = (config["visualization_target_patch_size"])
    patch_size = int(patch_size_raw)
    use_latlon = config["use_latlon"] if config["use_latlon"] is not None else True
    export_patches = config["export_patches"] if config["export_patches"] is not None else True

    if not image_path.exists():
        raise FileNotFoundError(f"Input image not found: {image_path}")
    if not weights_path.exists():
        raise FileNotFoundError(f"Model weights not found: {weights_path}")

    model = YOLO(str(weights_path))
    img = Image.open(image_path)
    img = tiff2png.ensure_png_ready(img).convert("RGB")
    results = model.predict(
        source=img,
        conf=conf_thres,
        imgsz=img.size,
        device=device,
        verbose=False,
        iou=iou_thres,
        augment=augment,
    )
    if not results:
        raise RuntimeError("Ultralytics did not return any prediction results.")

    result = results[0]
    with Image.open(image_path) as base_img:
        pred_boxes = prediction_boxes(result)
        if use_label:
            label_path = find_label_file(image_path)


        width, height = base_img.size

    output_path = (output_root / image_path.name).with_suffix(".png")
    output_path.parent.mkdir(parents=True, exist_ok=True)
  
    print(f"Detections: {len(pred_boxes)}")
    print(f"Comparison saved to: {output_path}")
    gt_label_for_vis = str(label_path) if label_path is not None else None
    pred_label_path = (output_root / image_path.stem / image_path.name).with_suffix(".txt")
    pred_boxes_for_label = [
        {"xyxy": (x1, y1, x2, y2), "cls": cls_idx} for x1, y1, x2, y2, _, cls_idx in pred_boxes
    ]
    write_label_file(
        Path(pred_label_path),
        pred_boxes_for_label,
        width,
        height,
    )
    if visualize:
        visual_img(
            tiff_path=str(image_path),
            pred_label_path=str(pred_label_path),
            gt_label_path=gt_label_for_vis,
            use_latlon=use_latlon,
            export_patches=export_patches,
            patch_output_dir=patch_root,
            patch_size=patch_size,
        )

    return {
        "image": image_path,
        "weights": weights_path,
        "output": output_path,
        "pred_label": Path(pred_label_path),
        "gt_label": label_path,
        "detections": len(pred_boxes),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare original, predicted, and ground-truth annotations.")
    parser.add_argument("--test_img_path", type=Path, help="Path to the input image to be predicted (config default).")
    parser.add_argument(
        "--test_weights_path",
        type=Path,
        help="Path to model weights (default: config test_weights_path or model/best.pt).",
    )
    parser.add_argument(
        "--test_output_path",
        type=Path,
        help="Output directory for the composed comparison image (defaults to config).",
    )
    parser.add_argument("--test_conf", type=float, help="Object confidence threshold.")
    parser.add_argument("--test_iou", type=float, help="IoU threshold.")
    parser.add_argument("--device", type=str, help="Device to perform inference on (defaults to config or cuda).")
    parser.add_argument("--patch_size", type=int, help="Patch size for visualization export.")
    parser.add_argument("--no_label", action="store_true", help="Skip loading ground-truth label.")
    parser.add_argument("--no_visual", action="store_true", help="Disable visualization export.")
    args = parser.parse_args()

    config = CONFIG()
    overrides: Dict[str, object] = {
        "test_img_path": args.test_img_path,
        "test_weights_path": args.test_weights_path,
        "test_output_path": args.test_output_path,
        "test_conf": args.test_conf,
        "test_iou": args.test_iou,
        "test_device": args.device,
        "target_patch_size": args.patch_size,
    }
    config.update_config(**overrides)
    if args.no_label:
        config.update_config(use_label=False)

    run_single_test(config, visualize=not args.no_visual)


if __name__ == "__main__":
    main()
