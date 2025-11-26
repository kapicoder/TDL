import argparse
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from PIL import Image, ImageDraw, ImageFont
import json
from ultralytics import YOLO  # type: ignore
from utils import tiff2png
with open("./config.json", "r") as cf:
    config = json.load(cf)


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


def overlay_caption(image: Image.Image, caption: str) -> None:
    """Overlay a simple caption on the top-left corner of the image."""
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    padding = 4
    text_w, text_h = measure_text(draw, caption, font)
    x1, y1 = 4, 4
    x2 = x1 + text_w + padding * 2
    y2 = y1 + text_h + padding * 2
    draw.rectangle([x1, y1, x2, y2], fill=(0, 0, 0))
    draw.text((x1 + padding, y1 + padding), caption, fill="white", font=font)


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



def main() -> None:
    parser = argparse.ArgumentParser(description="Compare original, predicted, and ground-truth annotations.")
   
    parser.add_argument("--image", type=Path, help="Path to the input image to be predicted.")
    parser.add_argument(
        "--weights",
        type=Path,
        default="model/best.pt",
        help="Path to model weights (defaults to latest result/**/weights/best.pt).",
    )
    
    parser.add_argument(
        "--output",
        type=Path,
        default="./result/result_single",
        help="Output path or directory for the composed comparison image.",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Object confidence threshold.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to perform inference on (default: cpu).",
    )
    parser.add_argument(
        "--no_label",
        action="store_true",
        help="Use label file if it exists.",
    )
    args=parser.parse_args()
    image_path = args.image
    if not image_path.exists():
        raise FileNotFoundError(f"Input image not found: {image_path}")

    weights_path = args.weights

    if not weights_path.exists():
        raise FileNotFoundError(f"Model weights not found: {weights_path}")

    model = YOLO(str(weights_path))
    img=Image.open(image_path)
    img = tiff2png.ensure_png_ready(img).convert("RGB")
    results = model.predict(
        source=img,
        conf=args.conf,
        imgsz=img.size,
        device=args.device,
        verbose=False,
        iou=0.6,
        augment=False
    )
    if not results:
        raise RuntimeError("Ultralytics did not return any prediction results.")

    result = results[0]
    
    with Image.open(image_path) as base_img:
        label_path = None
        base_img = tiff2png.ensure_png_ready(base_img).convert("RGB")   
        class_names = {int(k): v for k, v in model.names.items()}
        original_panel = base_img.copy()
        pred_boxes = prediction_boxes(result)
        prediction_panel = draw_boxes(base_img.copy(), pred_boxes, class_names)
        if not args.no_label:
            label_path = find_label_file(image_path)
        gt_boxes: List[Tuple[float, float, float, float, float, int]] = []
        #label_path = find_label_file(image_path)
        if label_path is not None:
            gt_boxes = load_ground_truth_boxes(label_path, base_img.width, base_img.height)
        ground_truth_panel = draw_boxes(base_img.copy(), gt_boxes, class_names)

        overlay_caption(original_panel, "Original")
        overlay_caption(prediction_panel, f"Prediction ({len(pred_boxes)})")
        if label_path is not None:
            overlay_caption(ground_truth_panel, f"Ground Truth ({len(gt_boxes)})")
        else:
            overlay_caption(ground_truth_panel, "Ground Truth (missing)")

        width, height = base_img.size
        composite = Image.new("RGB", (width * 3, height), color=(0, 0, 0))
        composite.paste(original_panel, (0, 0))
        composite.paste(prediction_panel, (width, 0))
        composite.paste(ground_truth_panel, (width * 2, 0))

    output_path = (Path(args.output) / image_path.name).with_suffix(".png")
    if not output_path.parent.exists():
        output_path.parent.mkdir(parents=True)
    composite.save(output_path)

    print(f"Detections: {len(pred_boxes)}")
    print(f"Ground-truth boxes: {len(gt_boxes)}")
    print(f"Comparison saved to: {output_path}")


if __name__ == "__main__":
    main()
