import argparse
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO  # type: ignore
from utils import tiff2png
from utils.config import CONFIG
from utils.showtiff_withLabel import visual_img,_draw_panel,_load_boxes,_prepare_image_for_save,pixel_to_lonlat,_save_panel_image
import matplotlib.pyplot as plt
import rasterio
try:
    from pyproj import Transformer
except Exception:  # pragma: no cover - optional dependency
    Transformer = None  # type: ignore
"""
单图推理脚本，用于对单个 TIFF 图像进行目标检测推理和可视化。
支持的图像格式仅包括 TIFF。
可选地加载与图像对应的标签文件进行对比。
推理结果只包含整张大图的识别结果，不包含裁剪之后的子图结果。
可视化结果会输出为 PNG 格式，并生成对应的标注文本文件。
"""
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


def _prepare_overlay_image(img_array: np.ndarray) -> np.ndarray:
    """Convert rasterio (C, H, W) array to uint8 RGB (H, W, 3) for drawing."""
    arr = img_array.astype(np.uint8)
    if arr.ndim == 2:
        arr = np.expand_dims(arr, axis=0)
    channels, _, _ = arr.shape
    if channels == 1:
        arr = np.repeat(arr, 3, axis=0)
    elif channels == 2:
        arr = np.concatenate([arr, arr[:1]], axis=0)
    elif channels > 3:
        arr = arr[:3, :, :]
    return arr.transpose(1, 2, 0)


def _clip_box_to_image(
    x1: float, y1: float, x2: float, y2: float, width: int, height: int
) -> Optional[Tuple[float, float, float, float]]:
    """Ensure boxes stay within image bounds; return None if invalid."""
    x1_c = max(0.0, min(float(width - 1), x1))
    y1_c = max(0.0, min(float(height - 1), y1))
    x2_c = max(0.0, min(float(width), x2))
    y2_c = max(0.0, min(float(height), y2))
    if x2_c <= x1_c or y2_c <= y1_c:
        return None
    return x1_c, y1_c, x2_c, y2_c


def save_detection_tiff(
    source_tiff: Path,
    boxes: List[Tuple[float, float, float, float, float, int]],
    output_path: Path,
) -> Path:
    """Save a 16-bit single-band TIFF (I;16) with detection boxes burned into the first band, keeping geoinfo."""
    with rasterio.open(source_tiff) as src:
        # 仅取首个波段，输出保持 I;16 模式
        band1 = src.read(1)
        profile = src.profile.copy()

    # 转为 uint16，避免负值
    overlay = np.clip(band1, 0, np.iinfo(np.uint16).max).astype(np.uint16, copy=True)
    height, width = overlay.shape
    val_draw = np.iinfo(np.uint16).max  # use pure white for box strokes
    thickness = 3
    for x1, y1, x2, y2, _, cls_idx in boxes:
        clipped = _clip_box_to_image(x1, y1, x2, y2, width, height)
        if clipped is None:
            continue
        cx1, cy1, cx2, cy2 = clipped
        x1_i, y1_i, x2_i, y2_i = map(int, (np.floor(cx1), np.floor(cy1), np.ceil(cx2), np.ceil(cy2)))
        x1_i = max(0, min(width - 1, x1_i))
        y1_i = max(0, min(height - 1, y1_i))
        x2_i = max(x1_i + 1, min(width, x2_i))
        y2_i = max(y1_i + 1, min(height, y2_i))
        overlay[y1_i:y1_i + thickness, x1_i:x2_i] = val_draw
        overlay[max(0, y2_i - thickness):y2_i, x1_i:x2_i] = val_draw
        overlay[y1_i:y2_i, x1_i:x1_i + thickness] = val_draw
        overlay[y1_i:y2_i, max(0, x2_i - thickness):x2_i] = val_draw

    output_path.parent.mkdir(parents=True, exist_ok=True)
    for key in ("blockxsize", "blockysize", "tiled", "interleave", "compress"):
        profile.pop(key, None)
    profile.update(
        driver="GTiff",
        height=overlay.shape[0],
        width=overlay.shape[1],
        count=1,
        dtype="uint16",
    )
    # 单波段灰度，移除可能遗留的 RGB/alpha 信息，保持地理元数据
    profile["photometric"] = "MINISBLACK"
    profile.pop("alpha", None)
    profile.pop("colormap", None)
    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(overlay, 1)
    return output_path


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


def show_img(
    tiff_path: str,
    pred_label_path: str,
    gt_label_path: Optional[str],
    use_latlon: bool,
    export_patches: bool,
    patch_output_dir: Path,
    patch_size: int,
    config: CONFIG,
) -> None:
    # 输出路径定义
    output_dir = Path(patch_output_dir or DEFAULT_PATCH_DIR) / Path(tiff_path).stem
    output_dir.mkdir(parents=True, exist_ok=True)
    pred_png_path = output_dir / "predicted.png"
    annotation_txt_path = output_dir / "annotations.txt"
    show_pred_panel=config["show_pred_panel"]  and pred_label_path is not None
    import time
    # 打开 TIFF 文件
    with rasterio.open(tiff_path) as dataset:
        img = dataset.read()
        transform_affine = dataset.transform
        crs = dataset.crs
        width = dataset.width
        height = dataset.height
    
    #获得地理转化矩阵
    geo_transformer = None

    if crs and Transformer is not None:
        geo_transformer = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    elif use_latlon:
        print(
            "Warning: TIFF file has no CRS; geographic coordinates will be reported in the original projection."
        )
    # 将图像转换为可显示的格式
    img = img.astype(np.uint8)
    display_img = img.transpose(1, 2, 0)
    time_start = time.time()
    # 载入预测和原始标签
    pred_boxes: List[dict] = []
    if show_pred_panel:
        pred_boxes = _load_boxes(pred_label_path, width, height)
        for idx, box in enumerate(pred_boxes, start=1):
            box["id"] = idx
    if show_pred_panel:
        pred_lines_for_file = _save_panel_image(
            display_img,
            pred_boxes,
            title="Predicted Results",
            save_path=pred_png_path,
            transform_affine=transform_affine,
            geo_transformer=geo_transformer,
            use_latlon=use_latlon,
            edge_color="orange",
            empty_message="no predict label",
            annotate_ids=True,
        )
    else:
        pred_lines_for_file = []
    time_end = time.time()
    print(f"Generated visualization in {time_end - time_start:.2f} seconds.")
    # 组合标注文本到独立的 txt 文件
    annotation_lines: List[str] = []
    if pred_lines_for_file:
        annotation_lines.extend(pred_lines_for_file)
    with annotation_txt_path.open("w", encoding="utf-8") as ann_fp:
        ann_fp.write("\n".join(annotation_lines))
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
    output_tiff=config["output_tiff"] if config["output_tiff"] is not None else False
    output_dir = output_root / image_path.stem
    if not image_path.exists():
        raise FileNotFoundError(f"Input image not found: {image_path}")
    if not weights_path.exists():
        raise FileNotFoundError(f"Model weights not found: {weights_path}")
    output_dir.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(weights_path))
    img = Image.open(image_path)
    img = tiff2png.ensure_png_ready(img).convert("RGB")
    print("weights_path",weights_path)
    import time
    start_time = time.time()
    results = model.predict(
        source=img,
        conf=conf_thres,
        imgsz=img.size,
        device=device,
        verbose=False,
        iou=iou_thres,
        augment=augment,
    )
    end_time = time.time()
    print(f"Inference completed in {end_time - start_time:.2f} seconds.")
    if not results:
        raise RuntimeError("Ultralytics did not return any prediction results.")

    label_path: Optional[Path] = None
    result = results[0]
    with Image.open(image_path) as base_img:
        pred_boxes = prediction_boxes(result)
        if use_label:
            label_path = find_label_file(image_path)


        width, height = base_img.size

    print(f"Detections: {len(pred_boxes)}")
    gt_label_for_vis = str(label_path) if label_path is not None else None
    pred_label_path = (output_dir / image_path.name).with_suffix(".txt")
    pred_boxes_for_label = [
        {"xyxy": (x1, y1, x2, y2), "cls": cls_idx} for x1, y1, x2, y2, _, cls_idx in pred_boxes
    ]
    write_label_file(
        Path(pred_label_path),
        pred_boxes_for_label,
        width,
        height,
    )
    if output_tiff:
        overlay_tiff_path = save_detection_tiff(
            source_tiff=image_path,
            boxes=pred_boxes,
            output_path=output_dir / f"{image_path.stem}_pred.tiff",
        )
        print(f"Detection overlay TIFF saved to: {overlay_tiff_path}")
    else:
        overlay_tiff_path = None
    if visualize:
        print("Generating visualization...")
        show_img(
            tiff_path=str(image_path),
            pred_label_path=str(pred_label_path),
            gt_label_path=gt_label_for_vis,
            use_latlon=use_latlon,
            export_patches=export_patches,
            patch_output_dir=output_root,
            patch_size=patch_size,
            config=config,
        )

    return {
        "image": image_path,
        "weights": weights_path,
        "output": overlay_tiff_path,
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
    parser.add_argument("--no_visual", action="store_true",default=False, help="Disable visualization export.")
    parser.add_argument("--output_tiff",action="store_true",default=False, help="Save overlay TIFF.")
    args = parser.parse_args()
    parser.add_argument("--show_pred_panel",action="store_true",default=True, help="Show prediction panel.")
    config = CONFIG()
    overrides: Dict[str, object] = {
        "test_img_path": args.test_img_path,
        "test_weights_path": args.test_weights_path,
        "test_output_path": args.test_output_path,
        "test_conf": args.test_conf,
        "test_iou": args.test_iou,
        "test_device": args.device,
        "target_patch_size": args.patch_size,
        "output_tiff": args.output_tiff,
        "show_pred_panel": args.show_pred_panel
    }
    config.update_config(**overrides)
    if args.no_label:
        config.update_config(use_label=False)
    import time
    start_time = time.time()
    run_single_test(config, visualize=not args.no_visual)
    end_time = time.time()
    print(f"Total processing time: {end_time - start_time:.2f} seconds.")

if __name__ == "__main__":
    main()
