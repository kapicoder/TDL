from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import List, Tuple

import numpy as np
import rasterio
from PIL import Image
from rasterio.windows import Window
from ultralytics import YOLO  # type: ignore

from utils.showtiff_withLabel import visual_img
from utils.tiff2png import ensure_png_ready

with open("./config.json", "r", encoding="utf-8") as cf:
    CONFIG = json.load(cf)


def _resolve_path(raw: str | None) -> Path | None:
    return Path(raw) if raw else None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="在线切片+PNG转换+推理合成的快速脚本（不落盘中间tile和PNG）"
    )
    parser.add_argument("--image", type=Path, default=None, help="需要推理的原始TIFF路径")
    parser.add_argument("--label", type=Path, default=None, help="可选：原始标签，用于可视化")
    parser.add_argument("--tile-size", type=int, default=None, help="切片尺寸")
    parser.add_argument("--overlap", type=float, default=None, help="切片重叠率 [0,1)")
    parser.add_argument("--weights", type=Path, default=None, help="模型权重路径")
    parser.add_argument("--conf", type=float, default=0.35, help="推理置信度阈值")
    parser.add_argument("--iou", type=float, default=0.6, help="推理IoU阈值")
    parser.add_argument("--imgsz", type=int, default=640, help="推理输入尺寸")
    parser.add_argument("--batch", type=int, default=32, help="批量推理的tile数量")
    parser.add_argument("--device", type=str, default="cuda", help="推理设备")
    parser.add_argument("--half", action="store_true", help="使用半精度推理（需GPU支持）")
    parser.add_argument("--merge-iou", type=float, default=0.6, help="合并重复预测的IoU阈值")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="输出目录（默认读取 config.test.output_path）",
    )
    parser.add_argument(
        "--pred-label",
        type=Path,
        default=None,
        help="整图预测标签输出路径（默认 <output-dir>/<影像名>_pred.txt）",
    )
    parser.add_argument("--no-visualize", action="store_true", help="跳过可视化与子图导出")
    parser.add_argument(
        "--visualize-original-label",
        action="store_true",
        help="可选：同时展示原始标注",
    )
    parser.add_argument(
        "--use-latlon",
        action="store_true",
        default=True,
        help="可视化时输出经纬度坐标",
    )
    parser.add_argument(
        "--export-detection-patches",
        action="store_true",
        default=True,
        help="导出检测目标居中子图",
    )
    parser.add_argument(
        "--patch-output-dir",
        type=Path,
        default=None,
        help="子图输出目录（默认读取 visualization.patch_output_dir）",
    )
    parser.add_argument(
        "--patch-size",
        type=int,
        default=None,
        help="目标子图尺寸（像素），默认读取 visualization.target_patch_size",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=True,
        help="输出关键配置摘要",
    )
    return parser.parse_args()


def load_defaults(args: argparse.Namespace):
    tiff_cfg = CONFIG.get("tiff_cut", {})
    test_cfg = CONFIG.get("test", {})
    dataset_name = test_cfg.get("test_dataset", "")
    dataset_cfg = test_cfg.get(f"test_{dataset_name}", {})
    vis_cfg = CONFIG.get("visualization", {})

    image_path = args.image or _resolve_path(dataset_cfg.get("test_img_path"))
    if image_path is None:
        raise ValueError("请通过 --image 或 config.test 指定TIFF路径")

    label_path = args.label
    if label_path is None:
        # 与旧脚本保持一致：默认从 images -> labels 推断标签路径
        label_path = Path(
            str(image_path).replace("images", "labels").replace(".tiff", ".txt")
        )

    tile_size = args.tile_size or int(tiff_cfg.get("tile_size", 512))
    overlap_rate = args.overlap
    if overlap_rate is None:
        overlap_rate = float(tiff_cfg.get("overlap_rate", 0.0))
    if not (0.0 <= overlap_rate < 1.0):
        raise ValueError("overlap 参数必须位于 [0, 1) 区间")

    weights_path = args.weights
    if weights_path is None:
        weights_path = _resolve_path(dataset_cfg.get("weights_path")) or _resolve_path(
            CONFIG.get("pretrained_model", {}).get("pretrained_model_path")
        )
    if weights_path is None:
        raise ValueError("请通过 --weights 或配置文件提供模型权重路径")

    output_dir = args.output_dir or _resolve_path(dataset_cfg.get("output_path")) or Path(
        "./result/test_results"
    )
    output_dir = Path(output_dir)

    patch_root = args.patch_output_dir or _resolve_path(vis_cfg.get("patch_output_dir")) or output_dir
    patch_root = Path(patch_root)

    pred_label_path = args.pred_label or (
        patch_root / image_path.stem / f"{image_path.stem}_pred.txt"
    )

    return (
        image_path,
        label_path,
        tile_size,
        overlap_rate,
        weights_path,
        output_dir,
        patch_root,
        pred_label_path,
    )


def print_run_summary(
    args: argparse.Namespace,
    image_path: Path,
    label_path: Path | None,
    tile_size: int,
    overlap_rate: float,
    weights_path: Path,
    pred_label_path: Path,
    output_dir: Path,
    patch_root: Path,
):
    print("\n===== fast_split 运行配置 =====")
    print(f"输入TIFF         : {image_path}")
    print(f"原始标签         : {label_path if label_path else '（未指定，仅预测可视化无法展示原标注）'}")
    print(f"输出根目录       : {output_dir}")
    print(f"标签/可视化目录  : {patch_root}")
    print(f"模型权重         : {weights_path}")
    print(f"整图预测输出     : {pred_label_path}")
    print(f"Tile尺寸/重叠    : {tile_size} / {overlap_rate:.2f}")
    print(
        f"推理参数         : conf={args.conf}, iou={args.iou}, imgsz={args.imgsz}, batch={args.batch}, device={args.device}, half={args.half}"
    )
    print(f"设备：           : {args.device}")
    print(f"合并IoU阈值      : {args.merge_iou}")
    print("==============================\n")


def prepare_tile_for_model(tile_arr: np.ndarray) -> np.ndarray:
    """将 rasterio 读出的 (C,H,W) tile 转成模型可直接推理的 BGR numpy 数组。"""
    if tile_arr.ndim == 3:
        tile_hwc = np.moveaxis(tile_arr, 0, -1)
    else:
        tile_hwc = tile_arr
    if tile_hwc.ndim == 3 and tile_hwc.shape[2] > 4:
        tile_hwc = tile_hwc[:, :, :3]
    if tile_hwc.ndim == 3 and tile_hwc.shape[2] == 1:
        tile_hwc = tile_hwc[:, :, 0]

    png_ready = ensure_png_ready(Image.fromarray(tile_hwc.astype(np.uint8)))
    png_np = np.array(png_ready)

    if png_np.ndim == 2:
        png_np = np.repeat(png_np[:, :, None], 3, axis=2)
    if png_np.shape[2] == 4:
        png_np = png_np[:, :, :3]
    if png_np.shape[2] == 3:
        png_np = png_np[:, :, ::-1]  # RGB -> BGR
    return png_np


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


def _box_iou(box_a: Tuple[float, float, float, float], box_b: Tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
        return 0.0
    inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter_area
    if union <= 0:
        return 0.0
    return inter_area / union


def merge_detections(detections: List[dict], merge_iou: float) -> List[dict]:
    merged: List[dict] = []
    for det in detections:
        merged_flag = False
        for kept in merged:
            if merge_iou > 0.0 and kept["cls"] == det["cls"]:
                if _box_iou(kept["xyxy"], det["xyxy"]) >= merge_iou:
                    merged_flag = True
                    if det["conf"] > kept["conf"]:
                        kept["xyxy"] = det["xyxy"]
                        kept["conf"] = det["conf"]
                    break
        if not merged_flag:
            merged.append(
                {
                    "cls": det["cls"],
                    "conf": det["conf"],
                    "xyxy": det["xyxy"],
                }
            )
    return merged


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
    print(f"[合成] 已写入 {len(lines)} 个预测框 -> {label_path}")
    return len(lines)


def collect_predictions(results, offsets: List[Tuple[int, int]]) -> List[dict]:
    detections: List[dict] = []
    for res, (row_offset, col_offset) in zip(results, offsets):
        if not getattr(res, "boxes", None):
            continue
        xyxy = res.boxes.xyxy.cpu().numpy()
        classes = res.boxes.cls.cpu().numpy().astype(int)
        confs = res.boxes.conf.cpu().numpy()
        for (x1, y1, x2, y2), cls_id, conf in zip(xyxy, classes, confs):
            gx1 = float(x1) + col_offset
            gy1 = float(y1) + row_offset
            gx2 = float(x2) + col_offset
            gy2 = float(y2) + row_offset
            detections.append(
                {
                    "cls": cls_id,
                    "conf": float(conf),
                    "xyxy": (gx1, gy1, gx2, gy2),
                }
            )
    return detections


def run_batch_predict(
    model: YOLO,
    images: List[np.ndarray],
    offsets: List[Tuple[int, int]],
    *,
    conf: float,
    iou: float,
    imgsz: int,
    batch: int,
    device: str,
    half: bool,
) -> List[dict]:
    results = model.predict(
        source=images,
        conf=conf,
        iou=iou,
        imgsz=imgsz,
        batch=batch,
        device=device,
        half=half,
        save=False,
        verbose=False,
    )
    return collect_predictions(results, offsets)


def stream_predict_on_tiles(
    model: YOLO,
    image_path: Path,
    tile_size: int,
    overlap_rate: float,
    *,
    conf: float,
    iou: float,
    imgsz: int,
    batch: int,
    device: str,
    half: bool,
) -> Tuple[List[dict], int, int]:
    stride = max(1, int(tile_size * (1.0 - overlap_rate)))
    detections: List[dict] = []
    with rasterio.open(image_path) as dataset:
        width, height = dataset.width, dataset.height
        est_rows = math.ceil(height / stride)
        est_cols = math.ceil(width / stride)
        est_tiles = est_rows * est_cols
        print(
            f"[fast] 开始在线切片推理，tile={tile_size}，overlap={overlap_rate:.2f}，stride={stride}，预计 {est_tiles} 片"
        )

        batch_imgs: List[np.ndarray] = []
        batch_offsets: List[Tuple[int, int]] = []
        processed = 0
        for row in range(0, height, stride):
            for col in range(0, width, stride):
                w = min(tile_size, width - col)
                h = min(tile_size, height - row)
                window = Window(col, row, w, h)
                tile_arr = dataset.read(window=window, boundless=True, fill_value=0)
                batch_imgs.append(prepare_tile_for_model(tile_arr))
                batch_offsets.append((row, col))
                if len(batch_imgs) >= batch:
                    detections.extend(
                        run_batch_predict(
                            model,
                            batch_imgs,
                            batch_offsets,
                            conf=conf,
                            iou=iou,
                            imgsz=imgsz,
                            batch=batch,
                            device=device,
                            half=half,
                        )
                    )
                    batch_imgs.clear()
                    batch_offsets.clear()
                processed += 1
                if processed % 50 == 0:
                    print(f"  已处理 {processed}/{est_tiles} 个切片")

        if batch_imgs:
            detections.extend(
                run_batch_predict(
                    model,
                    batch_imgs,
                    batch_offsets,
                    conf=conf,
                    iou=iou,
                    imgsz=imgsz,
                    batch=batch,
                    device=device,
                    half=half,
                )
            )
        print(f"[fast] 切片推理完成，共处理 {processed} 片，获得 {len(detections)} 个原始预测")
        return detections, width, height


def main() -> None:
    args = parse_args()
    (
        image_path,
        label_path,
        tile_size,
        overlap_rate,
        weights_path,
        output_dir,
        patch_root,
        pred_label_path,
    ) = load_defaults(args)

    if args.verbose:
        print_run_summary(
            args=args,
            image_path=image_path,
            label_path=label_path,
            tile_size=tile_size,
            overlap_rate=overlap_rate,
            weights_path=weights_path,
            pred_label_path=pred_label_path,
            output_dir=output_dir,
            patch_root=patch_root,
        )

    image_path = image_path.resolve()
    if not image_path.exists():
        raise FileNotFoundError(f"输入TIFF不存在: {image_path}")

    model = YOLO(str(weights_path))
    detections, width, height = stream_predict_on_tiles(
        model=model,
        image_path=image_path,
        tile_size=tile_size,
        overlap_rate=overlap_rate,
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        batch=max(1, args.batch),
        device=args.device,
        half=args.half,
    )

    merged = merge_detections(detections, args.merge_iou)
    write_label_file(pred_label_path, merged, width, height)

    if args.no_visualize:
        print("[完成] 已生成预测标签，未按 --no-visualize 进行可视化")
        return

    gt_label_for_vis: str | None = None
    if args.visualize_original_label:
        gt_label_for_vis = str(label_path)
    visual_img(
        tiff_path=str(image_path),
        pred_label_path=str(pred_label_path),
        gt_label_path=gt_label_for_vis,
        use_latlon=args.use_latlon,
        block=True,
        export_patches=args.export_detection_patches,
        patch_output_dir=patch_root,
        patch_size=args.patch_size,
    )


if __name__ == "__main__":
    main()
