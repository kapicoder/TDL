from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Dict, List

from PIL import Image
from ultralytics import YOLO  # type: ignore
from test_single import (
    find_label_file,
    prediction_boxes,
    save_detection_tiff,
    show_img,
    write_label_file,
)
from utils.config import CONFIG
from utils import tiff2png

TIFF_EXTS = {".tif", ".tiff"}
"""
批量推理脚本，用于对指定目录下的所有 TIFF 图像进行目标检测推理和可视化。图像送入 test_single.py 中的 run_single_test 函数处理。
支持的图像格式仅包括 TIFF。
并且推理结果只包含整张大图的识别结果，不包含裁剪之后的子图结果，以及原始图像和结果以及标注的可视化。
"""

def parse_args(config: CONFIG) -> argparse.Namespace:
    default_input = config["batch_test_img_path"] or config["test_img_path"]
    default_path = Path(default_input) if default_input else None
    parser = argparse.ArgumentParser(
        description="批量处理 TIFF 文件（支持单文件、目录及嵌套目录）"
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=default_path,
        help="单个 TIFF 文件或包含 TIFF 的目录（支持递归）",
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=config["test_weights_path"],
        help="模型权重路径（默认读取 config.test_weights_path）",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=config["test_output_path"],
        help="输出根目录（默认读取 config.test_output_path）",
    )
    parser.add_argument("--conf", type=float, default=config["test_conf"], help="置信度阈值")
    parser.add_argument("--iou", type=float, default=config["test_iou"], help="IoU 阈值")
    parser.add_argument(
        "--device",
        type=str,
        default=config["test_device"] or config["device"],
        help="推理设备（默认读取 config.test_device 或 device）",
    )
    parser.add_argument(
        "--patch-size",
        type=int,
        default=config["visualization_target_patch_size"],
        help="可视化子图尺寸",
    )
    parser.add_argument("--visualize", action="store_true", help="启用可视化")
    parser.add_argument("--no-visualize", action="store_true", help="关闭可视化")
    parser.add_argument("--no-label", action="store_true", help="不加载原始标签")
    parser.add_argument("--show_pred_panel",default=True, help="Show prediction panel.")
    return parser.parse_args()


def collect_tiff_images(input_path: Path) -> List[Path]:
    if not input_path.exists():
        raise FileNotFoundError(f"未找到输入路径: {input_path}")
    if input_path.is_file():
        if input_path.suffix.lower() in TIFF_EXTS:
            return [input_path]
        raise ValueError(f"输入文件不是 TIFF: {input_path}")
    return sorted(
        p
        for p in input_path.rglob("*")
        if p.is_file() and p.suffix.lower() in TIFF_EXTS
    )


def resolve_visualize(args: argparse.Namespace, config: CONFIG) -> bool:
    visualize = config["test_visualize"] if config["test_visualize"] is not None else False
    if args.visualize:
        visualize = True
    if args.no_visualize:
        visualize = False
    return visualize


def run_batch_test(
    config: CONFIG,
) -> List[Dict[str, object]]:
    input_path=config["batch_test_img_path"]
    images = collect_tiff_images(input_path)
    if not images:
        print(f"目录 {input_path} 下没有 TIFF 文件。")
        return []

    results: List[Dict[str, object]] = []
    batch_start = time.perf_counter()

    visualize=config["test_visualize"]
    weights_path = Path(config["test_weights_path"] or "model/best.pt")
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
    output_root = Path(config["test_output_path"] or "./result/result_single")

    if not weights_path.exists():
        raise FileNotFoundError(f"Model weights not found: {weights_path}")

    model = YOLO(str(weights_path))

    for idx, img_path in enumerate(images, start=1):
        inf_start = time.perf_counter()
        pil_img = Image.open(img_path)
        pil_img = tiff2png.ensure_png_ready(pil_img).convert("RGB")
        imgsz = pil_img.size
        pred_results = model.predict(
            source=pil_img,
            conf=conf_thres,
            device=device,
            verbose=False,
            iou=iou_thres,
            imgsz=imgsz,
            augment=augment,
            half=True,
        )
        inf_time = time.perf_counter() - inf_start
        print(f"[计时] 推理耗时 {inf_time:.2f}s")

        width, height = imgsz
        output_dir = output_root / img_path.stem
        output_dir.mkdir(parents=True, exist_ok=True)

        if not pred_results:
            print(f"[警告] {img_path} 未返回预测结果，跳过。")
            continue
        pred_boxes = prediction_boxes(pred_results[0])
        label_path = find_label_file(img_path) if use_label else None
        gt_label_for_vis = str(label_path) if label_path is not None else None

        pred_label_path = (output_dir / img_path.name).with_suffix(".txt")
        pred_boxes_for_label = [
            {"xyxy": (x1, y1, x2, y2), "cls": cls_idx} for x1, y1, x2, y2, _, cls_idx in pred_boxes
        ]
        write_label_file(
            Path(pred_label_path),
            pred_boxes_for_label,
            width,
            height,
        )
        overlay_tiff_path = None
        if output_tiff:
            overlay_tiff_path = save_detection_tiff(
                source_tiff=img_path,
                boxes=pred_boxes,
                output_path=output_dir / f"{img_path.stem}_pred.tiff",
            )
            print(f"Detection overlay TIFF saved to: {overlay_tiff_path}")
        if visualize:
            show_img(
                tiff_path=str(img_path),
                pred_label_path=str(pred_label_path),
                gt_label_path=gt_label_for_vis,
                use_latlon=use_latlon,
                export_patches=export_patches,
                patch_output_dir=output_root,
                patch_size=patch_size,
                config=config,
            )

        results.append(
            {
                "image": img_path,
                "weights": weights_path,
                "output": overlay_tiff_path,
                "pred_label": Path(pred_label_path),
                "gt_label": label_path,
                "detections": len(pred_boxes),
            }
        )
    print(f"[计时] 批量总耗时 {time.perf_counter() - batch_start:.2f}s")
    return results


def main() -> None:
    config = CONFIG()
    args = parse_args(config)

    if args.input is None:
        raise ValueError("未提供输入路径，请使用 --input 指定 TIFF 文件或目录。")

    config.update_config(
        batch_test_img_path=args.input,
        test_weights_path=args.weights,
        test_output_path=args.output_dir,
        test_conf=args.conf,
        test_iou=args.iou,
        test_device=args.device,
        visualization_target_patch_size=args.patch_size,
        target_patch_size=args.patch_size,
        show_pred_panel=args.show_pred_panel,
    )
    if args.no_label:
        config.update_config(use_label=False)

    visualize = resolve_visualize(args, config)
    results = run_batch_test(config=config, visualize=visualize)
    if not results:
        return

    print(f"\n批量推理完成，共处理 {len(results)} 张。")



if __name__ == "__main__":
    main()
