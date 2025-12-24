from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
from typing import Iterable, List
import torch
from ultralytics import YOLO  # type: ignore

import fast_split
from fast_split import (
    load_defaults,
    merge_detections,
    print_run_summary,
    stream_predict_on_tiles,
    visual_img,
    write_label_file,
)
"""
批量处理脚本，用于对指定目录下的所有图像进行目标检测推理和可视化。
支持的图像格式包括 TIFF、PNG、JPG 等常见格式。
送入推理的文本是经过裁剪的
"""
def _collect_images(images_dir: Path, extensions: Iterable[str]) -> List[Path]:
    ext_set = {ext.lower() for ext in extensions}
    if images_dir.is_file():
        return [images_dir] if images_dir.suffix.lower() in ext_set else []
    return [
        p
        for p in sorted(images_dir.rglob("*"))
        if p.is_file() and p.suffix.lower() in ext_set
    ]


def _build_fast_split_args(batch_args: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(
        image=None,
        label=None,
        tile_size=batch_args.tile_size,
        overlap=batch_args.overlap,
        weights=batch_args.weights,
        conf=batch_args.conf,
        iou=batch_args.iou,
        imgsz=batch_args.imgsz,
        batch=batch_args.batch,
        device=batch_args.device,
        half=batch_args.half,
        merge_iou=batch_args.merge_iou,
        output_dir=batch_args.output_dir,
        pred_label=None,
        no_visualize=batch_args.no_visualize,
        visualize_original_label=batch_args.visualize_original_label,
        use_latlon=not batch_args.no_use_latlon,
        export_detection_patches=not batch_args.no_export_detection_patches,
        patch_output_dir=batch_args.patch_output_dir,
        patch_size=batch_args.patch_size,
        verbose=not batch_args.quiet,
    )


def batch_split(
    images_dir: Path,
    fast_split_args: argparse.Namespace,
    *,
    labels_dir: Path | None = None,
    extensions: Iterable[str] = (".tif", ".tiff"),
    block_visual: bool = False,
) -> None:
    """
    使用 fast_split 对目录中的所有影像（或单个影像）批量裁剪/推理并输出结果。

    Args:
        images_dir: 需要处理的影像目录。
        fast_split_args: fast_split 期望的参数对象（使用 _build_fast_split_args 构建）。
        labels_dir: 可选，原始标签目录（用于可视化）；未找到会自动跳过。
        extensions: 需要处理的文件扩展名集合。
        block_visual: 是否在可视化时阻塞（批量处理中默认 False）。
    """
    images_dir = images_dir.expanduser().resolve()
    if not images_dir.exists():
        raise FileNotFoundError(f"未找到图像路径: {images_dir}")

    images = _collect_images(images_dir, extensions)
    if not images:
        print(f"{images_dir} 下没有匹配的图像（扩展名 {', '.join(extensions)}）。")
        return

    model: YOLO | None = None
    loaded_weights: Path | None = None
    ok = 0
    failed = 0
    for idx, img_path in enumerate(images, start=1):
        single_args = deepcopy(fast_split_args)
        single_args.image = img_path
        if labels_dir:
            candidate_label = labels_dir / f"{img_path.stem}.txt"
            if candidate_label.exists():
                single_args.label = candidate_label
            elif single_args.visualize_original_label:
                print(f"[警告] 找不到对应标签，跳过原始标注显示: {candidate_label}")

        try:
            (
                resolved_img,
                label_path,
                tile_size,
                overlap_rate,
                weights_path,
                output_dir,
                patch_root,
                pred_label_path,
            ) = load_defaults(single_args)

            if model is None:
                model = YOLO(str(weights_path))
                loaded_weights = weights_path
            elif loaded_weights and weights_path != loaded_weights:
                print(f"[警告] 检测到不同的权重路径 {weights_path}，继续使用已加载的 {loaded_weights}")
            if single_args.device == "cuda":
                if torch.cuda.is_available():
                    pass
                else:
                    single_args.device= "cpu"
                    print(f"[警告] CUDA 设备不可用，将使用 CPU 进行推理。")
            if single_args.verbose:
                print(f"\n=== [{idx}/{len(images)}] 处理 {resolved_img.name} ===")
                print_run_summary(
                    args=single_args,
                    image_path=resolved_img,
                    label_path=label_path,
                    tile_size=tile_size,
                    overlap_rate=overlap_rate,
                    weights_path=weights_path,
                    pred_label_path=pred_label_path,
                    output_dir=output_dir,
                    patch_root=patch_root,
                )

            detections, width, height = stream_predict_on_tiles(
                model=model,
                image_path=resolved_img,
                tile_size=tile_size,
                overlap_rate=overlap_rate,
                conf=single_args.conf,
                iou=single_args.iou,
                imgsz=single_args.imgsz,
                batch=max(1, single_args.batch),
                device=single_args.device,
                half=single_args.half,
            )
            merged = merge_detections(detections, single_args.merge_iou)
            write_label_file(pred_label_path, merged, width, height)

            if single_args.no_visualize:
                print(f"[完成] {resolved_img.name} 已生成预测标签，未执行可视化。")
            else:
                gt_label_for_vis = None
                if single_args.visualize_original_label and label_path and Path(label_path).exists():
                    gt_label_for_vis = str(label_path)
                visual_img(
                    tiff_path=str(resolved_img),
                    pred_label_path=str(pred_label_path),
                    gt_label_path=gt_label_for_vis,
                    use_latlon=single_args.use_latlon,
                    block=block_visual,
                    export_patches=single_args.export_detection_patches,
                    patch_output_dir=patch_root,
                    patch_size=single_args.patch_size,
                )
            ok += 1
        except Exception as exc:
            failed += 1
            print(f"[失败] {img_path}: {exc}")

    print(f"\n批量处理完成：成功 {ok} 张，失败 {failed} 张。")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="批量调用 fast_split 对目录中的影像进行裁剪推理并输出结果"
    )
    parser.add_argument(
        "--images-dir",
        type=Path,
        help="待处理影像所在目录或单个文件（默认尝试读取 config.test 中的 test_img_path 所在目录）",
    )
    parser.add_argument(
        "--labels-dir",
        type=Path,
        default=None,
        help="可选：原始标签目录，文件名需与影像同名",
    )
    parser.add_argument(
        "--ext",
        nargs="+",
        default=[".tif", ".tiff"],
        help="需要处理的文件扩展名列表",
    )
    parser.add_argument("--tile-size", type=int, default=None, help="覆盖 fast_split 的 tile 尺寸")
    parser.add_argument("--overlap", type=float, default=None, help="覆盖 fast_split 的切片重叠率")
    parser.add_argument("--weights", type=Path, default=None, help="模型权重路径")
    parser.add_argument("--conf", type=float, default=0.35, help="推理置信度阈值")
    parser.add_argument("--iou", type=float, default=0.6, help="推理 IoU 阈值")
    parser.add_argument("--imgsz", type=int, default=640, help="推理输入尺寸")
    parser.add_argument("--batch", type=int, default=32, help="tile 推理批大小")
    parser.add_argument("--device", type=str, default="cuda", help="推理设备")
    parser.add_argument("--half", action="store_true", help="使用半精度推理（需 GPU 支持）")
    parser.add_argument("--merge-iou", type=float, default=0.6, help="合并重叠预测的 IoU 阈值")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="覆盖 fast_split 的输出目录（默认为 config.test.output_path）",
    )
    parser.add_argument(
        "--patch-output-dir",
        type=Path,
        default=None,
        help="覆盖目标子图输出根目录（默认为 visualization.patch_output_dir）",
    )
    parser.add_argument("--patch-size", type=int, default=None, help="目标子图尺寸（像素）")
    parser.add_argument("--no-visualize", action="store_true", help="仅生成预测标签，跳过可视化")
    parser.add_argument(
        "--visualize-original-label",
        action="store_true",
        help="如果存在，与预测一起展示原始标注",
    )
    parser.add_argument(
        "--no-use-latlon",
        action="store_true",
        help="禁用可视化时的经纬度坐标输出",
    )
    parser.add_argument(
        "--no-export-detection-patches",
        action="store_true",
        help="不导出预测框对应的目标子图",
    )
    parser.add_argument("--quiet", action="store_true", help="不输出 fast_split 运行摘要")
    parser.add_argument(
        "--block",
        action="store_true",
        help="展示图像时阻塞（默认不阻塞，便于批量处理）",
    )

    args = parser.parse_args()
    if args.images_dir is None:
        parser.error("请通过 --images-dir 指定待处理目录，或在 config.json 中设置 test_img_path。")
    return args


def main() -> None:
    args = parse_args()
    fast_args = _build_fast_split_args(args)
    batch_split(
        images_dir=args.images_dir,
        labels_dir=args.labels_dir,
        extensions=args.ext,
        fast_split_args=fast_args,
        block_visual=args.block,
    )


if __name__ == "__main__":
    main()
