from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

from test_single import run_single_test
from utils.config import CONFIG

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
    visualize=config["test_visualize"]
    images = collect_tiff_images(input_path)
    if not images:
        print(f"目录 {input_path} 下没有 TIFF 文件。")
        return []

    results: List[Dict[str, object]] = []
    for img_path in images:
        config.update_config(test_img_path=img_path)
        result = run_single_test(config, visualize=visualize)
        results.append(result)
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
    )
    if args.no_label:
        config.update_config(use_label=False)

    visualize = resolve_visualize(args, config)
    results = run_batch_test(config=config)
    if not results:
        return

    print(f"\n批量推理完成，共处理 {len(results)} 张。")
    for item in results:
        print(f"- {item['image']} -> {item['output']}")


if __name__ == "__main__":
    main()
