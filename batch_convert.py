import argparse
from pathlib import Path
from typing import Dict, List, Optional

from test_single import run_single_test
from utils.config import CONFIG


def parse_args(config: CONFIG) -> argparse.Namespace:
    """Parse CLI arguments with defaults from config.json."""
    default_images_dir = config["batch_test_img_path"]
    parser = argparse.ArgumentParser(
        description="批量对目录中的 TIFF/PNG 进行单图推理与可视化（复用 test_single 逻辑）"
    )
    parser.add_argument(
        "--images-dir",
        type=Path,
        default=default_images_dir,
        help="待推理影像所在目录（默认取 config.test_img_path 的目录）",
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
    parser.add_argument("--patch-size", type=int, default=config["target_patch_size"], help="可视化子图尺寸")
    parser.add_argument("--visualize", action="store_true", help="可视化")
    parser.add_argument("--no-label", action="store_true", help="不加载原始标签")
    return parser.parse_args()


def collect_images(images_dir: Path) -> List[Path]:
    if not images_dir.exists():
        raise FileNotFoundError(f"未找到图像目录: {images_dir}")
    exts = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}
    return sorted(
        p
        for p in images_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in exts
    )

def run_batch_convert(
    config: CONFIG,
) -> List[Dict[str, object]]:
    """
    对指定目录下的影像逐张执行 test_single 的推理与可视化。

    Returns:
        逐张推理的结果列表，每个元素为 run_single_test 的返回字典。
    """
    images_dir: Optional[str] = config["batch_test_img_path"]
    visualize: bool = config["test_visualize"]
    imgs_dir = Path(images_dir) if images_dir is not None else None

    images = collect_images(imgs_dir)
    if not images:
        print(f"目录 {imgs_dir} 下没有符合要求的图像。")
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

    config.update_config(
        batch_test_img_path=args.images_dir,
        test_weights_path=args.weights,
        test_output_path=args.output_dir,
        test_conf=args.conf,
        test_iou=args.iou,
        test_device=args.device,
        target_patch_size=args.patch_size,
        test_visualize=args.visualize,
    )
    if args.no_label:
        config.update_config(use_label=False)

    results = run_batch_convert(config=config)

    if not results:
        return

    print(f"\n批量推理完成，共处理 {len(results)} 张。")
    for item in results:
        print(f"- {item['image']} -> {item['output']}")


if __name__ == "__main__":
    main()
