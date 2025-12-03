from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

from utils.tiff_cut import CutterSettings, cut_tiff
from utils.tiff2png import convert_tiff_to_png, iter_tiff_files

with open("config.json", "r", encoding="utf-8") as cf:
    CONFIG = json.load(cf)


def parse_args() -> argparse.Namespace:
    tiff_cfg = CONFIG.get("tiff_cut", {})
    default_output = Path(tiff_cfg.get("output_dir", "dataset/AIR-SARShip-1.0/data_spilt"))
    parser = argparse.ArgumentParser(
        description="批量裁剪 dataset/AIR-SARShip-1.0/train 中的 TIFF 并转换为 PNG"
    )
    parser.add_argument(
        "--images-dir",
        type=Path,
        default=Path("dataset/AIR-SARShip-1.0/train/images"),
        help="原始 TIFF 所在目录",
    )
    parser.add_argument(
        "--labels-dir",
        type=Path,
        default=Path("dataset/AIR-SARShip-1.0/train/labels"),
        help="与 TIFF 对应的标签目录",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_output,
        help="裁剪结果输出目录（内部会创建 images/ 与 labels/）",
    )
    parser.add_argument(
        "--png-dir",
        type=Path,
        default=None,
        help="PNG 切片输出目录（默认 output-dir/images_png）",
    )
    parser.add_argument(
        "--overwrite-png",
        action="store_true",
        help="PNG 已存在时是否覆盖",
    )
    return parser.parse_args()


def build_settings(image_path: Path, label_path: Path, output_dir: Path) -> CutterSettings:
    cfg = CONFIG.get("tiff_cut", {})
    tile_size = int(cfg.get("tile_size", 512))
    overlap_rate = float(cfg.get("overlap_rate", 0.0))
    if not (0.0 <= overlap_rate < 1.0):
        raise ValueError(f"overlap_rate 必须在 [0,1)，当前为 {overlap_rate}")
    stride = max(1, int(tile_size * (1.0 - overlap_rate)))
    save_empty = True  # 仅保留包含目标的切片，相当于启用 --no-save-empty
    output_ext = cfg.get("output_ext", ".tiff")
    if not output_ext.startswith("."):
        output_ext = f".{output_ext}"
    return CutterSettings(
        image_path=image_path,
        label_path=label_path,
        output_dir=output_dir,
        tile_size=tile_size,
        stride=stride,
        save_empty=save_empty,
        output_ext=output_ext,
        overlap_rate=overlap_rate,
    )


def collect_tiff_images(images_dir: Path) -> List[Path]:
    if not images_dir.exists():
        raise FileNotFoundError(f"未找到图像目录: {images_dir}")
    return sorted(p for p in images_dir.iterdir() if p.suffix.lower() in {".tif", ".tiff"})


def batch_cut(images: List[Path], labels_dir: Path, output_dir: Path) -> None:
    processed = 0
    skipped = 0
    for image_path in images:
        label_path = labels_dir / f"{image_path.stem}.txt"
        if not label_path.exists():
            print(f"[跳过] 找不到标签: {label_path}")
            skipped += 1
            continue
        settings = build_settings(image_path, label_path, output_dir)
        cut_tiff(settings)
        processed += 1
    print(f"裁剪完成：成功 {processed} 张，缺少标签跳过 {skipped} 张。")


def convert_tiles(output_dir: Path, png_dir: Path, overwrite: bool) -> None:
    tile_image_dir = output_dir / "images"
    if not tile_image_dir.exists():
        raise FileNotFoundError(f"未找到切片目录: {tile_image_dir}")
    png_dir.mkdir(parents=True, exist_ok=True)
    tiff_files = list(iter_tiff_files(tile_image_dir, recursive=False))
    if not tiff_files:
        print("未找到需要转换的 TIFF 切片。")
        return
    for src in tiff_files:
        rel = src.relative_to(tile_image_dir)
        dst = png_dir / rel.with_suffix(".png")
        convert_tiff_to_png(src, dst, overwrite)
    print(f"PNG 转换完成，输出目录: {png_dir}")


def main() -> None:
    args = parse_args()
    images = collect_tiff_images(args.images_dir)
    if not images:
        print(f"目录 {args.images_dir} 下没有 TIFF 文件。")
        return
    args.output_dir.mkdir(parents=True, exist_ok=True)
    batch_cut(images, args.labels_dir, args.output_dir)
    png_dir = args.png_dir or (args.output_dir / "images_png")
    convert_tiles(args.output_dir, png_dir, args.overwrite_png)


if __name__ == "__main__":
    main()
