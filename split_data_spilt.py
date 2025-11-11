from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path
from typing import List, Tuple

SUPPORTED_SUFFIXES = (".png",)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="将 dataset/AIR-SARShip-1.0/data_spilt 中的切片按 8:2 划分 train/test（使用 PNG 图像）"
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("dataset/AIR-SARShip-1.0/data_spilt"),
        help="包含 images/ 与 labels/ 的根目录",
    )
    parser.add_argument(
        "--images-subdir",
        type=str,
        default="images",
        help="PNG 图像所在子目录",
    )
    parser.add_argument(
        "--labels-subdir",
        type=str,
        default="labels",
        help="标签所在子目录",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="输出 train/test 的根目录（默认 root/split）",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="训练集占比",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子，确保可复现",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="若输出目录已存在则先删除",
    )
    return parser.parse_args()


def collect_pairs(images_dir: Path, labels_dir: Path) -> List[Tuple[Path, Path]]:
    if not images_dir.exists():
        raise FileNotFoundError(f"找不到图像目录: {images_dir}")
    if not labels_dir.exists():
        raise FileNotFoundError(f"找不到标签目录: {labels_dir}")

    pairs: List[Tuple[Path, Path]] = []
    for img_path in images_dir.iterdir():
        if img_path.suffix.lower() not in SUPPORTED_SUFFIXES:
            continue
        label_path = labels_dir / f"{img_path.stem}.txt"
        if not label_path.exists():
            print(f"[警告] 缺少标签，跳过: {label_path}")
            continue
        pairs.append((img_path, label_path))
    if not pairs:
        raise RuntimeError("没有收集到有效的图像-标签对。")
    return pairs


def prepare_output_dirs(output_root: Path, clean: bool) -> None:
    if clean and output_root.exists():
        shutil.rmtree(output_root)
    for subset in ("train", "test"):
        (output_root / subset / "images").mkdir(parents=True, exist_ok=True)
        (output_root / subset / "labels").mkdir(parents=True, exist_ok=True)


def copy_pairs(pairs: List[Tuple[Path, Path]], dest_root: Path) -> None:
    for img_path, label_path in pairs:
        subset_dir = dest_root / "images"
        subset_label_dir = dest_root / "labels"
        shutil.copy2(img_path, subset_dir / img_path.name)
        shutil.copy2(label_path, subset_label_dir / label_path.name)


def main() -> None:
    args = parse_args()
    images_dir = args.root / args.images_subdir
    labels_dir = args.root / args.labels_subdir
    output_dir = args.output_dir or (args.root / "split")

    pairs = collect_pairs(images_dir, labels_dir)
    random.seed(args.seed)
    random.shuffle(pairs)

    split_idx = int(len(pairs) * args.train_ratio)
    train_pairs = pairs[:split_idx]
    test_pairs = pairs[split_idx:]

    prepare_output_dirs(output_dir, args.clean)
    copy_pairs(train_pairs, output_dir / "train")
    copy_pairs(test_pairs, output_dir / "test")

    print(f"总样本数: {len(pairs)}")
    print(f"训练集: {len(train_pairs)}（{args.train_ratio*100:.1f}%）")
    print(f"测试集: {len(test_pairs)}（{(1-args.train_ratio)*100:.1f}%）")
    print(f"输出目录: {output_dir}")


if __name__ == "__main__":
    main()
