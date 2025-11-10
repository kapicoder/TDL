from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable
import numpy as np
from PIL import Image

SUPPORTED_TIFF_EXT = (".tif", ".tiff")


def iter_tiff_files(root: Path, recursive: bool) -> Iterable[Path]:
    """Yield TIFF files under root. Uses rglob when recursive=True."""
    pattern = "**/*" if recursive else "*"
    for path in root.glob(pattern):
        if path.is_file() and path.suffix.lower() in SUPPORTED_TIFF_EXT:
            yield path


def ensure_png_ready(img: Image.Image) -> Image.Image:
    """Convert PIL image to a mode that PNG supports broadly."""
    if img.mode in {"RGB", "RGBA", "L", "LA", "I;16"}:
        img_np=np.array(img)
        img_np=img_np.astype(np.uint8)
        img_pil=Image.fromarray(img_np)
        return img_pil
    # Convert palette or multispectral images to RGB
    
    return img.convert("RGB")


def convert_tiff_to_png(src: Path, dst: Path, overwrite: bool) -> None:
    if dst.exists() and not overwrite:
        print(f"[skip] {dst} (exists)")
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(src) as img:
        png_ready = ensure_png_ready(img)
        png_ready.save(dst, format="PNG")
    print(f"[ok] {src} -> {dst}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert all TIFF images in a folder to PNG format."
    )
    parser.add_argument(
        "--input_dir",
        type=Path,
        required=True,
        help="Directory that contains TIFF images.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help="Directory to save PNG images (defaults to input_dir).",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Search for TIFF images in sub-directories too.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite PNG files when they already exist.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir: Path = args.input_dir
    if not input_dir.exists() or not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    output_dir: Path = args.output_dir or input_dir
    recursive: bool = args.recursive

    tiff_files = list(iter_tiff_files(input_dir, recursive))
    if not tiff_files:
        print(f"No TIFF files found under {input_dir}")
        return

    for src in tiff_files:
        rel_path = src.relative_to(input_dir)
        dst = output_dir / rel_path.with_suffix(".png")
        convert_tiff_to_png(src, dst, args.overwrite)


if __name__ == "__main__":
    main()
