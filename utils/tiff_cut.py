from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import numpy as np
from PIL import Image
import rasterio
from rasterio.windows import Window
from affine import Affine

with open("./config.json", "r") as cf:
    CONFIG = json.load(cf)


@dataclass
class CutterSettings:
    image_path: Path
    label_path: Path
    output_dir: Path
    tile_size: int
    stride: int
    save_empty: bool
    output_ext: str
    overlap_rate: float

@dataclass
class PixelBox:
    cls_id: int
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def area(self) -> float:
        return max(0.0, (self.x2 - self.x1) * (self.y2 - self.y1))


def _coalesce_path(raw: str | None) -> Path | None:
    return Path(raw) if raw else None


def build_settings(args: argparse.Namespace) -> CutterSettings:
    cfg = CONFIG.get("tiff_cut", {})
    image_path = _coalesce_path(getattr(args, "image", None)) or _coalesce_path(cfg.get("source_image"))
    label_path = _coalesce_path(getattr(args, "label", None)) or _coalesce_path(cfg.get("source_label"))
    output_dir = Path(getattr(args, "output", None) or cfg.get("output_dir", "./tiff_tiles"))

    if image_path is None:
        raise ValueError("Please provide --image or set tiff_cut.source_image in config.json")
    if label_path is None:
        raise ValueError("Please provide --label or set tiff_cut.source_label in config.json")

    tile_size = int(getattr(args, "tile", None) or cfg.get("tile_size"))
    overlap_rate = float(getattr(args,'overlap_rate',None) or cfg.get("overlap_rate", 0.0))
    if not (0.0 <= overlap_rate < 1.0):
        raise ValueError(f"Invalid overlap_rate: {overlap_rate}. Must be in [0.0, 1.0).")
    stride = int(tile_size * (1.0 - overlap_rate))

    save_empty = (
        getattr(args, "save_empty", None)
        if getattr(args, "save_empty", None) is not None
        else bool(cfg.get("save_empty", False))
    )

    output_ext = (getattr(args, "ext", None) or cfg.get("output_ext") or ".png").lower()
    if not output_ext.startswith("."):
        output_ext = f".{output_ext}"

    return CutterSettings(
        image_path=image_path,
        label_path=label_path,
        output_dir=output_dir,
        tile_size=tile_size,
        stride=stride,
        overlap_rate=overlap_rate,
        save_empty=save_empty, 
        output_ext=output_ext,
    )


def read_yolo_labels(label_path: Path, width: int, height: int) -> List[PixelBox]:
    boxes: List[PixelBox] = []
    if not label_path.exists():
        raise FileNotFoundError(f"Label file not found: {label_path}")

    with label_path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            parts = raw.strip().split()
            if len(parts) != 5:
                continue
            try:
                cls_id = int(parts[0])
                xc, yc, bw, bh = map(float, parts[1:])
            except ValueError:
                continue
            box_w = bw * width
            box_h = bh * height
            center_x = xc * width
            center_y = yc * height
            x1 = center_x - box_w / 2.0
            y1 = center_y - box_h / 2.0
            x2 = center_x + box_w / 2.0
            y2 = center_y + box_h / 2.0
            boxes.append(PixelBox(cls_id, x1, y1, x2, y2))
    return boxes

#计算与tile_bounds相交的boxes，并将其转换为相对于tile_bounds的坐标系
def intersect_boxes(
    boxes: Sequence[PixelBox],
    tile_bounds: Tuple[float, float, float, float],
) -> List[Tuple[int, float, float, float, float]]:
    
    x1_tile, y1_tile, x2_tile, y2_tile = tile_bounds
    tile_w = x2_tile - x1_tile
    tile_h = y2_tile - y1_tile
    result: List[Tuple[int, float, float, float, float]] = []

    for box in boxes:
        
        ix1 = max(x1_tile, box.x1)
        iy1 = max(y1_tile, box.y1)
        ix2 = min(x2_tile, box.x2)
        iy2 = min(y2_tile, box.y2)
        if ix2 <= ix1 or iy2 <= iy1:
            continue #该box与tils_bounds没有相交的部位
        cx = ((ix1 + ix2) / 2.0 - x1_tile) / tile_w
        cy = ((iy1 + iy2) / 2.0 - y1_tile) / tile_h
        bw = (ix2 - ix1) / tile_w
        bh = (iy2 - iy1) / tile_h
        result.append((box.cls_id, cx, cy, bw, bh))
    return result


def save_label(label_path: Path, boxes: Iterable[Tuple[int, float, float, float, float]]) -> None:
    lines = [f"{cls_id} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}" for cls_id, xc, yc, bw, bh in boxes]
    label_path.write_text("\n".join(lines), encoding="utf-8")

def save_image(image_path: Path, arr: np.ndarray,meta: dict,transform_matrix) -> None:
    #将数据的meta一同写入tiff文件
    with rasterio.open(
        image_path,
        'w',
        driver='GTiff',
        height=arr.shape[1],
        width=arr.shape[2],
        count=arr.shape[0],
        dtype=arr.dtype,
        crs=meta['crs'],
        transform=transform_matrix,
    ) as dst:
        dst.write(arr)
def cut_tiff(settings: CutterSettings) -> None:
    image_dir = settings.output_dir / "images"
    label_dir = settings.output_dir / "labels"
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)

    dataset = rasterio.open(settings.image_path)
    width, height = dataset.width, dataset.height
    if settings.label_path.exists():
        boxes = read_yolo_labels(settings.label_path, width, height)
    else:
        boxes = []
    total_tiles = 0
    tiles_with_boxes = 0

    for row in range(0, height, settings.stride):
        for col in range(0, width, settings.stride):
            w = min(settings.tile_size, width - col)
            h = min(settings.tile_size, height - row)
            bounds = (col, row, col + w, row + h)
            if settings.label_path.exists():
                tile_boxes = intersect_boxes(boxes, bounds)
            else:
                tile_boxes = []
            if not tile_boxes and not settings.save_empty:
                continue

            window = Window(col, row, w, h)
            tile_img = dataset.read(window=window, boundless=True, fill_value=0)
           
            tile_name = f"{settings.image_path.stem}_r{row:05d}_c{col:05d}"
            image_path = image_dir / f"{tile_name}{settings.output_ext}"
            
            label_path = label_dir / f"{tile_name}.txt"

            save_label(label_path, tile_boxes)
            # 依据切片窗口计算新的仿射变换矩阵，使其对应裁剪后的图块
            new_transform_matrix = rasterio.windows.transform(window, dataset.transform)
            save_image(image_path, tile_img, dataset.meta, new_transform_matrix)
            total_tiles += 1
            if tile_boxes:
                tiles_with_boxes += 1

    dataset.close()
    print(
        f"Finished cutting {settings.image_path.name}: "
        f"{total_tiles} tiles ({tiles_with_boxes} with objects) written to {settings.output_dir}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cut a large TIFF and YOLO labels into fixed-size tiles.")
    parser.add_argument("--image", type=str, help="Path to source TIFF (overrides config).", default=None)
    parser.add_argument("--label", type=str, help="Path to YOLO label file (overrides config).", default=None)
    parser.add_argument("--output", type=str, help="Output directory for tiles.", default=None)
    parser.add_argument("--tile", type=int, help="Tile size in pixels.", default=None)
    parser.add_argument("--stride", type=int, help="Stride in pixels.", default=None)
    parser.add_argument("--min-overlap", type=float, dest="min_overlap", help="Minimum fraction of bbox kept.", default=None)
    parser.add_argument("--ext", type=str, help="Image extension for tiles (e.g. .png/.jpg).", default=None)
    parser.add_argument("--overlap_rate",type=float, help="Fractional overlap between image tiles (e.g. 0.2 = 20% overlap).", default=None)
    parser.add_argument("--save-empty", dest="save_empty", action="store_true", help="Also export tiles without objects.")
    parser.add_argument("--no-save-empty", dest="save_empty", action="store_false", help="Skip tiles without objects.")
    parser.set_defaults(save_empty=None)

    parser.add_argument("--keep-partial", dest="skip_partial", action="store_false", help="Keep tiles smaller than tile size.")
    parser.add_argument("--skip-partial", dest="skip_partial", action="store_true", help="Skip edge tiles smaller than tile size.")
    parser.set_defaults(skip_partial=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = build_settings(args)
    cut_tiff(settings)


if __name__ == "__main__":
    main()
