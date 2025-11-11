import argparse
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import rasterio

try:
    from pyproj import Transformer
except Exception:  # pragma: no cover - optional dependency
    Transformer = None  # type: ignore


def pixel_to_lonlat(x_pixel, y_pixel, transform_affine, geo_transformer):
    """Convert pixel coordinates to lon/lat using raster transform/CRS."""
    if transform_affine is None:
        raise ValueError("TIFF file does not contain an affine transform matrix.")
    x_geo, y_geo = transform_affine * (x_pixel, y_pixel)
    if geo_transformer is None:
        # CRS missing; return projected coordinates as best effort.
        return x_geo, y_geo
    lon, lat = geo_transformer.transform(x_geo, y_geo)
    return lon, lat


def _load_boxes(label_path: str | None, width: int, height: int) -> List[Tuple[int, float, float, float, float]]:
    boxes: List[Tuple[int, float, float, float, float]] = []
    if not label_path:
        return boxes
    path = Path(label_path)
    if not path.exists():
        print(f"Warning: label file not found: {label_path}")
        return boxes
    with path.open("r", encoding="utf-8") as handle:
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
            boxes.append((cls_id, x1, y1, x2, y2))
    return boxes


def _draw_panel(
    ax,
    image_array: np.ndarray,
    boxes: List[Tuple[int, float, float, float, float]],
    title: str,
    transform_affine,
    geo_transformer,
    use_latlon: bool,
    edge_color: str,
    empty_message: str,
):
    ax.imshow(image_array, cmap="gray")
    ax.set_title(title)
    ax.axis("off")
    if not boxes:
        ax.text(
            0.5,
            0.5,
            empty_message,
            ha="center",
            va="center",
            color="gray",
            fontsize=12,
            transform=ax.transAxes,
        )
        return

    bbox_info_lines = []
    for cls_id, x1, y1, x2, y2 in boxes:
        rect = plt.Rectangle(
            (x1, y1), x2 - x1, y2 - y1, edgecolor=edge_color, facecolor="none", linewidth=2
        )
        ax.add_patch(rect)
        if use_latlon and geo_transformer is not None:
            center_lon, center_lat = pixel_to_lonlat(
                (x1 + x2) / 2.0, (y1 + y2) / 2.0, transform_affine, geo_transformer
            )
            bbox_info_lines.append(
                f"cls {cls_id}: lat {center_lat:.6f}, lon {center_lon:.6f}"
            )
        elif use_latlon:
            print(f"cls {cls_id}: CRS missing, cannot convert pixel coordinates to lon/lat.")

    if bbox_info_lines:
        ax.text(
            0.5,
            -0.02,
            "\n".join(bbox_info_lines),
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=11,
            color=edge_color,
        )


def visual_img(
    tiff_path: str,
    pred_label_path: str | None = None,
    gt_label_path: str | None = None,
    use_latlon: bool = False,
    block: bool = True,
):
    # 打开 TIFF 文件
    with rasterio.open(tiff_path) as dataset:
        img = dataset.read()
        transform_affine = dataset.transform
        crs = dataset.crs
        width = dataset.width
        height = dataset.height
        dataset_meta = dataset.meta

    geo_transformer = None
    if Transformer is None and use_latlon:
        print(
            "Warning: pyproj is not installed; install it to enable lon/lat conversion "
            "(e.g., `pip install pyproj`)."
        )
    elif crs and Transformer is not None:
        geo_transformer = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    elif use_latlon:
        print(
            "Warning: TIFF file has no CRS; geographic coordinates will be reported in the original projection."
        )

    img = img.astype(np.uint8)
    display_img = img.transpose(1, 2, 0)
    if pred_label_path is not None:
        fig, axes = plt.subplots(1, 3, figsize=(18, 12))
    else:   
        fig, axes = plt.subplots(1, 2, figsize=(12, 8))
    x_mid = width // 2
    y_mid = height // 2
    axes[0].imshow(display_img, cmap="gray")
    axes[0].set_title("original image")
    axes[0].axis("off")
    if use_latlon and geo_transformer is not None:
        longitude, latitude = pixel_to_lonlat(
            x_mid, y_mid, transform_affine, geo_transformer
        )

        print(
            f"Image center @ pixel ({x_mid}, {y_mid}) -> ({latitude:.6f}, {longitude:.6f})"
        )
        axes[0].text(
            0.5,
            -0.03,
            f"center (lat {latitude:.4f}, lon {longitude:.4f})",
            size=12,
            ha="center",
            va="center",
            transform=axes[0].transAxes,
        )
    elif use_latlon:
        print("Skipping lon/lat annotation because CRS information is unavailable.")
    if pred_label_path is not None:
        pred_boxes = _load_boxes(pred_label_path, width, height)
        _draw_panel(
            axes[2],
            display_img,
            pred_boxes,
            title="Predicted Results",
            transform_affine=transform_affine,
            geo_transformer=geo_transformer,
            use_latlon=use_latlon,
            edge_color="orange",
            empty_message="no predict label",
        )

    gt_boxes = _load_boxes(gt_label_path, width, height)
    _draw_panel(
        axes[1],
        display_img,
        gt_boxes,
        title="origin label",
        transform_affine=transform_affine,
        geo_transformer=geo_transformer,
        use_latlon=use_latlon,
        edge_color="lime",
        empty_message="no original label",
    )

    fig.tight_layout()
    plt.show(block=block)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tiff_path", type=str, help="path to tiff image")
    parser.add_argument("--label_path", type=str, help="(兼容参数) 单一标签路径")
    parser.add_argument("--pred_label_path", type=str, help="预测结果标签路径")
    parser.add_argument("--gt_label_path", type=str, help="原始标注标签路径")
    parser.add_argument(
        "--use_latlon", action="store_true", help="whether use lat lon or not"
    )
    args = parser.parse_args()

    if args.tiff_path is None:
        raise ValueError("请提供 --tiff_path")

    pred_label_path = args.pred_label_path or args.label_path
    gt_label_path = args.gt_label_path
    if gt_label_path is None:
        gt_label_path = args.tiff_path.replace("images", "labels").replace(".tiff", ".txt")
    print(gt_label_path)
    visual_img(
        tiff_path=args.tiff_path,
        pred_label_path=pred_label_path,
        gt_label_path=gt_label_path,
        use_latlon=args.use_latlon,
    )
