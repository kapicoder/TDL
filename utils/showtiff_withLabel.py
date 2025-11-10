import io
import rasterio
import argparse
import matplotlib.pyplot as plt
import numpy as np

from PIL import Image
from pyproj import Transformer



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


def visual_img(tiff_path, label_path, use_latlon):
    # 打开 TIFF 文件
    with rasterio.open(tiff_path) as dataset:
        # 读取图像数据
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
    fig, axes = plt.subplots(1, 2, figsize=(10, 8))
    img=img.astype(np.uint8)
    axes[0].imshow(img.transpose(1, 2, 0), cmap="gray")
    axes[0].set_title("TIFF origin Image")
    axes[0].axis("off")
    x_mid = width // 2
    y_mid = height // 2
    if use_latlon and geo_transformer is not None:
        longitude, latitude = pixel_to_lonlat(
            x_mid, y_mid, transform_affine, geo_transformer
        )
        print("TIFF metadata:", dataset_meta)
        print("Affine transform:", transform_affine)
        print(
            f"Image center @ pixel ({x_mid}, {y_mid}) -> ({longitude:.6f}, {latitude:.6f})"
        )
        axes[0].text(
            0.5,
            -0.03,
            f"({longitude:.4f}, {latitude:.4f})",
            size=12,
            ha="center",
            va="center",
            transform=axes[0].transAxes,
        )
    elif use_latlon:
        print("Skipping lon/lat annotation because CRS information is unavailable.")
    axes[1].imshow(img.transpose(1, 2, 0), cmap="gray")
    # 读取标签文件并打印内容
    with open(label_path, "r") as label_file:
        for raw in label_file:
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

            rect = plt.Rectangle(
                (x1, y1), box_w, box_h, edgecolor="red", facecolor="none", linewidth=2
            )
            axes[1].add_patch(rect)
            axes[1].set_title("TIFF Image with Label")
            axes[1].axis("off")
            if use_latlon and geo_transformer is not None:
                center_lon, center_lat = pixel_to_lonlat(
                    center_x, center_y, transform_affine, geo_transformer
                )
                top_left_lon, top_left_lat = pixel_to_lonlat(
                    x1, y1, transform_affine, geo_transformer
                )
                bottom_right_lon, bottom_right_lat = pixel_to_lonlat(
                    x2, y2, transform_affine, geo_transformer
                )
                print(
                    f"cls {cls_id} bbox center ({center_x:.1f}, {center_y:.1f}) px -> "
                    f"({center_lon:.6f}, {center_lat:.6f})"
                )
                print(
                    f"    top-left ({x1:.1f}, {y1:.1f}) px -> ({top_left_lon:.6f}, {top_left_lat:.6f})"
                )
                print(
                    f"    bottom-right ({x2:.1f}, {y2:.1f}) px -> ({bottom_right_lon:.6f}, {bottom_right_lat:.6f})"
                )
                axes[1].text(
                    center_x,
                    center_y,
                    f"{center_lon:.4f}\n{center_lat:.4f}",
                    color="yellow",
                    fontsize=6,
                    ha="center",
                    va="center",
                )
            elif use_latlon:
                print(
                    f"cls {cls_id}: CRS missing, cannot convert pixel coordinates to lon/lat."
                )

    plt.tight_layout()
    plt.show()
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tiff_path", type=str, help="path to tiff image")
    parser.add_argument("--label_path", type=str, help="path to label file")
    parser.add_argument(
        "--use_latlon", action="store_true", help="whether use lat lon or not"
    )
    args = parser.parse_args()
    if args.label_path == None:
        args.label_path = args.tiff_path.replace("images", "labels").replace(
            ".tiff", ".txt"
        )
    visual_img(
        tiff_path=args.tiff_path, label_path=args.label_path, use_latlon=args.use_latlon
    )
