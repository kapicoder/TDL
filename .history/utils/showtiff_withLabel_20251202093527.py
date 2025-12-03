import argparse
import json
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import rasterio

try:
    from pyproj import Transformer
except Exception:  # pragma: no cover - optional dependency
    Transformer = None  # type: ignore

with open("config.json", "r", encoding="utf-8") as cf:
    CONFIG = json.load(cf)
VIS_CFG = CONFIG.get("visualization", {})
DEFAULT_PATCH_SIZE = int(VIS_CFG.get("target_patch_size", 512))
DEFAULT_PATCH_DIR = VIS_CFG.get(
    "patch_output_dir", "./result/test_result"
)


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


def _load_boxes(label_path: str | None, width: int, height: int) -> List[dict]:
    boxes: List[dict] = []
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
            boxes.append(
                {
                    "cls": cls_id,
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                }
            )
    return boxes


def _draw_panel(
    ax,
    image_array: np.ndarray,
    boxes: List[dict],
    title: str,
    transform_affine,
    geo_transformer,
    use_latlon: bool,
    edge_color: str,
    empty_message: str,
    annotate_ids: bool = False,
    id_prefix: str = "#",
    write_info_text: bool = True,
):
    img_for_draw = _prepare_image_for_save(image_array)
    ax.imshow(img_for_draw, cmap="gray")
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
    for idx, box in enumerate(boxes, start=1):
        cls_id = box["cls"]
        x1, y1, x2, y2 = box["x1"], box["y1"], box["x2"], box["y2"]
        rect = plt.Rectangle(
            (x1, y1), x2 - x1, y2 - y1, edgecolor=edge_color, facecolor="none", linewidth=2
        )
        ax.add_patch(rect)
        label_id = box.get("id", idx)
        if annotate_ids:
            ax.text(
                x1,
                max(0, y1 - 8),
                f"{id_prefix}{label_id}",
                color=edge_color,
                fontsize=10,
                weight="bold",
            )
        if use_latlon and geo_transformer is not None:
            center_lon, center_lat = pixel_to_lonlat(
                (x1 + x2) / 2.0, (y1 + y2) / 2.0, transform_affine, geo_transformer
            )
            bbox_info_lines.append(
                f"{id_prefix}{label_id} cls {cls_id}: lat {center_lat:.6f}, lon {center_lon:.6f}"
            )
        elif use_latlon:
            print(f"cls {cls_id}: CRS missing, cannot convert pixel coordinates to lon/lat.")

    if bbox_info_lines and write_info_text:
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
    return bbox_info_lines


def _export_detection_patches(
    image_name: str,
    image_array: np.ndarray,
    boxes: List[dict],
    patch_size: int,
    output_dir: Path,
    transform_affine,
    geo_transformer,
    use_latlon: bool,
):
    if not boxes:
        print("没有可导出的预测框，跳过目标子图输出。")
        return []
    output_root = Path(output_dir)
    detection_dir = output_root / image_name / "detection_patches"
    detection_dir.mkdir(parents=True, exist_ok=True)
    half = patch_size // 2
    height, width, _ = image_array.shape
    rendered_patches = []
    # 生成预测框子图
    for box in boxes:
        idx = box.get("id")
        if idx is None:
            continue
        cx = (box["x1"] + box["x2"]) / 2.0
        cy = (box["y1"] + box["y2"]) / 2.0
        x1_patch = int(max(0, cx - half))
        y1_patch = int(max(0, cy - half))
        x2_patch = int(min(width, max(0,cx + half)))
        y2_patch = int(min(height, cy + half))
        patch = image_array[y1_patch:y2_patch, x1_patch:x2_patch].copy()
        if patch.size == 0:
            continue
        fig, ax = plt.subplots(figsize=(4, 4.5), constrained_layout=True)
        patch_to_show = _prepare_image_for_save(patch)
        ax.imshow(patch_to_show, cmap="gray")
        ax.set_title(f"Detection #{idx}")
        ax.axis("off")
        rect = plt.Rectangle(
            (box["x1"] - x1_patch, box["y1"] - y1_patch),
            box["x2"] - box["x1"],
            box["y2"] - box["y1"],
            edgecolor="red",
            facecolor="none",
            linewidth=2,
        )
        ax.add_patch(rect)
        label_text = f"ID #{idx} | cls {box['cls']}"
        if use_latlon and geo_transformer is not None:
            lon, lat = pixel_to_lonlat(cx, cy, transform_affine, geo_transformer)
            label_text += f" | lat {lat:.4f}, lon {lon:.4f}"
        else:
            label_text += f" | pixel ({cx:.1f}, {cy:.1f})"
        ax.text(
            0.5,
            -0.08,
            label_text,
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=10,
            color="black",
        )
        save_path = detection_dir / f"{image_name}_det{idx}.png"
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        #print(f"[导出] 目标子图 -> {save_path}")
        plt.close(fig)
        rendered_patches.append(
            {
                "img": patch,
                "rect_origin": (box["x1"] - x1_patch, box["y1"] - y1_patch),
                "rect_size": (box["x2"] - box["x1"], box["y2"] - box["y1"]),
                "title": f"Detection #{idx}",
                "caption": label_text,
            }
        )
    return rendered_patches


def _save_panel_image(
    image_array: np.ndarray,
    boxes: List[dict],
    title: str,
    save_path: Path,
    transform_affine,
    geo_transformer,
    use_latlon: bool,
    edge_color: str,
    empty_message: str,
    annotate_ids: bool = False,
    id_prefix: str = "#",
):
    """Render a single panel image and persist it to disk."""
    fig, ax = plt.subplots(figsize=(8, 8))
    bbox_info_lines = _draw_panel(
        ax,
        image_array,
        boxes,
        title=title,
        transform_affine=transform_affine,
        geo_transformer=geo_transformer,
        use_latlon=use_latlon,
        edge_color=edge_color,
        empty_message=empty_message,
        annotate_ids=annotate_ids,
        id_prefix=id_prefix,
        write_info_text=False,
    )
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return bbox_info_lines


def _prepare_image_for_save(img: np.ndarray) -> np.ndarray:
    """Convert image to a format acceptable by plt.imsave."""
    if img.ndim == 3:
        if img.shape[2] == 1:
            return img[:, :, 0]  # squeeze single channel
        if img.shape[2] not in (3, 4):
            return img[:, :, 0]  # fallback to first channel for unsupported channel counts
    return img


def visual_img(
    tiff_path: str,
    pred_label_path: str | None = None,
    gt_label_path: str | None = None,
    use_latlon: bool = False,
    block: bool = True,
    export_patches: bool = False,
    patch_output_dir: str | Path | None = None,
    patch_size: int | None = None
):
    
    # 输出路径定义
    output_dir = Path(patch_output_dir or DEFAULT_PATCH_DIR) / Path(tiff_path).stem
    output_dir.mkdir(parents=True, exist_ok=True)
    store_path = output_dir / "result.png"
    original_png_path = output_dir / "original.png"
    gt_png_path = output_dir / "origin_label.png"
    pred_png_path = output_dir / "predicted.png"
    annotation_txt_path = output_dir / "annotations.txt"
    
    # 打开 TIFF 文件
    with rasterio.open(tiff_path) as dataset:
        img = dataset.read()
        transform_affine = dataset.transform
        crs = dataset.crs
        width = dataset.width
        height = dataset.height
        
    #获得地理转化矩阵
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
        
    # 将图像转换为可显示的格式
    img = img.astype(np.uint8)
    display_img = img.transpose(1, 2, 0)
    
    # 载入预测标签
    show_pred_panel = pred_label_path is not None
    show_original_panel = gt_label_path is not None
    gt_boxes = _load_boxes(gt_label_path, width, height) if gt_label_path else []
    pred_boxes: List[dict] = []
    if show_pred_panel:
        pred_boxes = _load_boxes(pred_label_path, width, height)
        for idx, box in enumerate(pred_boxes, start=1):
            box["id"] = idx
            
    # 计算输出图像轴的数量
    panel_cols = 3 
    if not show_pred_panel :
        panel_cols -= 1
    if not show_original_panel:
        panel_cols -= 1

    # 导出预测标签
    patch_imgs: List[dict] = []
    if export_patches and pred_boxes:
        patch_dir = Path(patch_output_dir or DEFAULT_PATCH_DIR)
        patch_px = int(patch_size or DEFAULT_PATCH_SIZE)
        patch_imgs = _export_detection_patches(
            image_name=Path(tiff_path).stem,
            image_array=display_img,
            boxes=pred_boxes,
            patch_size=patch_px,
            output_dir=patch_dir,
            transform_affine=transform_affine,
            geo_transformer=geo_transformer if use_latlon else None,
            use_latlon=use_latlon,
        )
    
    # 保存原始图像
    display_img_to_save = _prepare_image_for_save(display_img)
    if display_img_to_save.ndim == 2:
        plt.imsave(original_png_path, display_img_to_save, cmap="gray")
    else:
        plt.imsave(original_png_path, display_img_to_save)

    # 创建主图
    fig_main, axes_main = plt.subplots(1, panel_cols, figsize=(6 * panel_cols, 6))
    if panel_cols == 1:
        axes_main = [axes_main]
    else:
        axes_main = list(np.ravel(axes_main))
        
    # 显示原始图像
    ax_original = axes_main[0]
    x_mid = width // 2
    y_mid = height // 2
    ax_original.imshow(_prepare_image_for_save(display_img), cmap="gray")
    ax_original.set_title("original image")
    ax_original.axis("off")
    if use_latlon and geo_transformer is not None:
        longitude, latitude = pixel_to_lonlat(
            x_mid, y_mid, transform_affine, geo_transformer
        )

        print(
            f"Image center @ pixel ({x_mid}, {y_mid}) -> ({latitude:.6f}, {longitude:.6f})"
        )
    elif use_latlon:
        print("Skipping lon/lat annotation because CRS information is unavailable.")
    # 显示原始标签的数据
    if gt_label_path is not None:
        gt_axis_index = 1 if panel_cols > 1 else 0
        ax_gt = axes_main[gt_axis_index]
        _draw_panel(
            ax_gt,
            display_img,
            gt_boxes,
            title="origin label",
            transform_affine=transform_affine,
            geo_transformer=geo_transformer,
            use_latlon=use_latlon,
            edge_color="lime",
            empty_message="no original label",
            write_info_text=False,
        )

    # 显示预测标签
    if show_pred_panel:
        pred_index= 2 if show_original_panel else 1
        ax_pred = axes_main[pred_index]
        _draw_panel(
            ax_pred,
            display_img,
            pred_boxes,
            title="Predicted Results",
            transform_affine=transform_affine,
            geo_transformer=geo_transformer,
            use_latlon=use_latlon,
            edge_color="orange",
            empty_message="no predict label",
            annotate_ids=True,
            write_info_text=False,
        )

    # 保存主图像
    fig_main.savefig(store_path, dpi=300,bbox_inches='tight')
    
    # 显示预测标签
    if patch_imgs:
        plt.show(block=False)
    else:
        plt.show(block=block)

    # 逐份输出单独的标注图像
    gt_lines_for_file: List[str] = []
    if gt_label_path is not None:
        gt_lines_for_file = _save_panel_image(
            display_img,
            gt_boxes,
            title="origin label",
            save_path=gt_png_path,
            transform_affine=transform_affine,
            geo_transformer=geo_transformer,
            use_latlon=use_latlon,
            edge_color="lime",
            empty_message="no original label",
        )
        
    # 逐份输出单独的预测标签
    if show_pred_panel:
        pred_lines_for_file = _save_panel_image(
            display_img,
            pred_boxes,
            title="Predicted Results",
            save_path=pred_png_path,
            transform_affine=transform_affine,
            geo_transformer=geo_transformer,
            use_latlon=use_latlon,
            edge_color="orange",
            empty_message="no predict label",
            annotate_ids=True,
        )
    else:
        pred_lines_for_file = []

    # 组合标注文本到独立的 txt 文件
    annotation_lines: List[str] = []
    if pred_lines_for_file:
        annotation_lines.extend(pred_lines_for_file)
    with annotation_txt_path.open("w", encoding="utf-8") as ann_fp:
        ann_fp.write("\n".join(annotation_lines))

    #输出展示部分识别结果的子图
    if patch_imgs:
        preview = min(4, len(patch_imgs))
        fig_patch, axes_patch = plt.subplots(1, preview, figsize=(5 * preview, 5), constrained_layout=True)
        if preview == 1:
            axes_patch = [axes_patch]
        for ax, patch_info in zip(axes_patch, patch_imgs[:preview]):
            ax.imshow(_prepare_image_for_save(patch_info["img"]), cmap="gray")
            ax.set_title(patch_info["title"])
            ax.axis("off")
            rect = plt.Rectangle(
                patch_info["rect_origin"],
                patch_info["rect_size"][0],
                patch_info["rect_size"][1],
                edgecolor="red",
                facecolor="none",
                linewidth=2,
            )
            ax.add_patch(rect)
            ax.text(
                0.5,
                -0.08,
                patch_info["caption"],
                transform=ax.transAxes,
                ha="center",
                va="top",
                fontsize=10,
            )
        plt.show(block=block)
        plt.close(fig_patch)
    else:
        plt.show(block=block)
    plt.close(fig_main)
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tiff_path", type=str, help="path to tiff image")
    parser.add_argument("--label_path", type=str, help="(兼容参数) 单一标签路径")
    parser.add_argument("--pred_label_path", type=str, help="预测结果标签路径")
    parser.add_argument("--gt_label_path", type=str, help="原始标注标签路径")
    parser.add_argument(
        "--use_latlon", action="store_true", help="whether use lat lon or not"
    )
    parser.add_argument(
        "--export_patches",
        action="store_true",
        help="是否导出每个预测框的目标子图",
    )
    parser.add_argument(
        "--patch_output_dir",
        type=Path,
        default=None,
        help="目标子图输出目录（默认使用配置中的 visualization.patch_output_dir）",
    )
    parser.add_argument(
        "--patch_size",
        type=int,
        default=None,
        help="目标子图尺寸（像素），默认读取配置 visualization.target_patch_size",
    )
    args = parser.parse_args()

    if args.tiff_path is None:
        raise ValueError("请提供 --tiff_path")

    pred_label_path = args.pred_label_path or args.label_path
    gt_label_path = args.gt_label_path
    if gt_label_path is None:
        gt_label_path = args.tiff_path.replace("images", "labels").replace(".tiff", ".txt")
    visual_img(
        tiff_path=args.tiff_path,
        pred_label_path=pred_label_path,
        gt_label_path=gt_label_path,
        use_latlon=args.use_latlon,
        export_patches=args.export_patches,
        patch_output_dir=args.patch_output_dir,
        patch_size=args.patch_size,
    )
