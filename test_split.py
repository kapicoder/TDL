from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import List, Tuple

import rasterio
from ultralytics import YOLO  # type: ignore
from PIL import Image
from utils.tiff_cut import CutterSettings, cut_tiff
from utils.tiff2png import convert_tiff_to_png, iter_tiff_files
from utils.showtiff_withLabel import visual_img

with open("./config.json", "r", encoding="utf-8") as cf:
    CONFIG = json.load(cf)

TILE_PATTERN = re.compile(r"_r(\d+)_c(\d+)$")


def _resolve_path(raw: str | None) -> Path | None:
    return Path(raw) if raw else None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="将大尺寸TIFF裁剪-转换-推理-合成并可视化预测结果的全流程脚本"
    )
    parser.add_argument("--image", type=Path, default=None, help="需要推理的原始TIFF路径")
    parser.add_argument(
        "--label",
        type=Path,
        default=None,
        help="原始TIFF对应的YOLO标签（执行裁剪时需要）",
    )
    parser.add_argument(
        "--tiles-dir",
        type=Path,
        default=None,
        help="裁剪输出根目录（实际会在其下按大图名称创建子目录）",
    )
    parser.add_argument("--tile-size", type=int, default=None, help="裁剪tile尺寸")
    parser.add_argument("--overlap", type=float, default=None, help="相邻tile重叠率")
    parser.add_argument(
        "--tile-ext",
        type=str,
        default=None,
        help="裁剪tile的图像后缀（默认读取配置）",
    )
    parser.add_argument(
        "--save-empty",
        action="store_true",
        help="强制保留没有目标的tile（默认读取配置）",
    )
    parser.add_argument(
        "--no-save-empty",
        action="store_true",
        help="强制丢弃没有目标的tile（默认读取配置）",
    )
    parser.add_argument(
        "--png-dir",
        type=Path,
        default=None,
        help="PNG tile存放目录（默认 <tiles-dir>/<影像名>/images_png）",
    )
    parser.add_argument("--weights", type=Path, default=None, help="推理所用权重路径")
    parser.add_argument("--conf", type=float, default=0.35, help="推理置信度阈值")
    parser.add_argument("--iou", type=float, default=0.6, help="推理IOU阈值")
    parser.add_argument("--imgsz", type=int, default=640, help="推理输入尺寸")
    parser.add_argument("--batch", type=int, default=16, help="推理批大小")
    parser.add_argument("--device", type=str, default="cuda", help="推理设备")
    parser.add_argument(
        "--half",
        action="store_true",
        help="推理时使用半精度（需GPU支持）",
    )
    parser.add_argument(
        "--pred-dir",
        type=Path,
        default=None,
        help="整图预测标签输出目录（若未指定 --pred-label）",
    )
    parser.add_argument(
        "--pred-label",
        type=Path,
        default=None,
        help="整图预测标签（YOLO格式）的完整输出路径",
    )
    parser.add_argument(
        "--skip-cut",
        action="store_true",
        help="跳过裁剪，直接使用已存在的 tile",
    )
    parser.add_argument(
        "--skip-convert",
        action="store_true",
        help="跳过 tiff->png 转换，直接使用现成PNG",
    )
    parser.add_argument(
        "--skip-predict",
        action="store_true",
        help="跳过模型推理，直接复用已有预测标签",
    )
    parser.add_argument(
        "--overwrite-png",
        action="store_true",
        help="PNG 已存在时强制覆盖",
    )
    parser.add_argument(
        "--no-visualize",
        action="store_true",
        help="不调用 showtiff_withLabel 进行可视化",
    )
    parser.add_argument(
        "--use-latlon",
        action="store_true",
        default=True,
        help="可视化时输出经纬度坐标",
    )
    parser.add_argument(
        "--visualize-original-label",
        action="store_true",
        default=True,
        help="可选：在展示预测结果的同时输出原始标注可视化",
    )
    parser.add_argument(
        "--merge-iou",
        type=float,
        default=0.5,
        help="合并重叠预测时使用的IoU阈值（同类别重叠度高时仅保留置信度更高的一个）",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=True,
        help="输出关键配置摘要，便于确认当前运行参数",
    )
    parser.add_argument(
        "--export-detection-patches",
        action="store_true",
        default=True,
        help="导出每个预测框的目标居中子图",
    )
    parser.add_argument(
        "--patch-output-dir",
        type=Path,
        default=None,
        help="目标子图输出目录（默认读取 visualization.patch_output_dir）",
    )
    parser.add_argument(
        "--patch-size",
        type=int,
        default=None,
        help="目标子图尺寸（像素），默认读取 visualization.target_patch_size",
    )
    return parser.parse_args()


def ensure_tiles(
    image_path: Path,
    label_path: Path,
    tiles_dir: Path,
    tile_size: int,
    overlap_rate: float,
    save_empty: bool,
    output_ext: str,
) -> None:
    stride = max(1, int(tile_size * (1.0 - overlap_rate)))
    settings = CutterSettings(
        image_path=image_path,
        label_path=label_path,
        output_dir=tiles_dir,
        tile_size=tile_size,
        stride=stride,
        save_empty=save_empty,
        output_ext=output_ext,
        overlap_rate=overlap_rate,
    )
    print(
        f"[1/4] 使用 tiff_cut 切分 {image_path.name}，tile={tile_size}，overlap={overlap_rate:.2f}"
    )
    cut_tiff(settings)


def convert_tiles_to_png(tile_image_dir: Path, png_dir: Path, overwrite: bool) -> None:
    if not tile_image_dir.exists():
        raise FileNotFoundError(f"Tile 图像目录不存在: {tile_image_dir}")
    png_dir.mkdir(parents=True, exist_ok=True)
    print(f"[2/4] 将 {tile_image_dir} 中的切片转换为PNG -> {png_dir}")
    tiff_files = list(iter_tiff_files(tile_image_dir, recursive=False))
    if not tiff_files:
        raise FileNotFoundError(
            f"{tile_image_dir} 中未找到任何TIFF切片，请确认切图步骤是否执行。"
        )
    for src in tiff_files:
        rel = src.relative_to(tile_image_dir)
        dst = png_dir / rel.with_suffix(".png")
        convert_tiff_to_png(src, dst, overwrite)


def run_inference(
    weights: Path,
    source_dir: Path,
    conf: float,
    iou: float,
    imgsz: int,
    batch: int,
    device: str,
    half: bool,
):
    if not source_dir.exists():
        source_dir.mkdir(parents=True, exist_ok=True)
    png_files = sorted(source_dir.glob("*.png"))
    if not png_files:
        raise FileNotFoundError(f"{source_dir} 下未找到PNG切片")
    print(f"[3/4] 使用 {weights.name} 对 {len(png_files)} 张PNG切片进行推理")
    model = YOLO(str(weights))
    return model.predict(
        source=str(source_dir),
        conf=conf,
        iou=iou,
        imgsz=imgsz,
        batch=batch,
        device=device,
        half=half,
        save=False,
        verbose=False,
    )


def _parse_tile_offsets(tile_path: Path) -> Tuple[int, int]:
    match = TILE_PATTERN.search(tile_path.stem)
    if not match:
        raise ValueError(f"无法从文件名解析行列偏移: {tile_path.name}")
    return int(match.group(1)), int(match.group(2))


def _normalize_box(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    width: int,
    height: int,
) -> Tuple[float, float, float, float]:
    x1 = max(0.0, min(float(width), x1))
    x2 = max(0.0, min(float(width), x2))
    y1 = max(0.0, min(float(height), y1))
    y2 = max(0.0, min(float(height), y2))
    if x2 <= x1 or y2 <= y1:
        return 0.0, 0.0, 0.0, 0.0
    xc = ((x1 + x2) / 2.0) / width
    yc = ((y1 + y2) / 2.0) / height
    bw = (x2 - x1) / width
    bh = (y2 - y1) / height
    return xc, yc, bw, bh


def _box_iou(box_a: Tuple[float, float, float, float], box_b: Tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
        return 0.0
    inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter_area
    if union <= 0:
        return 0.0
    return inter_area / union


def stitch_predictions(
    results,
    original_width: int,
    original_height: int,
    output_label: Path,
    merge_iou: float,
) -> int:
    output_label.parent.mkdir(parents=True, exist_ok=True)
    print(f"[4/4] 合成整图预测标签 -> {output_label}")
    kept_boxes: List[dict] = []
    for result in results:
        if not getattr(result, "boxes", None):
            continue
        tile_path = Path(result.path)
        row_offset, col_offset = _parse_tile_offsets(tile_path)
        xyxy = result.boxes.xyxy.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy().astype(int)
        confs = result.boxes.conf.cpu().numpy()
        for (x1, y1, x2, y2), cls_id, conf in zip(xyxy, classes, confs):
            gx1 = float(x1) + col_offset
            gy1 = float(y1) + row_offset
            gx2 = float(x2) + col_offset
            gy2 = float(y2) + row_offset
            candidate_box = (gx1, gy1, gx2, gy2)
            merged = False
            for existing in kept_boxes:
                if merge_iou > 0.0 and existing["cls"] == cls_id:
                    if _box_iou(existing["xyxy"], candidate_box) >= merge_iou:
                        merged = True
                        if float(conf) > existing["conf"]:
                            existing["xyxy"] = candidate_box
                            existing["conf"] = float(conf)
                        break
            if not merged:
                kept_boxes.append(
                    {
                        "cls": cls_id,
                        "xyxy": candidate_box,
                        "conf": float(conf),
                    }
                )

    lines: List[str] = []
    total = 0
    for box in kept_boxes:
        xc, yc, bw, bh = _normalize_box(*box["xyxy"], original_width, original_height)
        if bw <= 0 or bh <= 0:
            continue
        lines.append(f"{box['cls']} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")
        total += 1

    output_label.write_text("\n".join(lines), encoding="utf-8")
    print(f"  已写入 {total} 个预测框")
    return total


def load_defaults(args: argparse.Namespace):
    tiff_cfg = CONFIG.get("tiff_cut", {})
    path_cfg = CONFIG.get("path", {})
    test_cfg = CONFIG.get("test", {})
    name=test_cfg.get("test_dataset",{})
    test_cfg=test_cfg.get(f"test_{name}",{})
    image_path = args.image or _resolve_path(test_cfg.get("test_img_path"))
    
    if image_path is None:
        raise ValueError("请通过 --image 或 config.test 指定TIFF路径")
    label_path = args.label 
    
    if label_path is None:
        print(image_path)
        label_path = str(image_path).replace("images","labels").replace(".tiff", ".txt")
        label_path = Path(label_path)
        print("label_path:",label_path)
    tiles_root = args.tiles_dir or _resolve_path(test_cfg.get("output_path")) or Path("./tiff_tiles")
    tiles_dir = tiles_root / image_path.stem
    tile_size = args.tile_size or int(tiff_cfg.get("tile_size", 512))
    overlap = args.overlap
    if overlap is None:
        overlap = float(tiff_cfg.get("overlap_rate", 0.0))
    if not (0.0 <= overlap < 1.0):
        raise ValueError("overlap 参数必须位于 [0, 1) 区间")

    if args.save_empty and args.no_save_empty:
        raise ValueError("--save-empty 与 --no-save-empty 不可同时使用")
    if args.save_empty:
        save_empty = True
    elif args.no_save_empty:
        save_empty = False
    else:
        save_empty = bool(tiff_cfg.get("save_empty", True))

    output_ext = args.tile_ext or tiff_cfg.get("output_ext", ".tiff")
    if not output_ext.startswith("."):
        output_ext = f".{output_ext}"

    png_dir = args.png_dir or (tiles_dir / "images_png")

    weights_path = args.weights
    if weights_path is None:
        weights_path = _resolve_path(test_cfg.get("weights_path")) or _resolve_path(
            CONFIG.get("pretrained_model", {}).get("pretrained_model_path")
        )
    if weights_path is None:
        raise ValueError("请通过 --weights 或配置文件提供模型权重路径")

    pred_dir = args.pred_dir or _resolve_path(path_cfg.get("test_result_path")) or Path("./result/test_result")
    pred_label = args.pred_label or (pred_dir / f"{image_path.stem}_pred.txt")

    return (
        image_path,
        label_path,
        tiles_dir,
        tile_size,
        overlap,
        save_empty,
        output_ext,
        png_dir,
        weights_path,
        pred_label,
    )


def print_run_summary(
    args: argparse.Namespace,
    image_path: Path,
    label_path: Path | None,
    tiles_dir: Path,
    png_dir: Path,
    weights_path: Path,
    pred_label_path: Path,
    tile_size: int,
    overlap_rate: float,
    save_empty: bool,
    output_ext: str,
):
    print("\n===== 当前运行配置 =====")
    print(f"输入TIFF         : {image_path}")
    print(f"原始标签         : {label_path if label_path else '（未指定，仅预测可视化无法展示原标注）'}")
    print(f"切片输出目录     : {tiles_dir}")
    print(f"PNG目录          : {png_dir}")
    print(f"模型权重         : {weights_path}")
    print(f"整图预测输出     : {pred_label_path}")
    print(f"Tile尺寸/重叠    : {tile_size} / {overlap_rate:.2f}")
    print(f"保留空Tile       : {save_empty}")
    print(f"Tile图像后缀     : {output_ext}")
    print(f"推理参数         : conf={args.conf}, iou={args.iou}, imgsz={args.imgsz}, batch={args.batch}, device={args.device}, half={args.half}")
    print(f"合并IoU阈值      : {args.merge_iou}")
    print(f"跳过步骤         : cut={args.skip_cut}, convert={args.skip_convert}, predict={args.skip_predict}")
    print("========================\n")


def main() -> None:
    args = parse_args()
    (
        image_path,
        label_path,
        tiles_dir,
        tile_size,
        overlap_rate,
        save_empty,
        output_ext,
        png_dir,
        weights_path,
        pred_label_path,
    ) = load_defaults(args)
    if args.verbose:
        print_run_summary(
            args=args,
            image_path=image_path,
            label_path=label_path,
            tiles_dir=tiles_dir,
            png_dir=png_dir,
            weights_path=weights_path,
            pred_label_path=pred_label_path,
            tile_size=tile_size,
            overlap_rate=overlap_rate,
            save_empty=save_empty,
            output_ext=output_ext,
        )

    image_path = image_path.resolve()
    if not image_path.exists():
        raise FileNotFoundError(f"输入TIFF不存在: {image_path}")

    # 1) 切图
    if not args.skip_cut:
        if label_path is None:
            raise ValueError("执行切图需要 --label 或配置中的 tiff_cut.source_label")
        ensure_tiles(
            image_path=image_path,
            label_path=label_path,
            tiles_dir=tiles_dir,
            tile_size=tile_size,
            overlap_rate=overlap_rate,
            save_empty=save_empty,
            output_ext=output_ext,
        )
    else:
        print("[跳过] 已根据 --skip-cut 指定跳过裁剪")

    tile_image_dir = tiles_dir / "images"

    # 2) 转PNG
    if not args.skip_convert:
        convert_tiles_to_png(tile_image_dir, png_dir, args.overwrite_png)
    else:
        print("[跳过] 已根据 --skip-convert 指定跳过 PNG 转换")

    # 3) 推理
    results = None
    if not args.skip_predict:
        results = run_inference(
            weights=weights_path,
            source_dir=png_dir,
            conf=args.conf,
            iou=args.iou,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            half=args.half,
        )
    else:
        print("[跳过] 已根据 --skip-predict 指定跳过推理")

    # 4) 合成 + 可视化
    with rasterio.open(image_path) as dataset:
        width, height = dataset.width, dataset.height

    if results is not None:
        stitch_predictions(
            results=results,
            original_width=width,
            original_height=height,
            output_label=pred_label_path,
            merge_iou=args.merge_iou,
        )
    elif not pred_label_path.exists():
        raise FileNotFoundError(
            f"未执行推理且找不到现有预测标签: {pred_label_path}"
        )

    if not args.no_visualize:
        gt_label_for_vis: str | None = None
        if args.visualize_original_label:
            if label_path is None:
                raise ValueError("需要提供 --label 或配置文件中的原始标签路径以展示原始标注结果")
            gt_label_for_vis = str(label_path)
        print(gt_label_for_vis)
        visual_img(
            tiff_path=str(image_path),
            pred_label_path=str(pred_label_path),
            gt_label_path=gt_label_for_vis,
            use_latlon=args.use_latlon,
            block=True,
            export_patches=args.export_detection_patches,
            patch_output_dir=args.patch_output_dir,
            patch_size=args.patch_size,
        )
    else:
        print("[完成] 已生成预测标签，未按照 --no-visualize 进行可视化")


if __name__ == "__main__":
    main()
